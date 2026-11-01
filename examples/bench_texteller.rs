//! Benchmark TexTeller formula recognition end-to-end.
//!
//! Usage:
//!   bench_texteller [--fp32|--fp16|--q8|--int8|--all] [--reps N] [--json OUT] <image.png> [...]
//!   bench_texteller --eval <dir> [--fp32|...|--all] [--json OUT]
//!
//! Without a variant flag the fp32 + KV-cache graph is used (historical
//! default). `--all` runs fp32, fp16, q8 and int8 back to back in one
//! process. `--eval` reads the `manifest.json` written by
//! `gen_texteller_eval` and scores every prediction against its GT
//! LaTeX (normalized exact match + char-level similarity).
//!
//! q8 is the experimental MatMulNBits pair from the unified serving repo
//! (external-data decoder); it is benchmark-only and is NOT a serving
//! tier — the repo's parity gate rejects it.
//!
//! Prints the recognized LaTeX plus per-image timing (1 warmup, then
//! `--reps` timed runs, default 3) so variants can be compared for both
//! speed and output parity.

use std::path::{Path, PathBuf};
use std::time::Instant;

use bobine::{ModelPrecision, TexTeller};
use serde::{Deserialize, Serialize};

/// Unified serving repo (fp32/fp16/int8/q8 all resolve here).
const REPO: &str = "opticsWolf/texteller3-onnx";
/// Pinned hub revision (same HEAD serves every variant).
const REV: &str = "5f817336d600ae880205362e16092994aecd01e1";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Fp32,
    Fp16,
    Q8,
    Int8,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Fp32 => "fp32",
            Variant::Fp16 => "fp16",
            Variant::Q8 => "q8",
            Variant::Int8 => "int8",
        }
    }

    fn all() -> [Variant; 4] {
        [Variant::Fp32, Variant::Fp16, Variant::Q8, Variant::Int8]
    }

    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "--fp32" => Some(Variant::Fp32),
            "--fp16" => Some(Variant::Fp16),
            "--q8" => Some(Variant::Q8),
            "--int8" => Some(Variant::Int8),
            _ => None,
        }
    }
}

struct Args {
    variants: Vec<Variant>,
    reps: usize,
    json: Option<PathBuf>,
    eval: Option<PathBuf>,
    images: Vec<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args: Vec<String> = std::env::args().skip(1).collect();
        let mut variants = Vec::new();
        let mut all = false;
        let mut reps = 3usize;
        let mut json = None;
        let mut eval = None;
        let mut images = Vec::new();

        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_str();
            if a == "--all" {
                all = true;
            } else if let Some(v) = Variant::from_flag(a) {
                variants.push(v);
            } else if a == "--reps" {
                i += 1;
                reps = args
                    .get(i)
                    .ok_or("--reps needs a value")?
                    .parse()
                    .map_err(|_| "--reps must be a positive integer")?;
            } else if a == "--json" {
                i += 1;
                json = Some(PathBuf::from(args.get(i).ok_or("--json needs a path")?));
            } else if a == "--eval" {
                i += 1;
                eval = Some(PathBuf::from(
                    args.get(i).ok_or("--eval needs a directory")?,
                ));
            } else if a.starts_with("--") {
                return Err(format!("unknown flag {a}"));
            } else {
                images.push(PathBuf::from(a));
            }
            i += 1;
        }

        if all {
            variants = Variant::all().to_vec();
        }
        if variants.is_empty() {
            variants = vec![Variant::Fp32];
        }
        variants.dedup();

        if eval.is_none() && images.is_empty() {
            return Err(
                "usage: bench_texteller [--fp32|--fp16|--q8|--int8|--all] [--reps N] [--json OUT] <image.png> [...]\n       bench_texteller --eval <dir> [--all] [--json OUT]"
                    .into(),
            );
        }
        if eval.is_some() && !images.is_empty() {
            return Err("--eval and positional images are mutually exclusive".into());
        }
        Ok(Args {
            variants,
            reps: reps.max(1),
            json,
            eval,
            images,
        })
    }
}

/// One prediction, timed. `latex` is kept for parity scoring.
#[derive(Serialize, Clone)]
struct Prediction {
    variant: String,
    provider: String,
    image: String,
    latex: String,
    min_s: f64,
    median_s: f64,
    mean_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct Report {
    provider: String,
    load_s: Vec<(String, f64)>,
    predictions: Vec<Prediction>,
}

fn providers_from_env() -> (Vec<String>, Vec<String>) {
    let dec: Vec<String> = std::env::var("BOBINE_ORT_PROVIDERS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let enc: Vec<String> = std::env::var("BOBINE_ORT_ENCODER_PROVIDERS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (enc, dec)
}

/// Load one variant. q8 is fetched here rather than in the library: the
/// serving surface deliberately has no q8 constructor.
fn load_variant(
    variant: Variant,
    cache: &Path,
    enc: Option<&[String]>,
    dec: &[String],
) -> Result<TexTeller, String> {
    match variant {
        Variant::Fp32 => {
            TexTeller::from_pretrained_split(REPO, cache, ModelPrecision::Fp32, enc, dec)
        }
        Variant::Fp16 => {
            TexTeller::from_pretrained_split(REPO, cache, ModelPrecision::Fp16, enc, dec)
        }
        Variant::Int8 => TexTeller::from_pretrained_int8_split(cache, enc, dec),
        Variant::Q8 => {
            let files = [
                "encoder_model_q8.onnx",
                // Both q8 graphs carry external-data sidecars.
                "encoder_model_q8.onnx.data",
                "decoder_model_merged_q8.onnx",
                "decoder_model_merged_q8.onnx.data",
                "tokenizer.json",
            ];
            let mut paths = Vec::with_capacity(files.len());
            for f in files {
                let p = embroider::fetch_repo_file(REPO, f, Some(REV), Some(cache.to_path_buf()))
                    .map_err(|e| format!("fetch q8 {f}: {e}"))?;
                paths.push(p);
            }
            TexTeller::load_with_provider_split(&paths[0], &paths[2], &paths[4], true, enc, dec)
        }
    }
    .map_err(|e| e.to_string())
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = samples.first().copied().unwrap_or(f64::NAN);
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    (min, median, mean)
}

fn provider_label(dec: &[String]) -> String {
    if dec.is_empty() {
        "default".into()
    } else {
        dec.join(",")
    }
}

// ---------------------------------------------------------------------------
// Accuracy scoring
// ---------------------------------------------------------------------------

/// Formula normalization: strip display fences, `\left`/`\right`,
/// `\limits`, and all whitespace (insignificant in TeX). Stricter than
/// the OCR harness's whitespace squeeze because both sides here are LaTeX.
fn normalize(tex: &str) -> String {
    let mut s = tex
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\limits", "");
    s = s.trim().to_string();
    if let Some(rest) = s.strip_prefix("\\[") {
        s = rest.to_string();
    }
    if let Some(rest) = s.strip_suffix("\\]") {
        s = rest.to_string();
    }
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// 0..=1 char-level similarity on the normalized strings.
fn similarity(gt: &str, pred: &str) -> f64 {
    let a: Vec<char> = normalize(gt).chars().collect();
    let b: Vec<char> = normalize(pred).chars().collect();
    let d = levenshtein(&a, &b);
    1.0 - d as f64 / a.len().max(b.len()).max(1) as f64
}

#[derive(Deserialize)]
struct Manifest {
    items: Vec<ManifestItem>,
}

#[derive(Deserialize, Clone)]
struct ManifestItem {
    id: String,
    paper: String,
    tex: String,
    png: String,
}

struct EvalRow {
    variant: String,
    load_s: f64,
    exact: usize,
    sim: f64,
    median_s: f64,
    mean_s: f64,
    total_s: f64,
    n: usize,
}

fn run_eval(
    args: &Args,
    cache: &Path,
    enc: Option<&[String]>,
    dec: &[String],
) -> Result<Report, Box<dyn std::error::Error>> {
    let dir = args.eval.as_ref().expect("eval dir");
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let items = &manifest.items;
    if items.is_empty() {
        return Err("eval manifest has no items".into());
    }
    println!(
        "eval: {} formulas from {} ({} provider)",
        items.len(),
        dir.display(),
        provider_label(dec)
    );

    let mut report = Report {
        provider: provider_label(dec),
        load_s: Vec::new(),
        predictions: Vec::new(),
    };
    let mut rows: Vec<EvalRow> = Vec::new();

    for &variant in &args.variants {
        let t0 = Instant::now();
        let mut tt = match load_variant(variant, cache, enc, dec) {
            Ok(tt) => tt,
            Err(e) => {
                println!("--- {}: LOAD FAILED: {e}", variant.name());
                continue;
            }
        };
        let load_s = t0.elapsed().as_secs_f64();
        report.load_s.push((variant.name().to_string(), load_s));
        println!("--- {}: loaded in {load_s:.2} s", variant.name());

        // One warmup on the first formula (lazy ORT init is otherwise
        // attributed to the first timed decode). A failing warmup skips
        // the variant instead of killing the whole matrix.
        if let Err(e) = tt.recognize(&dir.join(&items[0].png)) {
            println!("--- {}: warmup failed: {e}", variant.name());
            continue;
        }

        let mut samples = Vec::with_capacity(items.len());
        let mut exact = 0usize;
        let mut sim_sum = 0.0f64;
        let mut total = 0.0f64;

        for (i, item) in items.iter().enumerate() {
            let t = Instant::now();
            let latex = match tt.recognize(&dir.join(&item.png)) {
                Ok(l) => l,
                Err(e) => {
                    // A runtime failure (unsupported op on this EP, etc.)
                    // scores as a miss instead of aborting the whole run.
                    let dt = t.elapsed().as_secs_f64();
                    total += dt;
                    samples.push(dt);
                    report.predictions.push(Prediction {
                        variant: variant.name().to_string(),
                        provider: report.provider.clone(),
                        image: item.id.clone(),
                        latex: String::new(),
                        min_s: dt,
                        median_s: dt,
                        mean_s: dt,
                        error: Some(e.to_string()),
                    });
                    println!(
                        "  [{}/{}] {} {:.2}s ERROR: {}",
                        i + 1,
                        items.len(),
                        item.id,
                        dt,
                        e
                    );
                    continue;
                }
            };
            let dt = t.elapsed().as_secs_f64();
            total += dt;
            samples.push(dt);
            println!(
                "  [{}/{}] {} {:.2}s {} chars",
                i + 1,
                items.len(),
                item.id,
                dt,
                latex.chars().count()
            );

            if normalize(&latex) == normalize(&item.tex) {
                exact += 1;
            }
            sim_sum += similarity(&item.tex, &latex);

            report.predictions.push(Prediction {
                variant: variant.name().to_string(),
                provider: report.provider.clone(),
                image: item.id.clone(),
                latex,
                min_s: dt,
                median_s: dt,
                mean_s: dt,
                error: None,
            });
        }

        let (_, median_s, mean_s) = stats(samples);
        rows.push(EvalRow {
            variant: variant.name().to_string(),
            load_s,
            exact,
            sim: sim_sum / items.len() as f64,
            median_s,
            mean_s,
            total_s: total,
            n: items.len(),
        });
        drop(tt);
    }

    println!();
    println!("| Variant | Load | Formulas | Exact | Similarity | Median | Mean | Total |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {:.2} s | {} | {}/{} ({:.0}%) | {:.3} | {:.2} s | {:.2} s | {:.1} s |",
            r.variant,
            r.load_s,
            r.n,
            r.exact,
            r.n,
            100.0 * r.exact as f64 / r.n as f64,
            r.sim,
            r.median_s,
            r.mean_s,
            r.total_s
        );
    }

    // Parity against fp32 (the house reference tier), when both ran.
    let fp32: Vec<&Prediction> = report
        .predictions
        .iter()
        .filter(|p| p.variant == "fp32")
        .collect();
    if !fp32.is_empty() && rows.len() > 1 {
        println!();
        println!("| Variant vs fp32 | Exact parity | Mean similarity |");
        println!("|---|---|---|");
        for r in rows.iter().filter(|r| r.variant != "fp32") {
            let preds: Vec<&Prediction> = report
                .predictions
                .iter()
                .filter(|p| p.variant == r.variant)
                .collect();
            let mut same = 0usize;
            let mut sim = 0.0f64;
            for (a, b) in fp32.iter().zip(preds.iter()) {
                if normalize(&a.latex) == normalize(&b.latex) {
                    same += 1;
                }
                sim += similarity(&a.latex, &b.latex);
            }
            println!(
                "| {} | {}/{} ({:.0}%) | {:.3} |",
                r.variant,
                same,
                fp32.len(),
                100.0 * same as f64 / fp32.len() as f64,
                sim / fp32.len() as f64
            );
        }
    }

    Ok(report)
}

fn run_images(
    args: &Args,
    cache: &Path,
    enc: Option<&[String]>,
    dec: &[String],
) -> Result<Report, Box<dyn std::error::Error>> {
    let mut report = Report {
        provider: provider_label(dec),
        load_s: Vec::new(),
        predictions: Vec::new(),
    };
    for &variant in &args.variants {
        let t0 = Instant::now();
        let mut tt = match load_variant(variant, cache, enc, dec) {
            Ok(tt) => tt,
            Err(e) => {
                println!("--- {}: LOAD FAILED: {e}", variant.name());
                continue;
            }
        };
        let load_s = t0.elapsed().as_secs_f64();
        println!(
            "model load ({}, dec={:?}, enc={:?}): {load_s:.2} s",
            variant.name(),
            dec,
            if enc.is_none() {
                "<inherit>".to_string()
            } else {
                format!("{enc:?}")
            }
        );
        report.load_s.push((variant.name().to_string(), load_s));

        for path in &args.images {
            let warm = tt.recognize(path)?;
            let mut samples = Vec::with_capacity(args.reps);
            let mut last = String::new();
            for _ in 0..args.reps {
                let s = Instant::now();
                last = tt.recognize(path)?;
                samples.push(s.elapsed().as_secs_f64());
            }
            let (min_s, median_s, mean_s) = stats(samples);
            println!(
                "--- [{}] {} ({min_s:.2} s/run, min)",
                variant.name(),
                path.display()
            );
            println!("warmup: {warm}");
            println!("latex : {last}");
            report.predictions.push(Prediction {
                variant: variant.name().to_string(),
                provider: report.provider.clone(),
                image: path.display().to_string(),
                latex: last,
                min_s,
                median_s,
                mean_s,
                error: None,
            });
        }
        drop(tt);
    }
    Ok(report)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    // Persistent model cache: set BOBINE_CACHE_DIR (e.g. ~/.cache/bobine)
    // so downloads survive reboots; temp-dir cache is wiped by the OS.
    let cache = std::env::var_os("BOBINE_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("bobine_test").join("cache"));

    // BOBINE_ORT_PROVIDERS="cuda,cpu" routes sessions through accelerators;
    // absent/unavailable providers degrade gracefully to CPU.
    let (enc, dec) = providers_from_env();
    let enc_opt = (!enc.is_empty()).then_some(enc.as_slice());

    let report = if args.eval.is_some() {
        run_eval(&args, &cache, enc_opt, &dec)?
    } else {
        run_images(&args, &cache, enc_opt, &dec)?
    };

    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("json: {}", path.display());
    }
    Ok(())
}
