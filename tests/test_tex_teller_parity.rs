//! fp16-vs-fp32 parity gate for TexTeller (0.7.5 matrix: fp16 matches
//! fp32 on 28/30 CPU and 29/30 CUDA; the gate holds one below the worst
//! measured agreement so a precision regression fails loudly).
//!
//! Run: `BOBINE_TEX_PARITY=1 ORT_DYLIB_PATH=<ort-1.28.1> cargo test
//! --release --test test_tex_teller_parity -- --ignored --nocapture`
//!
//! Needs the labeled eval set (`cargo run --example gen_texteller_eval`:
//! 30 pdflatex-rendered GT formulas in `target/texteller_eval/`) and
//! downloads ~1.7 GB of weights on first run (fp32 pair + fp16 pair).
//! Compares fp16 against fp32 — not against GT — so the gate is pure
//! precision parity: identical inputs, any divergence is the cast.

use std::path::{Path, PathBuf};

#[derive(serde::Deserialize)]
struct Manifest {
    items: Vec<Item>,
}

#[derive(serde::Deserialize)]
struct Item {
    id: String,
    png: String,
}

/// Documented floor: one below the worst measured fp16-vs-fp32
/// agreement (28/30 CPU). The gate is pure precision parity — GT
/// accuracy is the benchmark's job (`docs/benchmarks.md`).
const MIN_PARITY: usize = 27;

#[test]
#[ignore = "needs eval set + ORT dylib + ~1.7GB weights: BOBINE_TEX_PARITY=1"]
fn tex_teller_fp16_parity_with_fp32() {
    if std::env::var("BOBINE_TEX_PARITY").unwrap_or_default() != "1" {
        eprintln!("skip: set BOBINE_TEX_PARITY=1 to run the fp16 parity gate");
        return;
    }
    if std::env::var("ORT_DYLIB_PATH").unwrap_or_default().is_empty() {
        eprintln!("skip: ORT_DYLIB_PATH not set");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/texteller_eval");
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        eprintln!("skip: {} missing (run gen_texteller_eval first)", manifest_path.display());
        return;
    }
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path).expect("read manifest"),
    )
    .expect("parse manifest");
    assert!(!manifest.items.is_empty(), "eval manifest has no items");

    let cache = std::env::var("BOBINE_BENCH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hub_cache_dir());
    let providers: Vec<String> = std::env::var("BOB_ORT_PROVIDERS")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let repo = "opticsWolf/texteller3-onnx";
    let mut fp32 = bobine::TexTeller::from_pretrained(
        repo,
        &cache,
        bobine::ModelPrecision::Fp32,
        &providers,
    )
    .expect("load fp32");
    let mut fp16 = bobine::TexTeller::from_pretrained(
        repo,
        &cache,
        bobine::ModelPrecision::Fp16,
        &providers,
    )
    .expect("load fp16");

    let mut parity = 0usize;
    for item in &manifest.items {
        let img = dir.join(&item.png);
        let a = fp32.recognize(&img).unwrap_or_else(|e| format!("<fp32 ERR {e}>"));
        let b = fp16.recognize(&img).unwrap_or_else(|e| format!("<fp16 ERR {e}>"));
        if a == b {
            parity += 1;
        } else {
            println!("DIVERGE {}:\n  fp32: {a:.150}\n  fp16: {b:.150}", item.id);
        }
    }
    let n = manifest.items.len();
    println!("parity fp16==fp32: {parity}/{n} (GT accuracy is the benchmark's job, not this gate's)");
    assert!(
        parity >= MIN_PARITY.min(n),
        "fp16 parity {parity}/{n} below floor {MIN_PARITY}"
    );
}

/// Shared hub cache: `$HF_HUB_CACHE`, else `~/.cache/huggingface/hub`,
/// else a temp dir. Mirrors the bench harness and the Python default.
fn hub_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(home);
    }
    let home = if cfg!(windows) {
        std::env::var_os("USERPROFILE")
    } else {
        std::env::var_os("HOME")
    };
    if let Some(home) = home {
        return PathBuf::from(home).join(".cache/huggingface/hub");
    }
    std::env::temp_dir().join("bobine_test").join("hub")
}
