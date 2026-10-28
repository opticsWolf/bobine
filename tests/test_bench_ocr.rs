//! P-D: TexTeller OCR benchmark vs the ikat ground truth.
//!
//! The ikat auto-proof corpus doubles as a labeled eval set: 10 papers,
//! each with publisher-MathML-derived LaTeX ground truth (TeX-verified)
//! in `ikat-corpus/ground-truth/*.md` and the original PDF in
//! `ikat-corpus/pdf/*.pdf`.
//!
//! Run: `BOBINE_BENCH_OCR=1 ORT_DYLIB_PATH=<ort-1.28.1> cargo test
//! --release --test test_bench_ocr -- --ignored --nocapture`
//! (`BOBINE_BENCH_PAPERS` overrides the paper set, comma-separated;
//! `BOBINE_BENCH_CACHE` overrides the model cache dir.) Published
//! numbers go in `docs/benchmarks.md` (release builds only). GPU (ORT
//! `-gpu_cuda13` build + CUDA/cuDNN on PATH) auto-enables CUDA for the
//! layout/OCR slots.
//!
//! KNOWN BLOCKER (0.7.1): the ikat corpus (PLOS ONE, Type0/CIDFontType2
//! fonts) renders glyph-corrupted under pdf_oxide 0.3.78, so Always-mode
//! OCR output inherits the corruption and accuracy numbers would measure
//! the renderer, not the models. The harness itself is validated (arXiv
//! Type1 PDFs OCR correctly); re-run once the upstream renderer is
//! fixed. See `docs/benchmarks.md` for the full analysis.
//!
//! Re-scoring without re-running OCR: set `BOBINE_BENCH_RESCORE=1` to
//! score the saved `target/bench_ocr/<paper>.md` files (cheap — picks
//! up extractor/scoring fixes).
//!
//! Scoring is normalized exact-match: whitespace is squeezed and
//! `\left`/`\right` are identity (equivalence classes from the
//! proposal). Rendered-pixel diff of mismatches needs a local TeX and
//! is out of scope for the harness — mismatching pairs are printed
//! with their GT labels for follow-up.
//!
//! Routing is forced to `RoutingMode::Always` (full layout + OCR +
//! TexTeller on every page): the default `Auto` takes the text-layer
//! fast path on born-digital pages, which would benchmark the PDF
//! text layer instead of TexTeller.

use std::path::{Path, PathBuf};

// ======================================================================
// Pure scoring logic (always tested, no models needed)
// ======================================================================

/// One display-math span: `$$tex \tag{label}$$` on its own line.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplaySpan {
    /// Normalized tag (`Eq 1`, `1`, …) or None when untagged.
    pub tag: Option<String>,
    /// Raw TeX between the fences (tag stripped).
    pub tex: String,
}

/// Extract `$$…$$` display spans (GT and bobine output share the shape).
/// Tolerates CRLF ground-truth files and multi-line OCR blocks
/// (`$$` alone on a line, tex on following lines, closing `$$`).
pub fn extract_displays(md: &str) -> Vec<DisplaySpan> {
    // Join multi-line `$$` blocks into single logical lines first.
    let mut logical: Vec<String> = Vec::new();
    let mut open = false;
    let mut buf = String::new();
    for line in md.lines() {
        let t = line.trim().trim_end_matches('\r').trim();
        if !open {
            if t.starts_with("$$") {
                if t.len() > 2 && t.ends_with("$$") {
                    logical.push(t.to_string());
                } else {
                    open = true;
                    buf = t.to_string();
                }
            }
        } else {
            buf.push(' ');
            buf.push_str(t);
            if t.ends_with("$$") {
                logical.push(std::mem::take(&mut buf));
                open = false;
            }
        }
    }
    if open && buf.starts_with("$$") {
        // Unterminated fence: salvage rather than drop.
        logical.push(buf);
    }
    let mut out = Vec::new();
    for t in &logical {
        // Strip one fence pair; a missing closer was salvaged above.
        let no_prefix = t.strip_prefix("$$").unwrap_or(t.as_str()).trim();
        let inner = no_prefix.strip_suffix("$$").unwrap_or(no_prefix).trim();
        let (tex, tag) = match split_tag(inner) {
            Some((tex, tag)) => (tex.to_string(), Some(tag.to_string())),
            None => (inner.to_string(), None),
        };
        if tex.is_empty() {
            continue;
        }
        out.push(DisplaySpan { tag, tex });
    }
    out
}

/// Split a trailing `\tag{…}`; the number form `(1)` unwraps to `1`.
fn split_tag(inner: &str) -> Option<(&str, &str)> {
    let t = inner.trim_end();
    if !t.ends_with('}') {
        return None;
    }
    let open = t.rfind("\\tag{")?;
    let tex = t[..open].trim_end();
    let tag = t[open + 5..t.len() - 1].trim();
    if tex.is_empty() || tag.is_empty() {
        return None;
    }
    Some((tex, tag))
}

/// Normalization for exact-match: squeeze whitespace runs, drop
/// `\left`/`\right` (equivalence class — fences scale identically).
pub fn normalize(tex: &str) -> String {
    let mut s = tex.replace("\\left", "").replace("\\right", "");
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
                space = true;
            }
        } else {
            // Drop a pending space before closing brackets and skip
            // spaces right after opening brackets (nil in TeX).
            if space && matches!(c, ')' | ']' | '}') {
                out.pop();
            }
            out.push(c);
            space = matches!(c, '(' | '[' | '{');
        }
    }
    s = out.trim().to_string();
    s
}

/// Order-based pairing (both sides carry the display sequence; GT
/// labels anchor the reporting side). Returns (ocr_idx, gt_idx) pairs
/// plus the unmatched tails.
pub fn align(ocr: &[DisplaySpan], gt: &[DisplaySpan]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n = ocr.len().min(gt.len());
    let pairs: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
    let missing: Vec<usize> = (n..gt.len()).collect();
    let spurious: Vec<usize> = (n..ocr.len()).collect();
    (pairs, missing, spurious)
}

#[derive(Debug)]
pub struct PaperScore {
    pub paper: String,
    pub n_gt: usize,
    pub n_ocr: usize,
    pub exact: usize,
    pub missing: usize,
    pub spurious: usize,
    /// (GT tag, GT tex, OCR tex) for follow-up rendering.
    pub mismatches: Vec<(Option<String>, String, String)>,
}

pub fn score(paper: &str, ocr: &[DisplaySpan], gt: &[DisplaySpan]) -> PaperScore {
    let (pairs, missing, spurious) = align(ocr, gt);
    let mut exact = 0;
    let mut mismatches = Vec::new();
    for (oi, gi) in pairs {
        if normalize(&ocr[oi].tex) == normalize(&gt[gi].tex) {
            exact += 1;
        } else {
            mismatches.push((gt[gi].tag.clone(), gt[gi].tex.clone(), ocr[oi].tex.clone()));
        }
    }
    PaperScore {
        paper: paper.to_string(),
        n_gt: gt.len(),
        n_ocr: ocr.len(),
        exact,
        missing: missing.len(),
        spurious: spurious.len(),
        mismatches,
    }
}

pub fn render_table(scores: &[PaperScore]) -> String {
    let mut s = String::from("| Paper | GT displays | OCR spans | Exact | Missing | Spurious |\n|---|---|---|---|---|---|\n");
    let (mut g, mut e) = (0, 0);
    for sc in scores {
        g += sc.n_gt;
        e += sc.exact;
        let rate = if sc.n_gt > 0 {
            format!(" {:.1}%", 100.0 * sc.exact as f64 / sc.n_gt as f64)
        } else {
            " —".to_string()
        };
        s.push_str(&format!(
            "| {} | {} | {} | {} ({}) | {} | {} |\n",
            sc.paper, sc.n_gt, sc.n_ocr, sc.exact, rate.trim(), sc.missing, sc.spurious
        ));
    }
    let total_rate = if g > 0 {
        format!(" {:.1}%", 100.0 * e as f64 / g as f64)
    } else {
        " —".to_string()
    };
    s.push_str(&format!("| **Total** | {g} |  | {e} ({}) |  |  |\n", total_rate.trim()));
    s
}

// ======================================================================
// Gated end-to-end (needs fixture PDFs + ORT + cached models)
// ======================================================================

fn bench_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/bench_ocr")
}

fn papers() -> Vec<String> {
    if let Ok(list) = std::env::var("BOBINE_BENCH_PAPERS") {
        return list.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    }
    let mut ids: Vec<String> = std::fs::read_dir("ikat-corpus/ground-truth")
        .expect("ikat-corpus/ground-truth missing (git-ignored fixture dir?)")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
        .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    ids.sort();
    ids
}

#[test]
#[ignore = "needs fixture PDFs + ORT dylib + cached models: BOBINE_BENCH_OCR=1"]
fn texsteller_accuracy_ikat_corpus() {
    if std::env::var("BOBINE_BENCH_OCR").unwrap_or_default() != "1" {
        eprintln!("skip: set BOBINE_BENCH_OCR=1 to run the OCR benchmark");
        return;
    }
    if std::env::var("ORT_DYLIB_PATH").unwrap_or_default().is_empty() {
        eprintln!("skip: ORT_DYLIB_PATH not set (needs onnxruntime >= 1.19)");
        return;
    }
    let root = bench_root();
    std::fs::create_dir_all(&root).unwrap();
    let cache = std::env::var("BOBINE_BENCH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_hub_cache_dir());

    let mut scores = Vec::new();
    for paper in papers() {
        let pdf = format!("ikat-corpus/pdf/{paper}.pdf");
        let gt_md = format!("ikat-corpus/ground-truth/{paper}.md");
        if !Path::new(&pdf).exists() {
            eprintln!("skip {paper}: {pdf} missing");
            continue;
        }
        let gt = extract_displays(&std::fs::read_to_string(&gt_md).unwrap());
        let rescore = std::env::var("BOBINE_BENCH_RESCORE").unwrap_or_default() == "1";
        let saved = root.join(format!("{paper}.md"));
        // Full OCR pipeline on every page (see module docs).
        let mut config = bobine::ConverterConfig::default();
        config.routing.routing_mode = bobine::RoutingMode::Always;
        let mut conv = bobine::HybridConverter::new(config, &cache);
        let work = root.join(&paper);
        std::fs::create_dir_all(&work).unwrap();
        let ocr_md = if rescore && saved.exists() {
            println!("{paper}: rescoring saved output");
            Some(std::fs::read_to_string(&saved).unwrap())
        } else {
            match conv.convert_pdf(Path::new(&pdf), &work) {
                Ok(md) => {
                    std::fs::write(root.join(format!("{paper}.md")), &md).unwrap();
                    Some(md)
                }
                Err(e) => {
                    eprintln!("{paper}: convert_pdf failed: {e}");
                    None
                }
            }
        };
        if let Some(md) = ocr_md {
            let ocr = extract_displays(&md);
            let sc = score(&paper, &ocr, &gt);
            println!(
                "{paper}: GT={} OCR={} exact={} missing={} spurious={}",
                sc.n_gt, sc.n_ocr, sc.exact, sc.missing, sc.spurious
            );
            for (tag, g, o) in &sc.mismatches {
                println!("  MISMATCH tag={tag:?}\n    GT : {g:.120}\n    OCR: {o:.120}");
            }
            scores.push(sc);
        }
    }
    scores.sort_by(|a, b| a.paper.cmp(&b.paper));
    let table = render_table(&scores);
    println!("\n{table}");
    std::fs::write(root.join("report.md"), &table).unwrap();
    // GT-label-anchored mismatch dump for rendered-pixel follow-up.
    let mut dump = String::new();
    for sc in &scores {
        for (tag, g, o) in &sc.mismatches {
            dump.push_str(&format!("## {} tag={tag:?}\nGT: {g}\nOCR: {o}\n\n", sc.paper));
        }
    }
    std::fs::write(root.join("mismatches.md"), &dump).unwrap();
}

/// Standard HF hub cache (mirrors the Python binding default).
fn hf_hub_cache_dir() -> PathBuf {
    hf_hub::resolve_cache_dir()
}

// ======================================================================
// Unit tests (no models)
// ======================================================================

#[cfg(test)]
mod scoring_tests {
    use super::*;

    #[test]
    fn extractor_handles_tags_and_crlf() {
        let md = "# T\r\n\r\n$$x^{2} \\tag{1}$$\r\n\r\n$$ y $$\r\n\r\nNot math $x$ inline.\r\n";
        let spans = extract_displays(md);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].tag.as_deref(), Some("1"));
        assert_eq!(spans[0].tex, "x^{2}");
        assert_eq!(spans[1].tag, None);
    }

    #[test]
    fn normalization_equivalence_classes() {
        assert_eq!(normalize("\\left( x \\right)"), normalize("(x)"));
        assert_eq!(normalize("a  b\nc"), "a b c");
        assert_ne!(normalize("x^2"), normalize("x_2"));
    }

    #[test]
    fn alignment_pairs_order_and_reports_tails() {
        let ocr = vec![
            DisplaySpan { tag: None, tex: "a".into() },
            DisplaySpan { tag: None, tex: "b".into() },
            DisplaySpan { tag: None, tex: "EXTRA".into() },
        ];
        let gt = vec![
            DisplaySpan { tag: Some("1".into()), tex: "a".into() },
            DisplaySpan { tag: Some("2".into()), tex: "c".into() },
        ];
        let sc = score("p", &ocr, &gt);
        assert_eq!(sc.exact, 1);
        assert_eq!(sc.missing, 0);
        assert_eq!(sc.spurious, 1);
        assert_eq!(sc.mismatches.len(), 1);
        assert_eq!(sc.mismatches[0].0.as_deref(), Some("2"));
    }

    #[test]
    fn extractor_joins_multiline_ocr_blocks() {
        let md = "text\n$$\nx^{2} + y\n$$\nmore\n$$a \\tag{3}$$\n";
        let spans = extract_displays(md);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].tex, "x^{2} + y");
        assert_eq!(spans[0].tag, None);
        assert_eq!(spans[1].tag.as_deref(), Some("3"));
        // Unterminated fence salvages instead of dropping.
        let spans = extract_displays("$$\nabc");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].tex, "abc");
    }

    #[test]
    fn gt_census_matches_mathml_counts() {
        // Ground-truth display spans per paper (labels verified against
        // the JATS <disp-formula> census, P-B probe).
        let mut total = 0;
        for entry in std::fs::read_dir("ikat-corpus/ground-truth").unwrap() {
            let p = entry.unwrap().path();
            if p.extension().map(|e| e == "md").unwrap_or(false) {
                let md = std::fs::read_to_string(&p).unwrap();
                total += extract_displays(&md).len();
            }
        }
        // Must equal the display-formula count, not the 692 all-math
        // census (inline spans are $…$, not $$…$$).
        println!("GT display total: {total}");
        assert!(total > 100, "sanity: GT has hundreds of display spans");
    }
}
