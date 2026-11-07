//! POTATR end-to-end parity gate (P1): Rust port vs the torch reference on
//! chem-olive-2024 p5 (21-cell table).
//!
//! Run: `BOBINE_POTATR_PARITY=1 ORT_DYLIB_PATH=<ort-1.28.1> cargo test
//! --release --test test_potatr_parity -- --ignored --nocapture`
//!
//! Downloads `potatr.onnx` (~116 MB) from `opticsWolf/potatr-onnx` on first
//! run. Fixtures in `tests/data/` (all produced by the reference side):
//! - `potatr_gate_chem_p5.png`: PyMuPDF 144-dpi render (the exact pixels the
//!   reference ran — production renders are pdf_oxide's, CID-corrupt on some
//!   papers, so the gate pins the input pixels rather than re-rendering).
//! - `potatr_gate_chem_p5_words.json`: page words from the same builder the
//!   reference used (`ikat_grits_gate.page_words`).
//! - `potatr_gate_chem_p5_ref.json`: torch-side `tables_from_outputs` output.
//!
//! Asserts byte-exact HTML + cell grid/text/header flags + caption, and
//! bboxes within 1e-3 (torch-vs-ORT output parity is 1e-6..6e-5). Compares
//! against the torch reference — not GT — so the gate is pure port parity:
//! identical inputs, any divergence is the port.

use std::path::{Path, PathBuf};

#[derive(serde::Deserialize)]
struct RefWord {
    bbox: [f32; 4],
    text: String,
    block_num: i64,
    line_num: i64,
    span_num: i64,
}

#[derive(serde::Deserialize)]
struct RefCell {
    bbox: [f32; 4],
    column_nums: Vec<usize>,
    row_nums: Vec<usize>,
    header: bool,
    cell_text: String,
}

#[derive(serde::Deserialize)]
struct RefTable {
    html: String,
    cells: Vec<RefCell>,
    caption: String,
    footer: String,
    label: String,
    bbox: [f32; 4],
}

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
    PathBuf::from("target/potatr_hub_cache")
}

#[test]
#[ignore = "needs ORT dylib + ~116MB weights: BOBINE_POTATR_PARITY=1"]
fn potatr_port_matches_torch_on_chem_p5() {
    if std::env::var("BOBINE_POTATR_PARITY").unwrap_or_default() != "1" {
        eprintln!("skip: set BOBINE_POTATR_PARITY=1 to run the POTATR parity gate");
        return;
    }
    if std::env::var("ORT_DYLIB_PATH").unwrap_or_default().is_empty() {
        eprintln!("skip: ORT_DYLIB_PATH not set");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let cache = std::env::var("BOBINE_BENCH_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hub_cache_dir());
    let providers = vec!["CPUExecutionProvider".to_string()];

    let img = image::open(dir.join("potatr_gate_chem_p5.png"))
        .expect("fixture render")
        .to_rgb8();
    let words: Vec<RefWord> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("potatr_gate_chem_p5_words.json")).expect("words"),
    )
    .expect("words parse");
    let expected: Vec<RefTable> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("potatr_gate_chem_p5_ref.json")).expect("ref"),
    )
    .expect("ref parse");

    let tokens: Vec<bobine::rapid_potatr::PotatrObject> = words
        .iter()
        .map(|w| {
            bobine::rapid_potatr::PotatrObject::token(
                &w.text,
                w.bbox,
                w.block_num,
                w.line_num,
                w.span_num,
            )
        })
        .collect();

    let model_path = bobine::rapid_potatr::download_potatr(&cache).expect("download potatr");
    let mut model =
        bobine::rapid_potatr::Potatr::load(&model_path, &providers).expect("load potatr");
    let out = model
        .forward(&image::DynamicImage::ImageRgb8(img))
        .expect("forward");
    let tables = bobine::rapid_potatr::tables_from_outputs(
        &out.node_logits,
        &out.boxes,
        &out.edge_logits,
        &tokens,
        out.geometry.orig_w as f32,
        out.geometry.orig_h as f32,
        0.5,
    );

    assert_eq!(tables.len(), expected.len(), "table count");
    let mut max_box_d = 0.0f32;
    for (got, exp) in tables.iter().zip(expected.iter()) {
        assert_eq!(got.html, exp.html, "byte-exact HTML");
        assert_eq!(got.caption, exp.caption, "caption");
        assert_eq!(got.footer, exp.footer, "footer");
        assert_eq!(got.label, exp.label, "label");
        assert_eq!(got.cells.len(), exp.cells.len(), "cell count");
        for (gc, ec) in got.cells.iter().zip(exp.cells.iter()) {
            assert_eq!(gc.row_nums, ec.row_nums, "row grid");
            assert_eq!(gc.column_nums, ec.column_nums, "column grid");
            assert_eq!(gc.header, ec.header, "header flag");
            assert_eq!(gc.cell_text, ec.cell_text, "cell text");
        }
        for (a, b) in got.bbox.iter().zip(exp.bbox.iter()) {
            let d = (a - b).abs();
            max_box_d = max_box_d.max(d);
            // Boxes ride the resampler delta (P0 gate: 0.0175 post-norm),
            // so sub-pixel drift is expected (measured 0.054px); texts/HTML
            // above are exact. Tolerance 0.5px ≈ 10x margin.
            assert!(d < 0.5, "table bbox within 0.5px: got {a} vs ref {b}");
        }
    }
    println!(
        "potatr parity: {} table(s), {} cells, HTML byte-exact, max box drift {max_box_d:.3}px",
        tables.len(),
        tables.iter().map(|t| t.cells.len()).sum::<usize>()
    );
}
