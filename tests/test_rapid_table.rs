// Integration tests for the RapidTable (SLANet-plus) module.
//
// These run the real slanet-plus.onnx (auto-downloaded, ~7.8 MB, cached in
// the system temp dir) and therefore require a working ONNX Runtime:
// set ORT_DYLIB_PATH when the runtime isn't on the default search path.

use std::path::PathBuf;

use bobine::rapid_ocr::OcrLine;
use bobine::rapid_table::RapidTable;

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join("bobine_table_test")
}

/// Render a simple 2x2 table image: white background, black grid lines,
/// and solid "ink" bars where text would sit. Cell geometry is returned
/// so tests can synthesize matching OCR lines.
fn render_table_image() -> (image::DynamicImage, [[f32; 2]; 4]) {
    let w = 400u32;
    let h = 200u32;
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
    // grid: header row 0..100, body row 100..200; cols 0..200, 200..400
    for y in 0..h {
        for x in 0..w {
            if y == 0 || y == 99 || y == 199 || x == 0 || x == 199 || x == 399 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
    }
    // ink bars inside each cell (text-like dark regions)
    let cells = [(10u32, 20u32), (210, 20), (10, 120), (210, 120)];
    for (cx, cy) in cells {
        for dy in 0..40u32 {
            for dx in 0..150u32 {
                img.put_pixel(cx + dx, cy + dy, image::Rgb([30, 30, 30]));
            }
        }
    }
    (
        image::DynamicImage::ImageRgb8(img),
        [[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]],
    )
}

fn ocr_line(x0: f32, y0: f32, x1: f32, y1: f32, text: &str) -> OcrLine {
    OcrLine {
        box_points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        text: text.to_string(),
        confidence: 0.95,
    }
}

#[test]
fn table_model_load_and_recognize() {
    let cache = temp_dir().join("cache");
    std::fs::create_dir_all(&cache).unwrap();

    // Auto-download via engine's default repo by loading through RapidTable
    // with an explicit path after triggering the download through the engine.
    let img_lines = render_table_image();
    let lines = vec![
        ocr_line(10.0, 20.0, 160.0, 60.0, "Name"),
        ocr_line(210.0, 20.0, 360.0, 60.0, "Value"),
        ocr_line(10.0, 120.0, 160.0, 160.0, "alpha"),
        ocr_line(210.0, 120.0, 360.0, 160.0, "42"),
    ];

    // Access engine through converter is private; use OnnxEngine directly.
    let mut engine = bobine::OnnxEngine::new(&bobine::ConverterConfig::default(), &cache);
    engine.set_table_model(
        &bobine::rapid_table::download_slanet_plus(&cache).expect("model download"),
    );
    let html = engine
        .recognize_table(&img_lines.0, &lines)
        .expect("recognize_table failed")
        .expect("no cells decoded");

    assert!(html.contains("<table"), "{html}");
    assert!(html.contains("</tr>"), "{html}");
    eprintln!("HTML: {html}");

    // GFM conversion round-trip via the public tables API.
    let gfm = bobine::html_tables_to_gfm(&html);
    assert!(gfm.contains('|'), "{gfm}");
}
