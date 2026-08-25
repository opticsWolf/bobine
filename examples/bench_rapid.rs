//! Benchmark the Rapid fp32 models (layout / OCR det+rec / table) on
//! whatever execution providers the loaded onnxruntime library offers.
//!
//! Usage: bench_rapid  (models + cache paths are the standard test cache;
//! providers come from BOB_ORT_PROVIDERS, default CPU)
//!
//! Compare runs:
//!   CPU: ORT_DYLIB_PATH=<cpu dll> BOB_ORT_PROVIDERS="cpu"
//!   GPU: ORT_DYLIB_PATH=<gpu dll> PATH=<cudnn/cuda> BOB_ORT_PROVIDERS="cuda"

use std::path::Path;
use std::time::Instant;

use bobine::rapid_layout::RapidLayout;
use bobine::rapid_ocr::RapidOcr;
use bobine::rapid_table::RapidTable;

fn synthetic_page(w: u32, h: u32) -> image::DynamicImage {
    // White page with black "text line" bars and figure boxes — enough
    // structure to keep det/layout honest, cheap to build.
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255u8, 255, 255]));
    let mut y = 30u32;
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    while y < h - 60 {
        let lw = 200 + rnd() % (w - 260);
        for yy in y..y + 18 {
            for xx in (30..lw).step_by(2) {
                img.put_pixel(xx, yy, image::Rgb([20, 20, 20]));
            }
        }
        y += 34;
        if rnd() % 7 == 0 {
            let bh = 100 + rnd() % 120;
            for edge in 0..3 {
                for xx in (500..w - 40).step_by(2) {
                    img.put_pixel(xx, y + edge, image::Rgb([0, 0, 0]));
                    img.put_pixel(xx, y + bh - edge, image::Rgb([0, 0, 0]));
                }
            }
            y += bh + 20;
        }
    }
    image::DynamicImage::ImageRgb8(img)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let layout_path = cache.join("doclayout_yolo_docstructbench_imgsz1024.onnx");
    let det_path = cache.join("PP-OCRv4/ch_PP-OCRv4_det_infer.onnx");
    let rec_path = cache.join("PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx");
    let table_path = cache.join("models/TabRec/SlanetPlus/slanet-plus.onnx");

    let providers: Vec<String> = std::env::var("BOB_ORT_PROVIDERS")
        .unwrap_or_else(|_| "cpu".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("providers = {providers:?}");

    let page = match std::env::var("BOB_PAGE_IMG") {
        Ok(p) => image::open(&p)?,
        Err(_) => synthetic_page(1024, 1024),
    };

    // ---- RapidLayout ----
    if layout_path.exists() {
        let t0 = Instant::now();
        let mut m = RapidLayout::load(&layout_path, &providers)?;
        println!("layout load: {:?}", t0.elapsed());
        let warm = m.detect(&page)?;
        if std::env::var("BOB_DEBUG_LABELS").is_ok() {
            for r in &warm {
                println!(
                    "  region {:?} conf={:.2} x={:.0} y={:.0} w={:.0} h={:.0}",
                    r.label,
                    r.confidence,
                    r.x0,
                    r.y0,
                    r.x1 - r.x0,
                    r.y1 - r.y0
                );
            }
        }
        let regions = warm.len();
        let t0 = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            let _ = m.detect(&page)?;
        }
        println!(
            "layout 1024x1024: {:?}/run ({} regions)",
            t0.elapsed() / reps,
            regions
        );
    } else {
        println!("layout model missing, skipped");
    }

    // ---- RapidOCR ----
    if det_path.exists() && rec_path.exists() {
        let t0 = Instant::now();
        let mut m = RapidOcr::load(&det_path, &rec_path, "en", &providers)?;
        println!("ocr load: {:?}", t0.elapsed());
        let _ = m.detect_and_recognize(&page)?; // warmup
        let n = {
            let t0 = Instant::now();
            let lines = m.detect_and_recognize(&page)?;
            println!(
                "ocr 1024x1024 det+rec: {:?}/run ({} lines)",
                t0.elapsed(),
                lines.len()
            );
            lines.len()
        };
        let _ = n;
    } else {
        println!("ocr models missing, skipped");
    }

    // ---- RapidTable ----
    if table_path.exists() {
        let t0 = Instant::now();
        let mut m = RapidTable::load(&table_path, &providers)?;
        println!("table load: {:?}", t0.elapsed());
        // table crop: use a slice of the page (cells get found via ocr_lines)
        let lines = vec![]; // structure-only timing; real cells vary
        for crop_size in [(700u32, 400u32), (1024, 1024)] {
            let crop = page.crop_imm(0, 0, crop_size.0, crop_size.1);
            for _ in 0..5 {
                let _ = m.recognize(&crop, &lines)?; // warmup
            }
            let mut total = std::time::Duration::ZERO;
            let mut best = std::time::Duration::MAX;
            let reps = 30;
            for _ in 0..reps {
                let t0 = Instant::now();
                let _ = m.recognize(&crop, &lines)?;
                let d = t0.elapsed();
                total += d;
                best = best.min(d);
            }
            println!(
                "table {}x{}: {:?}/run avg, {:?} best over {reps} reps",
                crop_size.0,
                crop_size.1,
                total / reps,
                best
            );
        }
    } else {
        println!("table model missing, skipped");
    }

    Ok(())
}
