// RapidLayout — DocLayout-YOLO ONNX layout analysis engine.
//
// Detects document regions: title, paragraph, table, figure, equation, etc.
// Model: doclayout_yolo_docstructbench_imgsz1024.onnx (YOLO-style detector)

use std::path::Path;

use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array4};
use ort::{inputs, session::Session};
use tracing::info;

use crate::error::{BobineError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INPUT_SIZE: u32 = 1024;
const CONF_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.45;
const PAD_COLOR: [u8; 3] = [114, 114, 114]; // YOLO gray padding

/// A detected layout region.
#[derive(Debug, Clone)]
pub struct LayoutRegion {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub label: String,
    pub confidence: f32,
}

/// DocLayout-YOLO layout analysis engine.
pub struct RapidLayout {
    session: Session,
    labels: Vec<String>,
}

impl RapidLayout {
    /// Load the DocLayout-YOLO ONNX model from a file path.
    pub fn load(model_path: &Path, providers: &[String]) -> Result<Self> {
        info!("Loading RapidLayout from {}", model_path.display());
        let session = crate::engine::apply_providers(
            Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?,
            providers,
        )?
        .commit_from_file(model_path)
        .map_err(|e| BobineError::Ort(e.to_string()))?;

        // Read label list from ONNX model metadata
        let labels = Self::read_labels(&session).unwrap_or_else(|| {
            vec![
                "title".into(),
                "plain text".into(),
                "abandon".into(),
                "figure".into(),
                "figure_caption".into(),
                "table".into(),
                "table_caption".into(),
                "table_footnote".into(),
                "isolate_formula".into(),
                "formula_caption".into(),
            ]
        });

        info!(
            num_labels = labels.len(),
            labels = ?labels,
            "RapidLayout ready"
        );

        Ok(Self { session, labels })
    }

    /// Read the "character" metadata key from the ONNX model.
    fn read_labels(session: &Session) -> Option<Vec<String>> {
        let meta = session.metadata().ok()?;
        let chars = meta.custom("character")?;
        Some(chars.lines().map(|s| s.trim().to_string()).collect())
    }

    /// Analyze a page image → layout regions.
    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<LayoutRegion>> {
        let orig_h = img.height() as f32;
        let orig_w = img.width() as f32;

        // 1. Preprocess: LetterBox → CHW, normalize
        let tensor = Self::preprocess(img)?; // [1, 3, 1024, 1024]

        // 2. Run ONNX
        let input = ort::value::Tensor::from_array(tensor)
            .map_err(|e| BobineError::Ort(format!("build input: {e}")))?;
        let outputs = self
            .session
            .run(inputs!["images" => input])
            .map_err(|e| BobineError::Ort(format!("run: {e}")))?;

        let preds = outputs["output0"]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("extract output: {e}")))?;

        // preds shape: [1, N, 6] where [x, y, w, h, conf, class_id] (xywh, center-based)
        let preds = preds.slice(s![0, .., ..]); // [N, 6]
        let n = preds.shape()[0];

        // 3. Filter by confidence
        let mut detections: Vec<(f32, f32, f32, f32, f32, usize)> = vec![];
        for i in 0..n {
            let conf = preds[[i, 4]];
            if conf < CONF_THRESHOLD {
                continue;
            }
            let x = preds[[i, 0]];
            let y = preds[[i, 1]];
            let w = preds[[i, 2]];
            let h = preds[[i, 3]];
            let cls = preds[[i, 5]] as usize;

            // Convert xywh → xyxy
            let x0 = x - w / 2.0;
            let y0 = y - h / 2.0;
            let x1 = x + w / 2.0;
            let y1 = y + h / 2.0;

            detections.push((x0, y0, x1, y1, conf, cls));
        }

        // 4. Scale boxes from 1024×1024 back to original image
        let gain = (INPUT_SIZE as f32 / orig_w.max(orig_h)).min(INPUT_SIZE as f32 / orig_h.max(orig_w));
        // Actually LetterBox: we need to compute the actual gain and padding
        let scale = INPUT_SIZE as f32 / orig_w.max(orig_h);
        let new_w = (orig_w * scale).round();
        let new_h = (orig_h * scale).round();
        let pad_w = (INPUT_SIZE as f32 - new_w) / 2.0;
        let pad_h = (INPUT_SIZE as f32 - new_h) / 2.0;

        for (x0, y0, x1, y1, _conf, _) in &mut detections {
            *x0 = (*x0 - pad_w) / scale;
            *y0 = (*y0 - pad_h) / scale;
            *x1 = (*x1 - pad_w) / scale;
            *y1 = (*y1 - pad_h) / scale;
            // Clip to image bounds
            *x0 = x0.max(0.0).min(orig_w);
            *y0 = y0.max(0.0).min(orig_h);
            *x1 = x1.max(0.0).min(orig_w);
            *y1 = y1.max(0.0).min(orig_h);
        }

        // 5. Per-class NMS
        detections.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let keep = multiclass_nms(&detections, IOU_THRESHOLD);

        let regions: Vec<LayoutRegion> = keep
            .iter()
            .map(|&idx| {
                let (x0, y0, x1, y1, conf, cls) = &detections[idx];
                LayoutRegion {
                    x0: *x0,
                    y0: *y0,
                    x1: *x1,
                    y1: *y1,
                    label: self.labels.get(*cls).cloned().unwrap_or_default(),
                    confidence: *conf,
                }
            })
            .collect();

        Ok(regions)
    }

    // ------------------------------------------------------------------
    // Preprocessing: LetterBox resize + BGR→RGB + CHW + normalize
    // ------------------------------------------------------------------

    fn preprocess(img: &DynamicImage) -> Result<Array4<f32>> {
        let (w, h) = (img.width(), img.height());

        // Resize: fit within INPUT_SIZE preserving aspect ratio
        let scale = INPUT_SIZE as f32 / w.max(h) as f32;
        let new_w = (w as f32 * scale).round() as u32;
        let new_h = (h as f32 * scale).round() as u32;

        let resized = image::imageops::resize(
            img,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle, // INTER_LINEAR
        );

        // Pad to INPUT_SIZE×INPUT_SIZE (center, gray 114)
        let pad_w = (INPUT_SIZE - new_w) / 2;
        let pad_h = (INPUT_SIZE - new_h) / 2;

        let mut padded = image::RgbImage::from_pixel(
            INPUT_SIZE,
            INPUT_SIZE,
            image::Rgb(PAD_COLOR),
        );
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x, y);
                padded.put_pixel(pad_w + x, pad_h + y, image::Rgb([p.0[0], p.0[1], p.0[2]]));
            }
        }

        // Convert to CHW float32 [0,1], BGR→RGB (YOLO convention)
        let mut arr = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for y in 0..INPUT_SIZE as usize {
            for x in 0..INPUT_SIZE as usize {
                let p = padded.get_pixel(x as u32, y as u32);
                // OpenCV reads as BGR; we already have RGB from image crate.
                // YOLO models are often trained with BGR input, but DocLayout-YOLO
                // preprocessing in rapid_layout does BGR→RGB via [..., ::-1].
                // We'll use RGB (as-is from image crate) and normalize.
                arr[[0, 0, y, x]] = p.0[0] as f32 / 255.0; // R
                arr[[0, 1, y, x]] = p.0[1] as f32 / 255.0; // G  
                arr[[0, 2, y, x]] = p.0[2] as f32 / 255.0; // B
            }
        }

        Ok(arr)
    }
}

// ---------------------------------------------------------------------------
// NMS helpers
// ---------------------------------------------------------------------------

fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a.2 - a.0) * (a.3 - a.1);
    let area_b = (b.2 - b.0) * (b.3 - b.1);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn multiclass_nms(
    detections: &[(f32, f32, f32, f32, f32, usize)],
    iou_threshold: f32,
) -> Vec<usize> {
    let n = detections.len();
    let mut suppressed = vec![false; n];
    let mut keep = vec![];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }
            if detections[i].5 == detections[j].5 {
                // Same class → check IoU
                let box_i = (detections[i].0, detections[i].1, detections[i].2, detections[i].3);
                let box_j = (detections[j].0, detections[j].1, detections[j].2, detections[j].3);
                if iou(box_i, box_j) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> image::DynamicImage {
        let mut img = image::RgbImage::new(800, 600);
        for y in 0..600 {
            for x in 0..800 {
                img.put_pixel(x, y, image::Rgb([220, 220, 220]));
            }
        }
        image::DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn preprocess_output_shape() {
        let img = test_image();
        let tensor = RapidLayout::preprocess(&img).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn preprocess_values_in_range() {
        let img = test_image();
        let tensor = RapidLayout::preprocess(&img).unwrap();
        let mut found_content = false;
        for v in tensor.iter() {
            assert!(*v >= 0.0 && *v <= 1.0, "value {v} out of [0,1]");
            if *v > 0.1 && *v < 0.99 {
                found_content = true;
            }
        }
        assert!(found_content, "should have non-border pixel values");
    }

    #[test]
    fn iou_identical() {
        let a = (0.0, 0.0, 10.0, 10.0);
        assert!((iou(a, a) - 1.0).abs() < 0.001);
    }

    #[test]
    fn iou_disjoint() {
        assert_eq!(iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 15.0, 15.0)), 0.0);
    }

    #[test]
    fn iou_partial() {
        let v = iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0));
        assert!(v > 0.1 && v < 0.2); // 25 / (100 + 100 - 25) ≈ 0.142
    }

    #[test]
    fn nms_suppresses_overlapping() {
        let dets = vec![
            (0.0, 0.0, 10.0, 10.0, 0.9, 0usize),
            (1.0, 1.0, 9.0, 9.0, 0.8, 0usize),  // highly overlapping
            (20.0, 20.0, 30.0, 30.0, 0.7, 0usize), // disjoint, same class
        ];
        let keep = multiclass_nms(&dets, 0.5);
        assert_eq!(keep.len(), 2); // first + third kept, second suppressed
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
    }
}
