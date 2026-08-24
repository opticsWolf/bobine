// RapidOCR — minimal text detection + recognition via PaddleOCR ONNX models.
//
// Models: ch_PP-OCRv4_det.onnx (DBNet) + ch_PP-OCRv4_server_rec.onnx (CRNN)
// Preprocessing: resize to multiple-of-32, normalize (x/255 - 0.5)/0.5
// Postprocessing: threshold the segmentation map, find contours, CTC decode

use std::path::Path;

use image::{DynamicImage, GenericImageView, GrayImage};
use ndarray::{Array3, Array4};
use ort::{inputs, session::Session};
use tracing::info;

use crate::error::{BobineError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DET_LIMIT_SIDE_LEN: u32 = 960;
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];  // ImageNet-style
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const DET_THRESH: f32 = 0.3;
const DET_BOX_THRESH: f32 = 0.5;
const REC_IMG_H: u32 = 48;
const REC_IMG_C: u32 = 3;

/// A detected text region.
#[derive(Debug, Clone)]
pub struct OcrLine {
    /// Box as [x0, y0, x1, y1, x2, y2, x3, y3] (4 corners, clockwise from top-left)
    pub box_points: [[f32; 2]; 4],
    pub text: String,
    pub confidence: f32,
}

/// Minimal RapidOCR engine (text detection + recognition).
pub struct RapidOcr {
    det_session: Session,
    rec_session: Session,
    /// Character list for CTC decoding (defaults to English charset)
    characters: Vec<String>,
}

/// CTC charset fallback when the rec model has no `character` metadata.
///
/// Only English/Latin is embedded; other languages rely on the model's
/// metadata (RapidAI ONNX exports embed their dict) and otherwise degrade
/// to ASCII with a warning at load time.
fn fallback_charset(lang: &str) -> Vec<String> {
    match lang.to_lowercase().as_str() {
        "en" | "latin" | "" => (b' '..=b'~').map(|b| String::from(b as char)).collect(),
        other => {
            tracing::warn!(
                lang = other,
                "no embedded charset for this language; \
                 relying on rec-model metadata (falling back to ASCII if absent)"
            );
            (b' '..=b'~').map(|b| String::from(b as char)).collect()
        }
    }
}

impl RapidOcr {
    pub fn load(det_model: &Path, rec_model: &Path, ocr_lang: &str, providers: &[String]) -> Result<Self> {
        info!("Loading RapidOCR det from {}", det_model.display());
        let det_session = crate::engine::apply_providers(
            Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?,
            providers,
        )?
        .commit_from_file(det_model)
        .map_err(|e| BobineError::Ort(e.to_string()))?;

        info!("Loading RapidOCR rec from {}", rec_model.display());
        let rec_session = crate::engine::apply_providers(
            Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?,
            providers,
        )?
        .commit_from_file(rec_model)
        .map_err(|e| BobineError::Ort(e.to_string()))?;

        // Charset: model metadata first, then language fallback
        let characters: Vec<String> = rec_session
            .metadata()
            .ok()
            .and_then(|m| m.custom("character"))
            .map(|c| c.lines().map(|s| s.to_string()).collect())
            .unwrap_or_else(|| fallback_charset(ocr_lang));

        info!(
            num_chars = characters.len(),
            "RapidOCR ready"
        );

        Ok(Self { det_session, rec_session, characters })
    }

    /// Run OCR on an image → text lines with bounding boxes.
    pub fn detect_and_recognize(&mut self, img: &DynamicImage) -> Result<Vec<OcrLine>> {
        // 1. Text detection
        let boxes = self.detect_text(img)?;
        if boxes.is_empty() {
            return Ok(vec![]);
        }

        // 2. Text recognition per box
        let mut lines: Vec<OcrLine> = Vec::new();
        for bbox in &boxes {
            if let Some(crop) = self.crop_box(img, bbox) {
                if let Some((text, conf)) = self.recognize_text(&crop)? {
                    lines.push(OcrLine {
                        box_points: *bbox,
                        text,
                        confidence: conf,
                    });
                }
            }
        }

        // Sort by reading order (top-to-bottom, left-to-right)
        lines.sort_by(|a, b| {
            let ay = a.box_points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            let by = b.box_points.iter().map(|p| p[1]).fold(f32::MAX, f32::min);
            ay.partial_cmp(&by)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ax = a.box_points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
                    let bx = b.box_points.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
                    ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        Ok(lines)
    }

    // ------------------------------------------------------------------
    // Text detection
    // ------------------------------------------------------------------

    fn detect_text(&mut self, img: &DynamicImage) -> Result<Vec<[[f32; 2]; 4]>> {
        let (h, w) = (img.height(), img.width());

        // Resize to multiple of 32, limit long side
        let ratio = DET_LIMIT_SIDE_LEN as f32 / w.max(h) as f32;
        let new_w = ((w as f32 * ratio) as u32 / 32 * 32).max(32);
        let new_h = ((h as f32 * ratio) as u32 / 32 * 32).max(32);

        let resized = image::imageops::resize(
            img, new_w, new_h, image::imageops::FilterType::Triangle,
        );

        // Normalize: (x/255 - mean) / std, CHW
        let mut arr = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let p = resized.get_pixel(x as u32, y as u32);
                arr[[0, 0, y, x]] = (p.0[0] as f32 / 255.0 - DET_MEAN[0]) / DET_STD[0];
                arr[[0, 1, y, x]] = (p.0[1] as f32 / 255.0 - DET_MEAN[1]) / DET_STD[1];
                arr[[0, 2, y, x]] = (p.0[2] as f32 / 255.0 - DET_MEAN[2]) / DET_STD[2];
            }
        }

        let input = ort::value::Tensor::from_array(arr)
            .map_err(|e| BobineError::Ort(format!("det input: {e}")))?;
        let outputs = self.det_session
            .run(inputs!["x" => input])
            .map_err(|e| BobineError::Ort(format!("det run: {e}")))?;

        // Output: [1, 1, H, W] probability map
        let prob = outputs["sigmoid_0.tmp_0"]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("det output: {e}")))?;

        let prob_h = prob.shape()[2];
        let prob_w = prob.shape()[3];

        // Build binary mask and find boxes
        let boxes = Self::extract_boxes_from_prob(
            &prob, prob_h, prob_w, new_w, new_h, w, h,
        );

        Ok(boxes)
    }

    /// Simple box extraction: threshold + contour finding via connected components
    fn extract_boxes_from_prob(
        prob: &ndarray::ArrayViewD<f32>,
        ph: usize, pw: usize, rw: u32, rh: u32, ow: u32, oh: u32,
    ) -> Vec<[[f32; 2]; 4]> {
        // Build binary image from probability map
        let mut mask = GrayImage::new(pw as u32, ph as u32);
        for y in 0..ph {
            for x in 0..pw {
                let v = prob[[0, 0, y, x]];
                mask.put_pixel(x as u32, y as u32, image::Luma([(v > DET_THRESH) as u8 * 255]));
            }
        }

        // Find connected components
        let scale_x = ow as f32 / rw as f32;
        let scale_y = oh as f32 / rh as f32;

        // Simple approach: scan for connected regions of white pixels
        let mut visited = vec![vec![false; pw]; ph];
        let mut boxes: Vec<[[f32; 2]; 4]> = vec![];

        for y in 0..ph {
            for x in 0..pw {
                if mask.get_pixel(x as u32, y as u32).0[0] < 128 || visited[y][x] {
                    continue;
                }

                // Flood fill to find component bounds
                let mut min_x = x as u32;
                let mut min_y = y as u32;
                let mut max_x = x as u32;
                let mut max_y = y as u32;
                let mut area = 0u32;
                let mut stack = vec![(x as u32, y as u32)];
                visited[y][x] = true;

                while let Some((cx, cy)) = stack.pop() {
                    area += 1;
                    min_x = min_x.min(cx);
                    min_y = min_y.min(cy);
                    max_x = max_x.max(cx);
                    max_y = max_y.max(cy);

                    for (nx, ny) in [(cx.wrapping_sub(1), cy), (cx+1, cy), (cx, cy.wrapping_sub(1)), (cx, cy+1)] {
                        if nx < pw as u32 && ny < ph as u32
                            && !visited[ny as usize][nx as usize]
                            && mask.get_pixel(nx, ny).0[0] >= 128
                        {
                            visited[ny as usize][nx as usize] = true;
                            stack.push((nx, ny));
                        }
                    }
                }

                // Filter tiny regions
                if area < 50 {
                    continue;
                }

                // Scale back to original image coordinates
                let bx0 = min_x as f32 * scale_x;
                let by0 = min_y as f32 * scale_y;
                let bx1 = max_x as f32 * scale_x;
                let by1 = max_y as f32 * scale_y;

                boxes.push([
                    [bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1],
                ]);
            }
        }

        boxes
    }

    // ------------------------------------------------------------------
    // Box cropping
    // ------------------------------------------------------------------

    fn crop_box(&self, img: &DynamicImage, bbox: &[[f32; 2]; 4]) -> Option<DynamicImage> {
        let x0 = bbox.iter().map(|p| p[0]).fold(f32::MAX, f32::min).max(0.0) as u32;
        let y0 = bbox.iter().map(|p| p[1]).fold(f32::MAX, f32::min).max(0.0) as u32;
        let x1 = bbox.iter().map(|p| p[0]).fold(0.0, f32::max).min(img.width() as f32) as u32;
        let y1 = bbox.iter().map(|p| p[1]).fold(0.0, f32::max).min(img.height() as f32) as u32;
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        Some(img.crop_imm(x0, y0, x1 - x0, y1 - y0))
    }

    // ------------------------------------------------------------------
    // Text recognition
    // ------------------------------------------------------------------

    fn recognize_text(&mut self, crop: &DynamicImage) -> Result<Option<(String, f32)>> {
        let (h, w) = (crop.height(), crop.width());
        if h < 4 || w < 4 {
            return Ok(None);
        }

        // Resize: height → 48, width preserves aspect ratio
        let new_h = REC_IMG_H;
        let ratio = new_h as f32 / h as f32;
        let new_w = (w as f32 * ratio).round() as u32;

        let resized = image::imageops::resize(
            crop, new_w, new_h, image::imageops::FilterType::Triangle,
        );
        let mut rgb = image::RgbImage::new(new_w, new_h);
        for y in 0..new_h {
            for x in 0..new_w {
                let p = resized.get_pixel(x, y);
                rgb.put_pixel(x, y, image::Rgb([p.0[0], p.0[1], p.0[2]]));
            }
        }

        // Normalize: (x/255 - 0.5) / 0.5, shape [1, 3, 48, W]
        let max_w = new_w.max(32) as usize;
        let mut arr = Array4::<f32>::zeros((1, 3, new_h as usize, max_w));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let p = rgb.get_pixel(x as u32, y as u32);
                arr[[0, 0, y, x]] = p.0[0] as f32 / 255.0 - 0.5;
                arr[[0, 1, y, x]] = p.0[1] as f32 / 255.0 - 0.5;
                arr[[0, 2, y, x]] = p.0[2] as f32 / 255.0 - 0.5;
            }
        }

        let input = ort::value::Tensor::from_array(arr)
            .map_err(|e| BobineError::Ort(format!("rec input: {e}")))?;
        let outputs = self.rec_session
            .run(inputs!["x" => input])
            .map_err(|e| BobineError::Ort(format!("rec run: {e}")))?;

        // Output: [1, T, num_classes] CTC log-probabilities
        let logits = outputs["softmax_0.tmp_0"]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("rec output: {e}")))?;

        // Greedy CTC decode
        let (_batch, timesteps, num_classes) = (
            logits.shape()[0], logits.shape()[1], logits.shape()[2]
        );

        let mut prev = num_classes; // blank
        let mut text = String::new();
        let mut total_conf = 0.0f32;
        let mut count = 0u32;

        for t in 0..timesteps {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let v = logits[[0, t, c]];
                if v > best_val {
                    best_val = v;
                    best_idx = c;
                }
            }
            if best_idx != prev && best_idx < self.characters.len() {
                let ch = &self.characters[best_idx];
                if ch != " " || !text.ends_with(' ') {
                    text.push_str(ch);
                }
                total_conf += best_val;
                count += 1;
            }
            prev = best_idx;
        }

        if text.trim().is_empty() {
            return Ok(None);
        }

        let avg_conf = if count > 0 { total_conf / count as f32 } else { 0.0 };
        Ok(Some((text.trim().to_string(), avg_conf)))
    }
}
