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
// DB postprocess geometry helpers
// ---------------------------------------------------------------------------

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Andrew's monotone chain convex hull.
fn convex_hull(mut pts: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f32, f32)> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<(f32, f32)> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area enclosing rectangle of the hull via rotating calipers.
/// Returns corners ordered clockwise from top-left plus (width, height).
fn min_area_rect(hull: &[(f32, f32)]) -> Option<([[f32; 2]; 4], f32, f32)> {
    if hull.len() < 3 {
        return None;
    }
    let mut best_area = f32::INFINITY;
    let mut best: Option<([[f32; 2]; 4], f32, f32)> = None;
    for i in 0..hull.len() {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % hull.len()];
        let len = dist([p0.0, p0.1], [p1.0, p1.1]);
        if len < 1e-6 {
            continue;
        }
        let dx = (p1.0 - p0.0) / len;
        let dy = (p1.1 - p0.1) / len;
        let (mut min_u, mut max_u, mut min_v, mut max_v) =
            (f32::MAX, f32::NEG_INFINITY, f32::MAX, f32::NEG_INFINITY);
        for p in hull {
            let u = p.0 * dx + p.1 * dy;
            let v = -p.0 * dy + p.1 * dx;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if area < best_area {
            best_area = area;
            // Corner positions in rotated frame.
            let cs = [
                [min_u, min_v],
                [max_u, min_v],
                [max_u, max_v],
                [min_u, max_v],
            ];
            let mut corners = [[0.0f32; 2]; 4];
            for (c, s) in corners.iter_mut().zip(cs.iter()) {
                c[0] = s[0] * dx - s[1] * dy;
                c[1] = s[0] * dy + s[1] * dx;
            }
            best = Some((order_clockwise(corners), max_u - min_u, max_v - min_v));
        }
    }
    best
}

/// Order 4 corners as [TL, TR, BR, BL] (clockwise from top-left),
/// mirroring cv2 minAreaRect's boxPoints convention used by RapidOCR.
fn order_clockwise(mut pts: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    let (l0, l1) = (pts[0], pts[1]); // two leftmost
    let (r0, r1) = (pts[2], pts[3]); // two rightmost
    let tl_bl = if l0[1] <= l1[1] { (l0, l1) } else { (l1, l0) };
    let tr_br = if r0[1] <= r1[1] { (r0, r1) } else { (r1, r0) };
    [tl_bl.0, tr_br.0, tr_br.1, tl_bl.1]
}

/// Offset each edge of an axis-aligned-in-own-frame rectangle outward by `delta`.
fn expand_rect(corners: [[f32; 2]; 4], delta: f32) -> [[f32; 2]; 4] {
    // Unit vectors along top and left edges from TL (=corners[0]).
    let w = dist(corners[0], corners[1]);
    let h = dist(corners[3], corners[0]);
    if w < 1e-6 || h < 1e-6 {
        return corners;
    }
    let ux = [
        (corners[1][0] - corners[0][0]) / w,
        (corners[1][1] - corners[0][1]) / w,
    ];
    let uy = [
        (corners[3][0] - corners[0][0]) / h,
        (corners[3][1] - corners[0][1]) / h,
    ];
    [
        [
            corners[0][0] + (-ux[0] - uy[0]) * delta,
            corners[0][1] + (-ux[1] - uy[1]) * delta,
        ], // TL
        [
            corners[1][0] + (ux[0] - uy[0]) * delta,
            corners[1][1] + (ux[1] - uy[1]) * delta,
        ], // TR
        [
            corners[2][0] + (ux[0] + uy[0]) * delta,
            corners[2][1] + (ux[1] + uy[1]) * delta,
        ], // BR
        [
            corners[3][0] + (-ux[0] + uy[0]) * delta,
            corners[3][1] + (-ux[1] + uy[1]) * delta,
        ], // BL
    ]
}

/// Bilinear sample of an Rgb8 buffer at float coordinates.
fn bilinear_sample(img: &image::RgbImage, x: f32, y: f32) -> image::Rgb<u8> {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let px = |xx: u32, yy: u32| {
        img.get_pixel(xx.min(img.width() - 1), yy.min(img.height() - 1))
            .0
    };
    let p00 = px(x0, y0);
    let p10 = px(x0 + 1, y0);
    let p01 = px(x0, y0 + 1);
    let p11 = px(x0 + 1, y0 + 1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let bot = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    image::Rgb(out)
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DET_LIMIT_SIDE_LEN: u32 = 960;
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406]; // ImageNet-style
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const DET_THRESH: f32 = 0.3;
const DET_BOX_THRESH: f32 = 0.5;
/// DB postprocess: min side of the mini-box before unclip.
const DET_MIN_SIZE: f32 = 3.0;
/// DB polygon offset ratio (RapidOCR default).
const DET_UNCLIP_RATIO: f32 = 1.6;
/// Rotations below this angle are treated as axis-aligned when cropping.
const CROP_ROTATION_TOLERANCE: f32 = 0.05; // ≈ 2.9°
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
    /// Whether `characters` has been aligned to the model's class count.
    charset_aligned: bool,
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

/// Crop a (possibly rotated) quad from the image; near-axis boxes take
/// the cheap axis-aligned path.
fn crop_quad(img: &DynamicImage, bbox: &[[f32; 2]; 4]) -> Option<DynamicImage> {
    let w = dist(bbox[0], bbox[1]);
    let h = dist(bbox[1], bbox[2]);
    if w < 4.0 || h < 4.0 {
        return None;
    }

    // Top-edge angle; near-axis boxes take the cheap path.
    let angle = (bbox[1][1] - bbox[0][1]).atan2(bbox[1][0] - bbox[0][0]);
    if angle.abs() < CROP_ROTATION_TOLERANCE {
        let x0 = bbox.iter().map(|p| p[0]).fold(f32::MAX, f32::min).max(0.0) as u32;
        let y0 = bbox.iter().map(|p| p[1]).fold(f32::MAX, f32::min).max(0.0) as u32;
        let x1 = bbox
            .iter()
            .map(|p| p[0])
            .fold(0.0, f32::max)
            .min(img.width() as f32) as u32;
        let y1 = bbox
            .iter()
            .map(|p| p[1])
            .fold(0.0, f32::max)
            .min(img.height() as f32) as u32;
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        return Some(img.crop_imm(x0, y0, x1 - x0, y1 - y0));
    }

    // Rotated box: inverse-map each upright pixel into the source quad.
    let rgb = img.to_rgb8();
    let (iw, ih) = (rgb.width() as i32, rgb.height() as i32);
    let (w, h) = (w.round().max(4.0) as u32, h.round().max(4.0) as u32);
    let ex = ((angle.cos()), (angle.sin())); // unit vector TL→TR
    let ey = (-angle.sin(), angle.cos()); // unit vector TL→BL
    let mut out = image::RgbImage::new(w, h);
    for v in 0..h {
        for u in 0..w {
            let sx = bbox[0][0] + ex.0 * u as f32 + ey.0 * v as f32;
            let sy = bbox[0][1] + ex.1 * u as f32 + ey.1 * v as f32;
            if sx < 0.0 || sy < 0.0 || sx >= (iw - 1) as f32 || sy >= (ih - 1) as f32 {
                out.put_pixel(u, v, image::Rgb([255, 255, 255]));
                continue;
            }
            out.put_pixel(u, v, bilinear_sample(&rgb, sx, sy));
        }
    }
    Some(DynamicImage::ImageRgb8(out))
}

impl RapidOcr {
    pub fn load(
        det_model: &Path,
        rec_model: &Path,
        ocr_lang: &str,
        providers: &[String],
    ) -> Result<Self> {
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

        info!(num_chars = characters.len(), "RapidOCR ready");

        Ok(Self {
            det_session,
            rec_session,
            characters,
            charset_aligned: false,
        })
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

        let resized =
            image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);

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
        let outputs = self
            .det_session
            .run(inputs!["x" => input])
            .map_err(|e| BobineError::Ort(format!("det run: {e}")))?;

        // Output: [1, 1, H, W] probability map
        let prob = outputs["sigmoid_0.tmp_0"]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("det output: {e}")))?;

        let prob_h = prob.shape()[2];
        let prob_w = prob.shape()[3];

        // Build binary mask and find boxes
        let boxes = Self::extract_boxes_from_prob(&prob, prob_h, prob_w, new_w, new_h, w, h);

        Ok(boxes)
    }

    /// DB-style box extraction: threshold → connected components →
    /// convex hull → min-area rect (rotating calipers) → probability score →
    /// polygon unclip. Handles rotated text, unlike plain bounding boxes.
    fn extract_boxes_from_prob(
        prob: &ndarray::ArrayViewD<f32>,
        ph: usize,
        pw: usize,
        rw: u32,
        rh: u32,
        ow: u32,
        oh: u32,
    ) -> Vec<[[f32; 2]; 4]> {
        let scale_x = ow as f32 / rw as f32;
        let scale_y = oh as f32 / rh as f32;

        let mut visited = vec![vec![false; pw]; ph];
        let mut boxes: Vec<[[f32; 2]; 4]> = vec![];

        let at = |x: usize, y: usize| prob[[0, 0, y, x]];

        for cy in 0..ph {
            for cx in 0..pw {
                if visited[cy][cx] || at(cx, cy) <= DET_THRESH {
                    continue;
                }

                // Flood fill: collect component pixels + probability mass.
                let mut pixels: Vec<(u32, u32)> = Vec::new();
                let mut prob_sum = 0.0f32;
                let mut stack = vec![(cx as u32, cy as u32)];
                visited[cy][cx] = true;
                while let Some((x, y)) = stack.pop() {
                    pixels.push((x, y));
                    prob_sum += at(x as usize, y as usize);
                    for (nx, ny) in [
                        (x.wrapping_sub(1), y),
                        (x + 1, y),
                        (x, y.wrapping_sub(1)),
                        (x, y + 1),
                    ] {
                        if nx < pw as u32
                            && ny < ph as u32
                            && !visited[ny as usize][nx as usize]
                            && at(nx as usize, ny as usize) > DET_THRESH
                        {
                            visited[ny as usize][nx as usize] = true;
                            stack.push((nx, ny));
                        }
                    }
                }

                if pixels.len() < 50 {
                    continue; // noise
                }

                // Mean probability inside the region (box_score_fast analog).
                let score = prob_sum / pixels.len() as f32;
                if score < DET_BOX_THRESH {
                    continue;
                }

                let hull =
                    convex_hull(pixels.iter().map(|(x, y)| (*x as f32, *y as f32)).collect());
                let Some((corners, bw, bh)) = min_area_rect(&hull) else {
                    continue;
                };
                if bw.min(bh) < DET_MIN_SIZE {
                    continue;
                }

                // DB unclip: offset each edge outward by δ = area·r/perimeter.
                let delta = (bw * bh) * DET_UNCLIP_RATIO / (2.0 * (bw + bh));
                if (bw + 2.0 * delta).min(bh + 2.0 * delta) < DET_MIN_SIZE + 2.0 {
                    continue;
                }
                let expanded = expand_rect(corners, delta);

                // Scale back to original image coordinates (round + clamp).
                let mut out = [[0.0f32; 2]; 4];
                for (o, p) in out.iter_mut().zip(expanded.iter()) {
                    o[0] = ((p[0] * scale_x).round()).clamp(0.0, ow as f32);
                    o[1] = ((p[1] * scale_y).round()).clamp(0.0, oh as f32);
                }
                if boxes.len() < 1000 {
                    boxes.push(out);
                }
            }
        }

        boxes
    }

    // ------------------------------------------------------------------
    // Box cropping (rotation-aware)
    // ------------------------------------------------------------------

    fn crop_box(&self, img: &DynamicImage, bbox: &[[f32; 2]; 4]) -> Option<DynamicImage> {
        crop_quad(img, bbox)
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

        let resized =
            image::imageops::resize(crop, new_w, new_h, image::imageops::FilterType::Triangle);
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
        let outputs = self
            .rec_session
            .run(inputs!["x" => input])
            .map_err(|e| BobineError::Ort(format!("rec run: {e}")))?;

        // Output: [1, T, num_classes] CTC log-probabilities. Paddle2ONNX
        // names the softmax node differently across export versions
        // (softmax_0.tmp_0, softmax_11.tmp_0, ...) - pick the first output
        // whose name starts with "softmax", else the first output.
        let logits_view = outputs
            .iter()
            .find(|(k, _)| k.starts_with("softmax"))
            .or_else(|| outputs.iter().next())
            .map(|(_, v)| v)
            .ok_or_else(|| BobineError::Ort("rec model has no outputs".into()))?;
        let logits = logits_view
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("rec output: {e}")))?;

        // Greedy CTC decode. PaddleOCR exports vary in whether the metadata
        // charset includes the leading blank class and/or trailing space
        // class - align once using the actual output class count.
        let num_classes = logits.shape()[2];
        if !self.charset_aligned {
            let l = self.characters.len();
            if num_classes == l + 2 {
                self.characters.insert(0, String::new()); // blank
                self.characters.push(" ".to_string()); // space
            } else if num_classes == l + 1 {
                self.characters.insert(0, String::new()); // blank
            }
            self.charset_aligned = true;
        }
        let (_batch, timesteps, num_classes) =
            (logits.shape()[0], logits.shape()[1], logits.shape()[2]);

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

        let avg_conf = if count > 0 {
            total_conf / count as f32
        } else {
            0.0
        };
        Ok(Some((text.trim().to_string(), avg_conf)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn run_extract(prob: Array4<f32>, ph: usize, pw: usize) -> Vec<[[f32; 2]; 4]> {
        let view = prob.view().into_dyn();
        extract_boxes_from_prob_for_test(&view, ph, pw, pw as u32, ph as u32, pw as u32, ph as u32)
    }

    // Thin wrapper so tests can call the associated fn without a session.
    fn extract_boxes_from_prob_for_test(
        prob: &ndarray::ArrayViewD<f32>,
        ph: usize,
        pw: usize,
        rw: u32,
        rh: u32,
        ow: u32,
        oh: u32,
    ) -> Vec<[[f32; 2]; 4]> {
        RapidOcr::extract_boxes_from_prob(prob, ph, pw, rw, rh, ow, oh)
    }

    #[test]
    fn test_axis_aligned_blob() {
        let (ph, pw) = (64usize, 64usize);
        let mut prob = Array4::<f32>::zeros((1, 1, ph, pw));
        for y in 10..30 {
            for x in 8..40 {
                prob[[0, 0, y, x]] = 0.9;
            }
        }
        let boxes = run_extract(prob, ph, pw);
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        // Unclip legitimately expands beyond the blob; the box must *contain*
        // it (with a small tolerance) rather than hug its bounds.
        assert!(b[0][0] <= 9.0, "x0={}", b[0][0]);
        assert!(b[0][1] <= 11.0, "y0={}", b[0][1]);
        assert!(b[2][0] >= 38.5, "x2={} (unclip expands)", b[2][0]);
        assert!(b[2][1] >= 29.5, "y2={}", b[2][1]);
    }

    #[test]
    fn test_rotated_blob_detected_with_angle() {
        let (ph, pw) = (80usize, 80usize);
        let mut prob = Array4::<f32>::zeros((1, 1, ph, pw));
        let theta = 15.0f32.to_radians();
        let (cx, cy) = (40.0f32, 40.0f32);
        for y in 0..ph {
            for x in 0..pw {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                // rotate point into rect frame
                let ru = dx * theta.cos() + dy * theta.sin();
                let rv = -dx * theta.sin() + dy * theta.cos();
                if ru.abs() < 20.0 && rv.abs() < 4.0 {
                    prob[[0, 0, y, x]] = 0.95;
                }
            }
        }
        let boxes = run_extract(prob, ph, pw);
        assert_eq!(boxes.len(), 1);
        let b = boxes[0];
        // top edge angle should be ≈ 15°
        let angle = ((b[1][1] - b[0][1]).atan2(b[1][0] - b[0][0])).to_degrees();
        assert!(angle.abs() > 10.0 && angle.abs() < 20.0, "angle={}", angle);
        // width along rotated axis ≈ 40+ before unclip expansion
        let w = dist(b[0], b[1]);
        assert!(w > 38.0, "w={}", w);
    }

    #[test]
    fn test_low_probability_filtered() {
        let (ph, pw) = (64usize, 64usize);
        let mut prob = Array4::<f32>::zeros((1, 1, ph, pw));
        for y in 10..30 {
            for x in 8..40 {
                prob[[0, 0, y, x]] = 0.35; // above thresh, below box_thresh
            }
        }
        assert!(run_extract(prob, ph, pw).is_empty());
    }

    #[test]
    fn test_tiny_component_filtered() {
        let (ph, pw) = (64usize, 64usize);
        let mut prob = Array4::<f32>::zeros((1, 1, ph, pw));
        for y in 30..34 {
            for x in 30..34 {
                prob[[0, 0, y, x]] = 0.9; // 16 px < 50 px noise floor
            }
        }
        assert!(run_extract(prob, ph, pw).is_empty());
    }

    #[test]
    fn test_order_clockwise() {
        let pts = [
            [10.0, 12.0],
            [30.0, 10.0],
            [32.0, 30.0],
            [12.0, 32.0], // jumbled quad
        ];
        let ordered = order_clockwise(pts);
        assert_eq!(ordered[0], [10.0, 12.0]); // TL
        assert_eq!(ordered[1], [30.0, 10.0]); // TR
        assert_eq!(ordered[2], [32.0, 30.0]); // BR
        assert_eq!(ordered[3], [12.0, 32.0]); // BL
    }

    #[test]
    fn test_expand_rect_grows_uniformly() {
        let rect = [[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]];
        let e = expand_rect(rect, 3.0);
        assert!((e[0][0] - 7.0).abs() < 1e-4 && (e[0][1] - 7.0).abs() < 1e-4);
        assert!((e[2][0] - 33.0).abs() < 1e-4 && (e[2][1] - 23.0).abs() < 1e-4);
    }

    #[test]
    fn test_convex_hull_square() {
        let mut pts: Vec<(f32, f32)> = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                pts.push((x as f32, y as f32));
            }
        }
        let hull = convex_hull(pts);
        assert_eq!(hull.len(), 4);
        // all four square corners present
        for c in [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)] {
            assert!(hull.contains(&c), "{:?} not in {:?}", c, hull);
        }
    }

    #[test]
    fn test_rotation_aware_crop() {
        // 100x100 image with a white diagonal stripe on black; rotate-crop an
        // angled box aligned to the stripe and expect mostly-white output.
        let mut img = image::RgbImage::from_pixel(100, 100, image::Rgb([0, 0, 0]));
        let theta = 20.0f32.to_radians();
        let cx = 50.0f32;
        for y in 0..100u32 {
            for x in 0..100u32 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cx;
                let rv = -dx * theta.sin() + dy * theta.cos();
                if rv.abs() < 5.0 {
                    img.put_pixel(x, y, image::Rgb([255, 255, 255]));
                }
            }
        }
        // Center the 8px-tall quad on the stripe (shift up by half height).
        let tl = [
            cx - 30.0 * theta.cos() + 4.0 * theta.sin(),
            cx - 30.0 * theta.sin() - 4.0 * theta.cos(),
        ];
        let bbox = [
            tl,
            [tl[0] + 60.0 * theta.cos(), tl[1] + 60.0 * theta.sin()],
            [
                tl[0] + 60.0 * theta.cos() + 8.0 * (-theta.sin()),
                tl[1] + 60.0 * theta.sin() + 8.0 * theta.cos(),
            ],
            [tl[0] + 8.0 * (-theta.sin()), tl[1] + 8.0 * theta.cos()],
        ];
        let crop = crop_quad(&DynamicImage::ImageRgb8(img), &bbox).unwrap();
        assert_eq!(crop.width(), 60);
        let rgb = crop.to_rgb8();
        let white = rgb.pixels().filter(|p| p.0[0] > 200).count();
        let total = (crop.width() * crop.height()) as usize;
        assert!(
            white > total * 70 / 100,
            "white fraction {}/{}",
            white,
            total
        );
    }
}
