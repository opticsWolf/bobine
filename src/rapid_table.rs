// RapidTable — table structure recognition via SLANet-plus ONNX.
//
// Port of RapidAI/RapidTable's PP-Structure path: preprocess (488 canvas),
// structure-token decoding with per-cell quad boxes, then OCR-line → cell
// matching to emit a final HTML table (converted to GFM by `tables.rs`).
//
// Model: slanet-plus.onnx (~7.8 MB), auto-downloaded from HuggingFace
// (`opendatalab/PDF-Extract-Kit-1.0`, sha256 d57a942a…, identical to
// RapidAI's modelscope release). Structure charset is read from the ONNX
// metadata key `character`.

use std::path::Path;

use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::{inputs, session::Session};
use tracing::info;

use crate::error::{BobineError, Result};
use crate::rapid_ocr::OcrLine;

const TABLE_MAX_LEN: u32 = 488;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Tokens that carry a cell bbox in the bbox head output.
const TD_TOKENS: [&str; 3] = ["<td>", "<td", "<td></td>"];
/// Wrapper tags stripped from the final HTML (matching RapidTable).
const FILTERED_TAGS: [&str; 4] = ["<thead>", "</thead>", "<tbody>", "</tbody>"];

/// Prepare the structure charset exactly like RapidTable's TableLabelDecode:
/// drop bare `<td>` (no-span merge), add `<td></td>`, wrap with sos/eos.
pub fn prepare_charset(mut dict: Vec<String>) -> Vec<String> {
    if !dict.iter().any(|c| c == "<td></td>") {
        dict.push("<td></td>".to_string());
    }
    dict.retain(|c| c != "<td>");
    let mut out = vec!["sos".to_string()];
    out.extend(dict);
    out.push("eos".to_string());
    out
}

/// L1-style corner distance used by RapidTable's matcher.
fn distance(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dis = (b[0] - a[0]).abs() + (b[1] - a[1]).abs() + (b[2] - a[2]).abs() + (b[3] - a[3]).abs();
    let dis2 = (b[0] - a[0]).abs() + (b[1] - a[1]).abs();
    let dis3 = (b[2] - a[2]).abs() + (b[3] - a[3]).abs();
    dis + dis2.min(dis3)
}

/// Rectangle IoU on xyxy boxes.
fn rect_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let area_a = ((a[2] - a[0]) * (a[3] - a[1])).max(0.0);
    let area_b = ((b[2] - b[0]) * (b[3] - b[1])).max(0.0);
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = a[2].min(b[2]);
    let bottom = a[3].min(b[3]);
    if right <= left || bottom <= top {
        return 0.0;
    }
    let inter = (right - left) * (bottom - top);
    inter / (area_a + area_b - inter)
}

fn quad_to_xyxy(q: &[f32; 8]) -> [f32; 4] {
    [
        q[0].min(q[2]).min(q[4]).min(q[6]),
        q[1].min(q[3]).min(q[5]).min(q[7]),
        q[0].max(q[2]).max(q[4]).max(q[6]),
        q[1].max(q[3]).max(q[5]).max(q[7]),
    ]
}

/// Match OCR line indices to cell indices (RapidTable `match_result`).
pub fn match_cells(cell_xyxy: &[[f32; 4]], det_xyxy: &[[f32; 4]]) -> Vec<Vec<usize>> {
    const MIN_IOU: f32 = 1e-8;
    let mut matched: Vec<Vec<usize>> = vec![Vec::new(); cell_xyxy.len()];
    for (i, gt) in det_xyxy.iter().enumerate() {
        let mut best: Option<(usize, f32, f32)> = None;
        for (j, pred) in cell_xyxy.iter().enumerate() {
            let one_minus_iou = 1.0 - rect_iou(*gt, *pred);
            let dist = distance(*gt, *pred);
            // sort key: (1-iou, distance) ascending — keep strict best
            if one_minus_iou >= 1.0 - MIN_IOU {
                continue;
            }
            if best.map_or(true, |(_, bi, bd)| {
                one_minus_iou < bi || (one_minus_iou == bi && dist < bd)
            }) {
                best = Some((j, one_minus_iou, dist));
            }
        }
        if let Some((j, _, _)) = best {
            matched[j].push(i);
        }
    }
    matched
}

/// Build the final HTML from structure tokens + matched OCR contents
/// (port of RapidTable `get_pred_html`, `<b>` merging included).
pub fn build_html(structures: &[String], matched: &[Vec<usize>], ocr_texts: &[String]) -> String {
    let mut out = String::from("<html><body><table>");
    let mut td_index = 0usize;
    for tag in structures {
        if !tag.contains("</td>") {
            out.push_str(tag);
            continue;
        }

        let empty_cell = tag == "<td></td>";
        if empty_cell {
            out.push_str("<td>");
        }

        if let Some(idxs) = matched.get(td_index) {
            if !idxs.is_empty() {
                let b_with = idxs.len() > 1 && ocr_texts[idxs[0]].contains("<b>");
                if b_with {
                    out.push_str("<b>");
                }
                let last = idxs.len() - 1;
                for (i, di) in idxs.iter().enumerate() {
                    let mut content = ocr_texts[*di].clone();
                    if idxs.len() > 1 {
                        if content.is_empty() {
                            continue;
                        }
                        if content.starts_with(' ') {
                            content = content[1..].to_string();
                        }
                        if content.contains("<b>") {
                            content = content[3..].to_string();
                        }
                        if content.ends_with("</b>") {
                            content = content[..content.len() - 4].to_string();
                        }
                        if content.is_empty() {
                            continue;
                        }
                        if i != last && !content.ends_with(' ') {
                            content.push(' ');
                        }
                    }
                    out.push_str(&content);
                }
                if b_with {
                    out.push_str("</b>");
                }
            }
        }

        if empty_cell {
            out.push_str("</td>");
        } else {
            out.push_str(tag);
        }
        td_index += 1;
    }
    out.push_str("</table></body></html>");

    // Filter thead/tbody wrapper tags.
    let mut html = out;
    for t in FILTERED_TAGS {
        html = html.replace(t, "");
    }
    html
}

/// Download location of the SLANet-plus ONNX model.
pub const SLANET_PLUS_REPO: (&str, &str) = ("opendatalab", "PDF-Extract-Kit-1.0");
pub const SLANET_PLUS_FILENAME: &str = "models/TabRec/SlanetPlus/slanet-plus.onnx";

/// Destination of the SLANet-plus ONNX model:
/// `<cache_dir>/models/TabRec/SlanetPlus/slanet-plus.onnx`.
pub fn slanet_plus_dest(cache_dir: &Path) -> std::path::PathBuf {
    cache_dir.join(SLANET_PLUS_FILENAME)
}

/// Fetch slanet-plus.onnx (~7.8 MB) under `cache_dir`, returning its path.
/// Skips the network entirely when the file is already cached (hf-hub's
/// single-file + local_dir path does not do this check itself).
pub fn download_slanet_plus(cache_dir: &Path) -> Result<std::path::PathBuf> {
    let dest = slanet_plus_dest(cache_dir);
    if dest.exists() {
        return Ok(dest);
    }
    info!("Downloading slanet-plus.onnx from HuggingFace...");
    let client =
        hf_hub::HFClientSync::new().map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?;
    let repo_api = client.model(SLANET_PLUS_REPO.0, SLANET_PLUS_REPO.1);
    repo_api
        .download_file()
        .filename(SLANET_PLUS_FILENAME.to_string())
        .local_dir(cache_dir.to_path_buf())
        .send()
        .map_err(|e| BobineError::Ort(format!("download table model: {e}")))
}

/// SLANet-plus table structure recognizer.
pub struct RapidTable {
    session: Session,
    /// Decoded charset including sos/eos.
    characters: Vec<String>,
    beg_idx: usize,
    end_idx: usize,
}

impl RapidTable {
    pub fn load(model_path: &Path, providers: &[String]) -> Result<Self> {
        info!(
            "Loading RapidTable (SLANet-plus) from {}",
            model_path.display()
        );
        let session = crate::engine::apply_providers(
            Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?,
            providers,
        )?
        .commit_from_file(model_path)
        .map_err(|e| BobineError::Ort(e.to_string()))?;

        let dict: Vec<String> = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("character"))
            .map(|c| {
                c.lines()
                    .filter(|l| !l.is_empty())
                    .map(|s| s.to_string())
                    .collect()
            })
            .ok_or_else(|| {
                BobineError::ModelNotAvailable(
                    "table model has no `character` metadata; not a valid SLANet export".into(),
                )
            })?;
        let characters = prepare_charset(dict);

        let beg_idx = characters
            .iter()
            .position(|c| c == "sos")
            .ok_or_else(|| BobineError::ModelNotAvailable("charset missing sos".into()))?;
        let end_idx = characters
            .iter()
            .position(|c| c == "eos")
            .ok_or_else(|| BobineError::ModelNotAvailable("charset missing eos".into()))?;

        info!(num_tokens = characters.len(), "RapidTable ready");
        Ok(Self {
            session,
            characters,
            beg_idx,
            end_idx,
        })
    }

    /// Recognize a table crop given its OCR lines; returns full HTML or
    /// `None` when no cells were decoded.
    pub fn recognize(
        &mut self,
        img: &DynamicImage,
        ocr_lines: &[OcrLine],
    ) -> Result<Option<String>> {
        let input = preprocess(img)?;
        let tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| BobineError::Ort(format!("table input: {e}")))?;

        // Identify outputs by name (scale_0 = bboxes, scale_1 = structure probs).
        let (mut struct_name, mut bbox_name) = (String::new(), String::new());
        for o in self.session.outputs() {
            let name = o.name().to_string();
            if name.contains("scale_0") {
                bbox_name = name;
            } else if name.contains("scale_1") {
                struct_name = name;
            }
        }
        if struct_name.is_empty() || bbox_name.is_empty() {
            return Err(BobineError::ModelNotAvailable(
                "unexpected SLANet output layout".into(),
            ));
        }

        let outputs = self
            .session
            .run(inputs!["x" => tensor])
            .map_err(|e| BobineError::Ort(format!("table run: {e}")))?;

        let probs = outputs[struct_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("table probs: {e}")))?;
        let bboxes = outputs[bbox_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("table bboxes: {e}")))?;

        let ps = probs.shape();
        let vocab = *ps.last().expect("probs rank");
        let seq_len = ps[ps.len() - 2];
        let bs = probs.view();
        let bb = bboxes.view();

        let (orig_h, orig_w) = (img.height() as f32, img.width() as f32);
        let ratio = TABLE_MAX_LEN as f32 / orig_h.max(orig_w);

        let mut structures: Vec<String> = Vec::new();
        let mut cell_quads: Vec<[f32; 8]> = Vec::new();

        let batched_probs = ps.len() == 3;
        let bbox_steps = if batched_probs {
            bb.shape()[1]
        } else {
            bb.shape()[0]
        };

        for t in 0..seq_len {
            let ti = t.min(bbox_steps - 1); // bbox head may emit fewer steps
            let row_probs: ndarray::ArrayView1<f32> = if batched_probs {
                bs.slice(ndarray::s![0, t, ..])
            } else {
                bs.slice(ndarray::s![t, ..])
            };
            let row_bbox: ndarray::ArrayView1<f32> = if batched_probs {
                bb.slice(ndarray::s![0, ti, ..])
            } else {
                bb.slice(ndarray::s![ti, ..])
            };
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (c, v) in row_probs.iter().enumerate() {
                if *v > best_val {
                    best_val = *v;
                    best_idx = c;
                }
            }
            if t > 0 && best_idx == self.end_idx {
                break;
            }
            if best_idx == self.beg_idx || best_idx == self.end_idx {
                continue;
            }
            let token = self.characters[best_idx].clone();
            if TD_TOKENS.contains(&token.as_str()) {
                let mut quad = [0.0f32; 8];
                for k in 0..8 {
                    quad[k] = row_bbox[k];
                }
                // _bbox_decode: x *= w, y *= h (original dims)
                for k in (0..8).step_by(2) {
                    quad[k] *= orig_w;
                }
                for k in (1..8).step_by(2) {
                    quad[k] *= orig_h;
                }
                // rescale_cell_bboxes (slanet_plus): map padded canvas → orig px
                let w_ratio = TABLE_MAX_LEN as f32 / (orig_w * ratio);
                let h_ratio = TABLE_MAX_LEN as f32 / (orig_h * ratio);
                for k in (0..8).step_by(2) {
                    quad[k] *= w_ratio;
                }
                for k in (1..8).step_by(2) {
                    quad[k] *= h_ratio;
                }
                if quad.iter().all(|v| v.abs() < 1e-6) {
                    continue; // placeholder cell
                }
                cell_quads.push(quad);
            }
            structures.push(token);
        }

        if cell_quads.is_empty() {
            return Ok(None);
        }

        // build_html() adds the html/body/table wrapper itself.

        // OCR filter: lines fully above the topmost cell are page furniture.
        let cell_xyxy: Vec<[f32; 4]> = cell_quads.iter().map(quad_to_xyxy).collect();
        let min_cell_y = cell_xyxy.iter().map(|b| b[1]).fold(f32::MAX, f32::min);
        let kept: Vec<&OcrLine> = ocr_lines
            .iter()
            .filter(|l| {
                l.box_points
                    .iter()
                    .map(|p| p[1])
                    .fold(f32::NEG_INFINITY, f32::max)
                    >= min_cell_y
            })
            .collect();

        let det_xyxy: Vec<[f32; 4]> = kept
            .iter()
            .map(|l| {
                let xs: Vec<f32> = l.box_points.iter().map(|p| p[0]).collect();
                let ys: Vec<f32> = l.box_points.iter().map(|p| p[1]).collect();
                [
                    xs.iter().cloned().fold(f32::MAX, f32::min),
                    ys.iter().cloned().fold(f32::MAX, f32::min),
                    xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                    ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                ]
            })
            .collect();
        let texts: Vec<String> = kept.iter().map(|l| l.text.clone()).collect();

        let matched = match_cells(&cell_xyxy, &det_xyxy);
        Ok(Some(build_html(&structures, &matched, &texts)))
    }
}

/// Resize longest side to 488 (aspect-preserving), ImageNet-normalize,
/// pad bottom-right to 488×488, CHW output.
fn preprocess(img: &DynamicImage) -> Result<Array4<f32>> {
    let (w, h) = (img.width(), img.height());
    let ratio = TABLE_MAX_LEN as f32 / w.max(h) as f32;
    let new_w = ((w as f32 * ratio).round() as u32).clamp(1, TABLE_MAX_LEN);
    let new_h = ((h as f32 * ratio).round() as u32).clamp(1, TABLE_MAX_LEN);

    let resized = image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::Triangle);

    let mut arr = Array4::<f32>::zeros((1, 3, TABLE_MAX_LEN as usize, TABLE_MAX_LEN as usize));
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let p = resized.get_pixel(x as u32, y as u32).0;
            arr[[0, 0, y, x]] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            arr[[0, 1, y, x]] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            arr[[0, 2, y, x]] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }
    }
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_charset_merges_no_span() {
        let dict = vec![
            "<thead>".to_string(),
            "<tr>".to_string(),
            "<td>".to_string(),
            "</td>".to_string(),
        ];
        let cs = prepare_charset(dict);
        assert_eq!(cs.first().unwrap(), "sos");
        assert_eq!(cs.last().unwrap(), "eos");
        assert!(
            cs.contains(&"<td></td>".to_string()),
            "empty-cell token added"
        );
        assert!(!cs.contains(&"<td>".to_string()), "bare <td> removed");
        assert_eq!(cs.len(), 1 + 4 + 1); // -1 td +1 empty +2 specials
    }

    #[test]
    fn test_rect_iou_and_distance() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let iou = rect_iou(a, b);
        assert!((iou - 25.0 / 175.0).abs() < 1e-5);
        assert_eq!(rect_iou(a, a), 1.0);
        assert_eq!(rect_iou(a, [100.0, 100.0, 110.0, 110.0]), 0.0);

        let d = distance([0.0, 0.0, 4.0, 0.0], [1.0, 0.0, 5.0, 0.0]);
        assert!((d - (1.0 + 0.0 + 1.0 + 0.0 + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_match_cells_prefers_iou_then_distance() {
        // two cells side by side; one det box overlapping each
        let cells = [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]];
        let dets = [[1.0, 1.0, 9.0, 9.0], [21.0, 1.0, 29.0, 9.0]];
        let m = match_cells(&cells, &dets);
        assert_eq!(m[0], vec![0]);
        assert_eq!(m[1], vec![1]);

        // empty case: det nowhere near any cell → unmatched everywhere
        let far = [[100.0, 100.0, 110.0, 110.0]];
        let m2 = match_cells(&cells, &far);
        assert!(m2.iter().all(Vec::is_empty));

        // multiple dets into one cell keep order
        let stacked = [[1.0, 1.0, 9.0, 5.0], [1.0, 5.0, 9.0, 9.0]];
        let m3 = match_cells(&cells[..1], &stacked);
        assert_eq!(m3[0], vec![0, 1]);
    }

    #[test]
    fn test_build_html_basic_and_merge() {
        let structures = vec![
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
        ];
        // single content per cell
        let texts = vec!["hello".to_string(), "world".to_string()];
        let matched = vec![vec![0], vec![1]];
        let html = build_html(&structures, &matched, &texts);
        assert!(
            html.contains("<table><tr><td>hello</td><td>world</td></tr>"),
            "{html}"
        );
        assert!(!html.contains("<thead>"));

        // multi-line merge joins with spaces
        let texts2 = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let matched2 = vec![vec![0, 1], vec![2, 3]];
        let html2 = build_html(&structures, &matched2, &texts2);
        assert!(html2.contains("<td>a b</td>"), "{html2}");

        // unmatched cell stays empty
        let matched3 = vec![vec![], vec![0]];
        let html3 = build_html(&structures, &matched3, &["x".to_string()]);
        assert!(html3.contains("<td></td>"), "{html3}");
    }

    #[test]
    fn test_preprocess_pads_to_canvas() {
        let img = DynamicImage::new_rgb8(200, 100);
        let arr = preprocess(&img).unwrap();
        assert_eq!(arr.shape(), &[1, 3, 488, 488]);
        // resized area is 488x244 at top-left; bottom-right padding is zeros
        assert_eq!(arr[[0, 0, 400, 400]], 0.0);
        // inside the resized area: black px normalizes to (0/255-MEAN)/STD
        let expect = (0.0 / 255.0 - MEAN[0]) / STD[0];
        assert!((arr[[0, 0, 10, 10]] - expect).abs() < 1e-6);
    }
}
