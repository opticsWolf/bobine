// POTATR — page-level table extraction via ONNX (P0: model slot + preprocess).
//
// Reference: kensho-technologies/POTATR (`potatr.inference`, `potatr.datasets`).
// Unlike SLANet-plus (`rapid_table.rs`, crop-level), POTATR takes the FULL
// PAGE as input and produces detections + structure + text-bearing cells in
// one forward pass. Page words are consumed downstream by the post-processor
// (P1) — they never enter the network (`POTATR.forward` takes images only).
//
// Model: potatr.onnx (116 MB) from the first-party serving repo
// `opticsWolf/potatr-onnx` (MIT end to end; exported from
// `kensho/POTATR-v1.1-Pub` safetensors at 4f853618…, opset 18, torch 2.15,
// dynamic H/W axes; torch-vs-ORT max|d| ≤ 6e-5; see manifest.json upstream).
// Gate: ikat micro GriTS-Con F1 0.900, TEDS 0.873 (torch side, 39 GT tables).

use std::path::Path;

use image::DynamicImage;
use ndarray::Array4;
use ort::{inputs, session::Session};
use tracing::info;

use crate::error::{BobineError, Result};

/// Longest side of the preprocess canvas (eval transform: `RandomMaxResize(1000, 1000)`).
pub const POTATR_LONGEST_SIDE: u32 = 1000;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Download location of the POTATR ONNX model.
pub const POTATR_REPO: (&str, &str) = ("opticsWolf", "potatr-onnx");
/// First-party serving repo: exported from `kensho/POTATR-v1.1-Pub`
/// safetensors at upstream `4f853618…` (MIT); torch-vs-ORT parity ≤ 6e-5.
pub const POTATR_FILENAME: &str = "potatr.onnx";
/// Pinned hub revision of the serving repo.
pub const POTATR_REV: &str = "e7f1ce67372e5f3972d7c93c8dc639258c6f1944";

/// Fetch potatr.onnx (~116 MB) through the shared hub cache (see
/// [`crate::hub_cache`]), returning its snapshot path to load.
/// `cache_dir` is the hub-cache root. Skips the network entirely when
/// the file is already cached ("using cached").
pub fn download_potatr(cache_dir: &Path) -> Result<std::path::PathBuf> {
    crate::hub_cache::hub_fetch(
        cache_dir,
        POTATR_REPO.0,
        POTATR_REPO.1,
        POTATR_FILENAME,
        "potatr",
        Some(POTATR_REV),
    )
}

/// Geometry of one preprocessed page: P1 needs the exact scale to map the
/// normalized `(cx,cy,h,w)` outputs back onto the original canvas
/// (same convention as `rapid_table.rs::recognize`).
#[derive(Debug, Clone, Copy)]
pub struct PotatrGeometry {
    /// `POTATR_LONGEST_SIDE / max(orig_w, orig_h)`.
    pub scale: f32,
    /// Resized canvas dims actually fed to the network.
    pub resized_w: u32,
    /// Resized canvas dims actually fed to the network.
    pub resized_h: u32,
    /// Original page-render dims.
    pub orig_w: u32,
    /// Original page-render dims.
    pub orig_h: u32,
}

/// Resize longest side to 1000 (aspect-preserving, PIL-rounding), RGB/255,
/// ImageNet-normalize, CHW output — mirroring the reference `eval` transform
/// (`RandomMaxResize(1000,1000)` + `ToTensor` + `Normalize`, `datasets.py:226,244`).
///
/// Resample: PIL `Image.resize` with default resample is BICUBIC on Pillow
/// ≥ 9.1 (verified empirically: default == BICUBIC, != NEAREST); the closest
/// `image`-crate filter is `CatmullRom`. The two kernels differ in the
/// far-lobe weights, so bit parity is NOT expected — the parity test allows
/// max|d| ≤ 3/255 pre-normalize. Sub-pixel shifts are structure-irrelevant
/// (GriTS match radii are far coarser); end-to-end parity is P1's gate.
pub fn preprocess(img: &DynamicImage) -> Result<(Array4<f32>, PotatrGeometry)> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(BobineError::Other("empty page render".into()));
    }
    let scale = POTATR_LONGEST_SIDE as f32 / w.max(h) as f32;
    // PIL: `int(round(scale * dim))` per axis (`datasets.py:134`).
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);

    let rgb = img.to_rgb8();
    let resized =
        image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::CatmullRom);

    let mut arr = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let p = resized.get_pixel(x as u32, y as u32).0;
            arr[[0, 0, y, x]] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            arr[[0, 1, y, x]] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            arr[[0, 2, y, x]] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }
    }
    let geo = PotatrGeometry {
        scale,
        resized_w: new_w,
        resized_h: new_h,
        orig_w: w,
        orig_h: h,
    };
    Ok((arr, geo))
}

/// Raw forward outputs: node logits `[1,250,17]`, normalized boxes
/// `[1,250,4]` (cx,cy,h,w over the *resized* canvas), edge logits
/// `[1,250,250,2]`. Shapes are dynamic (`ArrayD`) — P1 indexes them by
/// rank. Post-processing (P1) turns these into tables.
pub struct PotatrOutputs {
    /// Class logits per query (16 classes + no-object).
    pub node_logits: ndarray::ArrayD<f32>,
    /// Normalized boxes per query.
    pub boxes: ndarray::ArrayD<f32>,
    /// Edge logits per query pair.
    pub edge_logits: ndarray::ArrayD<f32>,
    /// Geometry to map boxes back to the original canvas.
    pub geometry: PotatrGeometry,
}

/// POTATR page-level table extractor.
pub struct Potatr {
    session: Session,
}

impl Potatr {
    pub fn load(model_path: &Path, providers: &[String]) -> Result<Self> {
        info!("Loading POTATR from {}", model_path.display());
        let session = crate::engine::session_builder(providers)?
            .commit_from_file(model_path)
            .map_err(|e| BobineError::Ort(e.to_string()))?;
        info!("POTATR ready");
        Ok(Self { session })
    }

    /// Run one page: preprocess → forward → raw outputs (P1 post-processes).
    pub fn forward(&mut self, img: &DynamicImage) -> Result<PotatrOutputs> {
        let (input, geometry) = preprocess(img)?;
        let tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| BobineError::Ort(format!("potatr input: {e}")))?;

        // Identify outputs by name (robust to session output ordering),
        // same pattern as `rapid_table.rs::recognize`.
        let (mut node_name, mut box_name, mut edge_name) =
            (String::new(), String::new(), String::new());
        for o in self.session.outputs() {
            let name = o.name().to_string();
            if name.contains("pred_node_logits") {
                node_name = name;
            } else if name.contains("pred_boxes") {
                box_name = name;
            } else if name.contains("pred_edge_logits") {
                edge_name = name;
            }
        }
        if node_name.is_empty() || box_name.is_empty() || edge_name.is_empty() {
            return Err(BobineError::ModelNotAvailable(
                "unexpected POTATR output layout".into(),
            ));
        }

        let outputs = self
            .session
            .run(inputs!["image" => tensor])
            .map_err(|e| BobineError::Ort(format!("potatr run: {e}")))?;

        let node_logits = outputs[node_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr nodes: {e}")))?
            .into_owned();
        let boxes = outputs[box_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr boxes: {e}")))?
            .into_owned();
        let edge_raw = outputs[edge_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr edges: {e}")))?
            .into_owned();
        let edge_logits = edge_raw.into_owned();

        Ok(PotatrOutputs {
            node_logits,
            boxes,
            edge_logits,
            geometry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_geometry_pillow_rounding() {
        // 1500×900: scale = 1000/1500; PIL rounds each dim independently.
        let img = DynamicImage::new_rgb8(1500, 900);
        let (_, geo) = preprocess(&img).expect("preprocess");
        assert_eq!((geo.resized_w, geo.resized_h), (1000, 600));
        assert!((geo.scale - 1000.0 / 1500.0).abs() < 1e-6);
        assert_eq!((geo.orig_w, geo.orig_h), (1500, 900));
    }

    #[test]
    fn test_preprocess_rejects_empty() {
        // Zero-size images cannot exist via DynamicImage constructors, so
        // exercise the guard through a degenerate 1px input instead: it
        // must upscale, not divide by zero.
        let img = DynamicImage::new_rgb8(1, 1);
        let (arr, geo) = preprocess(&img).expect("preprocess");
        assert_eq!((geo.resized_w, geo.resized_h), (1000, 1000));
        assert_eq!(arr.dim(), (1, 3, 1000, 1000));
    }
}

// ---------------------------------------------------------------------------
// P1: post-processor port (`postprocess.py` + `metrics.cells_to_html` +
// `inference.linked_text_parts` / `tables_from_outputs`). Deterministic
// geometry only — no ML. Faithful to the reference, including its quirks
// (noted inline); behavior deltas are bugs, not improvements.
// ---------------------------------------------------------------------------

/// A bounding box, [x0, y0, x1, y1] in pixels.
pub type BBox = [f32; 4];

/// Class names in query-index order (`postprocess.class_idx2name`).
pub const CLASS_IDX2NAME: [&str; 17] = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
    "table caption",
    "table footer",
    "table rotated",
    "table column rotated",
    "table row rotated",
    "table column header rotated",
    "table projected row header rotated",
    "table spanning cell rotated",
    "table caption rotated",
    "table footer rotated",
    "no object",
];

/// Internal thresholds: every real class 0.8, "no object" effectively never
/// (`postprocess.class_thresholds`). These govern the *structure* refinement
/// and stay fixed for fidelity; `potatr_node_threshold` (config) filters the
/// *final table claims* in P2 routing instead.
fn class_threshold(label: &str) -> f32 {
    if label == "no object" {
        10.0
    } else {
        0.8
    }
}

/// A detected object, word token, or table structure. Python dicts with
/// optional keys become one struct with defaults (`..Default::default()`):
/// tokens carry block/line/span order keys, structures carry header flags
/// and grid numbers; unused fields stay at their defaults.
#[derive(Debug, Clone, Default)]
pub struct PotatrObject {
    /// Query index (detections); `usize::MAX` for word tokens.
    pub idx: usize,
    /// Class label ("table", "table row", …); empty for word tokens.
    pub label: String,
    /// Word text for tokens; empty for detections.
    pub text: String,
    pub score: f32,
    pub bbox: BBox,
    // Word-token order keys (`extract_text_from_spans` sorts by these).
    pub block_num: i64,
    pub line_num: i64,
    pub span_num: i64,
    // Structure flags set during refinement.
    pub header: bool,
    pub subheader: bool,
    /// `"span" in supercell` — never set on the inference path (no
    /// caller constructs one); kept so the `align_supercells` span branches
    /// port verbatim instead of being silently dropped.
    pub is_span: bool,
    pub row_numbers: Vec<usize>,
    pub column_numbers: Vec<usize>,
    pub propagated: bool,
}

impl PotatrObject {
    /// A page word token (`ikat_grits_gate.page_words` shape).
    pub fn token(text: &str, text_bbox: BBox, block_num: i64, line_num: i64, span_num: i64) -> Self {
        Self {
            idx: usize::MAX,
            text: text.to_string(),
            bbox: text_bbox,
            block_num,
            line_num,
            span_num,
            ..Default::default()
        }
    }
}

/// One table cell with its text (`table_structure_to_cells`).
#[derive(Debug, Clone, Default)]
pub struct PotatrCell {
    pub bbox: BBox,
    pub column_nums: Vec<usize>,
    pub row_nums: Vec<usize>,
    pub header: bool,
    pub subheader: bool,
    pub cell_text: String,
    /// Word tokens slotted into the cell (reference `cell["spans"]`).
    pub spans: Vec<PotatrObject>,
}

/// Caption/footer text linked to a table with its source object.
#[derive(Debug, Clone, Default)]
pub struct LinkedText {
    pub text: String,
    pub bbox: BBox,
    pub score: f32,
    pub edge_score: f32,
}

/// One page table (`inference.tables_from_outputs`).
#[derive(Debug, Clone, Default)]
pub struct PotatrTable {
    pub html: String,
    pub cells: Vec<PotatrCell>,
    pub caption: String,
    pub footer: String,
    pub caption_parts: Vec<LinkedText>,
    pub footer_parts: Vec<LinkedText>,
    pub label: String,
    pub bbox: BBox,
    pub score: f32,
}

fn bbox_area(b: &BBox) -> f32 {
    ((b[2] - b[0]).max(0.0)) * ((b[3] - b[1]).max(0.0))
}

fn bbox_intersect(a: &BBox, b: &BBox) -> BBox {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]
}

/// Union accumulator with `None` as identity (`_bbox_include`).
fn bbox_include(current: Option<BBox>, b: &BBox) -> BBox {
    match current {
        None => *b,
        Some(c) => [c[0].min(b[0]), c[1].min(b[1]), c[2].max(b[2]), c[3].max(b[3])],
    }
}

/// Fraction of `a` covered by `b` (`overlaps`, threshold-externalized).
fn overlaps_frac(a: &BBox, b: &BBox) -> f32 {
    let area = bbox_area(a);
    if area == 0.0 {
        return 0.0;
    }
    bbox_area(&bbox_intersect(a, b)) / area
}

/// Test if at least `threshold` fraction of `a` overlaps `b` (`overlaps`).
fn overlaps(a: &BBox, b: &BBox, threshold: f32) -> bool {
    overlaps_frac(a, b) >= threshold
}

/// Intersection area over the area of `a` (`iob`).
fn iob(a: &BBox, b: &BBox) -> f32 {
    let area = bbox_area(a);
    if area > 0.0 {
        bbox_area(&bbox_intersect(a, b)) / area
    } else {
        0.0
    }
}

/// Filter out objects below a score (`apply_threshold`).
fn apply_threshold(objects: &[PotatrObject], threshold: f32) -> Vec<PotatrObject> {
    objects
        .iter()
        .filter(|o| o.score >= threshold)
        .cloned()
        .collect()
}

fn sort_by_score_desc(objects: &mut [PotatrObject]) {
    objects.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn sort_left_to_right(objects: &mut [PotatrObject]) {
    objects.sort_by(|a, b| {
        (a.bbox[0] + a.bbox[2])
            .partial_cmp(&(b.bbox[0] + b.bbox[2]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn sort_top_to_bottom(objects: &mut [PotatrObject]) {
    objects.sort_by(|a, b| {
        (a.bbox[1] + a.bbox[3])
            .partial_cmp(&(b.bbox[1] + b.bbox[3]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Reduce the spans to those at least `threshold`-inside `bbox`
/// (`get_bbox_span_subset`).
fn get_bbox_span_subset(
    spans: &[PotatrObject],
    bbox: &BBox,
    threshold: f32,
) -> Vec<PotatrObject> {
    spans
        .iter()
        .filter(|s| overlaps(&s.bbox, bbox, threshold))
        .cloned()
        .collect()
}

/// Convert page tokens into one text string (`extract_text_from_spans`),
/// block/line/span ordered with the reference's end-of-line spacing rule
/// (including its hyphen exception and its join-with-empty quirk — verbatim).
fn extract_text_from_spans(spans: &[PotatrObject], join_with_space: bool) -> String {
    if spans.is_empty() {
        return String::new();
    }
    let join_char = if join_with_space { " " } else { "" };
    // Three successive stable sorts (span, line, block) == one (block, line, span) key.
    let mut ordered: Vec<&PotatrObject> = spans.iter().collect();
    ordered.sort_by(|a, b| {
        (a.block_num, a.line_num, a.span_num).cmp(&(b.block_num, b.line_num, b.span_num))
    });
    let mut line_texts: Vec<String> = Vec::new();
    let mut line_span_texts = vec![ordered[0].text.trim().to_string()];
    for pair in ordered.windows(2) {
        let (s1, s2) = (pair[0], pair[1]);
        if s1.block_num != s2.block_num || s1.line_num != s2.line_num {
            let mut line_text = line_span_texts.join(join_char);
            line_text = line_text.trim().to_string();
            if !line_text.is_empty()
                && !line_text.ends_with(' ')
                && !(line_text.len() > 1
                    && line_text.ends_with('-')
                    && !line_text[..line_text.len() - 1].ends_with(' '))
            {
                if !join_with_space {
                    line_text.push(' ');
                }
            }
            line_texts.push(line_text);
            line_span_texts = vec![s2.text.trim().to_string()];
        } else {
            line_span_texts.push(s2.text.trim().to_string());
        }
    }
    line_texts.push(line_span_texts.join(join_char));
    line_texts.join(join_char).trim().to_string()
}

/// Text inside a bbox plus the spans it came from (`extract_text_inside_bbox`).
fn extract_text_inside_bbox(
    spans: &[PotatrObject],
    bbox: &BBox,
) -> (String, Vec<PotatrObject>) {
    let subset = get_bbox_span_subset(spans, bbox, 0.5);
    let text = extract_text_from_spans(&subset, true);
    (text, subset)
}

/// Remove, in place, objects containing no text (`remove_objects_without_content`).
fn remove_objects_without_content(page_spans: &[PotatrObject], objects: &mut Vec<PotatrObject>) {
    objects.retain(|obj| {
        let (text, _) = extract_text_inside_bbox(page_spans, &obj.bbox);
        !text.trim().is_empty()
    });
}

/// Slot each package into the container holding its largest area fraction
/// (`slot_into_containers`). Returns (package indices by container,
/// container indices by package, each package's largest fraction).
fn slot_into_containers(
    containers: &[PotatrObject],
    packages: &[PotatrObject],
    overlap_threshold: f32,
    unique_assignment: bool,
    forced_assignment: bool,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<f32>) {
    let mut by_container: Vec<Vec<usize>> =
        (0..containers.len()).map(|_| Vec::new()).collect();
    let mut by_package: Vec<Vec<usize>> = (0..packages.len()).map(|_| Vec::new()).collect();
    let mut best_scores: Vec<f32> = Vec::new();
    if containers.is_empty() || packages.is_empty() {
        return (by_container, by_package, best_scores);
    }
    for (package_num, package) in packages.iter().enumerate() {
        let package_area = bbox_area(&package.bbox);
        // (container index, overlap fraction), best first.
        let mut scored: Vec<(usize, f32)> = containers
            .iter()
            .enumerate()
            .map(|(ci, c)| {
                let inter = bbox_area(&bbox_intersect(&c.bbox, &package.bbox));
                (ci, if package_area > 0.0 { inter / package_area } else { 0.0 })
            })
            .collect();
        // Reference sorts dicts by score desc (stable); index tiebreak keeps
        // container order, matching Python's stable sort on equal scores.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let (best_ci, best_score) = scored[0];
        best_scores.push(best_score);
        if forced_assignment || best_score >= overlap_threshold {
            by_container[best_ci].push(package_num);
            by_package[package_num].push(best_ci);
        }
        if !unique_assignment {
            for &(ci, s) in &scored[1..] {
                if s >= overlap_threshold {
                    by_container[ci].push(package_num);
                    by_package[package_num].push(ci);
                } else {
                    break;
                }
            }
        }
    }
    (by_container, by_package, best_scores)
}

/// NMS by shared package containment (`nms_by_containment`).
fn nms_by_containment(
    containers: &[PotatrObject],
    packages: &[PotatrObject],
    overlap_threshold: f32,
) -> Vec<PotatrObject> {
    let mut sorted: Vec<PotatrObject> = containers.to_vec();
    sort_by_score_desc(&mut sorted);
    let (by_container, _, _) = slot_into_containers(
        &sorted,
        packages,
        overlap_threshold,
        true,
        false,
    );
    let n = sorted.len();
    let mut suppressed = vec![false; n];
    for o2 in 1..n {
        let set2: std::collections::HashSet<usize> = by_container[o2].iter().copied().collect();
        if set2.is_empty() {
            suppressed[o2] = true;
        }
        for o1 in 0..o2 {
            if suppressed[o1] {
                continue;
            }
            let set1: std::collections::HashSet<usize> =
                by_container[o1].iter().copied().collect();
            if !set1.is_disjoint(&set2) {
                suppressed[o2] = true;
            }
        }
    }
    sorted
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !suppressed[*i])
        .map(|(_, o)| o)
        .collect()
}

/// Suppress overlapping rows / drop textless ones (`refine_rows`).
fn refine_rows(rows: &[PotatrObject], tokens: &[PotatrObject]) -> Vec<PotatrObject> {
    let mut out = if !tokens.is_empty() {
        let kept = nms_by_containment(rows, tokens, 0.5);
        let mut kept = kept;
        remove_objects_without_content(tokens, &mut kept);
        kept
    } else {
        nms(rows, MatchCriteria::Object2Overlap, 0.5, true)
    };
    if out.len() > 1 {
        sort_top_to_bottom(&mut out);
    }
    out
}

/// Suppress overlapping columns / drop textless ones (`refine_columns`).
fn refine_columns(columns: &[PotatrObject], tokens: &[PotatrObject]) -> Vec<PotatrObject> {
    let mut out = if !tokens.is_empty() {
        let kept = nms_by_containment(columns, tokens, 0.5);
        let mut kept = kept;
        remove_objects_without_content(tokens, &mut kept);
        kept
    } else {
        nms(columns, MatchCriteria::Object2Overlap, 0.25, true)
    };
    if out.len() > 1 {
        sort_left_to_right(&mut out);
    }
    out
}

/// How two boxes "overlap" for NMS (`match_criteria`). Only Object2Overlap
/// fires on the inference path; the others port the reference API.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum MatchCriteria {
    Object1Overlap,
    Object2Overlap,
    Iou,
}

/// Customizable NMS (`nms`): a lower-scored object overlapping at least
/// `threshold` (by the criterion) with a higher-scored one is removed.
/// Zero-area boxes yield no metric and never suppress (ZeroDivisionError).
fn nms(
    objects: &[PotatrObject],
    criteria: MatchCriteria,
    threshold: f32,
    keep_higher: bool,
) -> Vec<PotatrObject> {
    if objects.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<PotatrObject> = objects.to_vec();
    if keep_higher {
        sort_by_score_desc(&mut sorted);
    } else {
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let n = sorted.len();
    let mut suppressed = vec![false; n];
    for o2 in 1..n {
        let area2 = bbox_area(&sorted[o2].bbox);
        for o1 in 0..o2 {
            if suppressed[o1] {
                continue;
            }
            let area1 = bbox_area(&sorted[o1].bbox);
            let inter = bbox_area(&bbox_intersect(&sorted[o1].bbox, &sorted[o2].bbox));
            let metric = match criteria {
                MatchCriteria::Object1Overlap => {
                    if area1 > 0.0 {
                        Some(inter / area1)
                    } else {
                        None
                    }
                }
                MatchCriteria::Object2Overlap => {
                    if area2 > 0.0 {
                        Some(inter / area2)
                    } else {
                        None
                    }
                }
                MatchCriteria::Iou => {
                    let union = area1 + area2 - inter;
                    if union > 0.0 {
                        Some(inter / union)
                    } else {
                        None
                    }
                }
            };
            if let Some(m) = metric {
                if m >= threshold {
                    suppressed[o2] = true;
                    break;
                }
            }
        }
    }
    sorted
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !suppressed[*i])
        .map(|(_, o)| o)
        .collect()
}

/// Stretch every column to the table box vertically (`align_columns`).
fn align_columns(columns: &mut [PotatrObject], bbox: &BBox) {
    for c in columns.iter_mut() {
        c.bbox[1] = bbox[1];
        c.bbox[3] = bbox[3];
    }
}

/// Stretch every row to the table box horizontally (`align_rows`).
fn align_rows(rows: &mut [PotatrObject], bbox: &BBox) {
    for r in rows.iter_mut() {
        r.bbox[0] = bbox[0];
        r.bbox[2] = bbox[2];
    }
}

/// Align each supercell to the rows/columns it covers (`align_supercells`):
/// a supercell covers a row with ≥ 50% height overlap (span supercells take
/// the max of the two directions), cannot cross the header boundary, and
/// survives only if it joins ≥ 2 grid cells. Span supercells in the header
/// propagate upward copies (reference quirk, ported verbatim).
fn align_supercells(
    supercells: &[PotatrObject],
    rows: &[PotatrObject],
    columns: &[PotatrObject],
) -> Vec<PotatrObject> {
    let mut aligned: Vec<PotatrObject> = Vec::new();
    for sc in supercells {
        let mut supercell = sc.clone();
        supercell.header = false;
        let mut row_rect: Option<BBox> = None;
        let mut col_rect: Option<BBox> = None;
        let mut header_rows: std::collections::HashSet<usize> = Default::default();
        let mut data_rows: std::collections::HashSet<usize> = Default::default();
        for (row_num, row) in rows.iter().enumerate() {
            let row_h = row.bbox[3] - row.bbox[1];
            let sc_h = supercell.bbox[3] - supercell.bbox[1];
            let oh = (row.bbox[3].min(supercell.bbox[3]) - row.bbox[1].max(supercell.bbox[1]))
                .max(0.0);
            // Reference divides unguarded; zero heights crash it. Rust guards
            // (documented delta: degenerate inputs keep the supercell instead
            // of raising). In practice model boxes have positive area.
            let frac = if supercell.is_span {
                let a = if row_h > 0.0 { oh / row_h } else { 0.0 };
                let b = if sc_h > 0.0 { oh / sc_h } else { 0.0 };
                a.max(b)
            } else if row_h > 0.0 {
                oh / row_h
            } else {
                0.0
            };
            if frac >= 0.5 {
                if row.header {
                    header_rows.insert(row_num);
                } else {
                    data_rows.insert(row_num);
                }
            }
        }
        // Header boundary: drop the smaller group.
        supercell.header = false;
        if !data_rows.is_empty() && !header_rows.is_empty() {
            if data_rows.len() > header_rows.len() {
                header_rows.clear();
            } else {
                data_rows.clear();
            }
        }
        if !header_rows.is_empty() {
            supercell.header = true;
        } else if supercell.is_span {
            continue; // span supercells must be in the header
        }
        let intersecting: std::collections::HashSet<usize> =
            data_rows.union(&header_rows).copied().collect();
        for &rn in &intersecting {
            row_rect = Some(bbox_include(row_rect, &rows[rn].bbox));
        }
        let Some(row_rect) = row_rect else { continue };
        let mut intersecting_cols: Vec<usize> = Vec::new();
        for (col_num, col) in columns.iter().enumerate() {
            let col_w = col.bbox[2] - col.bbox[0];
            let sc_w = supercell.bbox[2] - supercell.bbox[0];
            let ow = (col.bbox[2].min(supercell.bbox[2]) - col.bbox[0].max(supercell.bbox[0]))
                .max(0.0);
            let frac = if supercell.is_span {
                let a = if col_w > 0.0 { ow / col_w } else { 0.0 };
                let b = if sc_w > 0.0 { ow / sc_w } else { 0.0 };
                let mut f = a.max(b);
                if supercell.header {
                    f *= 2.0; // effectively lowers the threshold to 0.25
                }
                f
            } else if col_w > 0.0 {
                ow / col_w
            } else {
                0.0
            };
            if frac >= 0.5 {
                intersecting_cols.push(col_num);
                col_rect = Some(bbox_include(col_rect, &col.bbox));
            }
        }
        let Some(col_rect) = col_rect else { continue };
        supercell.bbox = bbox_intersect(&row_rect, &col_rect);
        if !intersecting.is_empty()
            && !intersecting_cols.is_empty()
            && (intersecting.len() > 1 || intersecting_cols.len() > 1)
        {
            supercell.row_numbers = intersecting.into_iter().collect();
            supercell.column_numbers = intersecting_cols;
            let n_cols = supercell.column_numbers.len();
            let min_row = *supercell.row_numbers.iter().min().unwrap_or(&0);
            aligned.push(supercell.clone());
            if supercell.is_span && supercell.header && n_cols > 1 {
                for rn in 0..min_row {
                    let cols: Vec<&BBox> = supercell
                        .column_numbers
                        .iter()
                        .map(|&c| &columns[c].bbox)
                        .collect();
                    let rows_sel: Vec<&BBox> = supercell
                        .row_numbers
                        .iter()
                        .map(|&r| &rows[r].bbox)
                        .collect();
                    let bbox = [
                        cols.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min),
                        rows_sel.iter().map(|b| b[1]).fold(f32::INFINITY, f32::min),
                        cols.iter().map(|b| b[2]).fold(f32::NEG_INFINITY, f32::max),
                        rows_sel.iter().map(|b| b[3]).fold(f32::NEG_INFINITY, f32::max),
                    ];
                    aligned.push(PotatrObject {
                        row_numbers: vec![rn],
                        column_numbers: supercell.column_numbers.clone(),
                        score: supercell.score,
                        propagated: true,
                        bbox,
                        ..Default::default()
                    });
                }
            }
        }
    }
    aligned
}

/// Shrink `s2` until disjoint from `s1` (`remove_supercell_overlap`), one
/// edge row/column at a time (columns first when wider than tall).
fn remove_supercell_overlap(s1: &PotatrObject, s2: &mut PotatrObject) {
    let mut common_rows: std::collections::HashSet<usize> = s1
        .row_numbers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .intersection(&s2.row_numbers.iter().copied().collect())
        .copied()
        .collect();
    let mut common_cols: std::collections::HashSet<usize> = s1
        .column_numbers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .intersection(&s2.column_numbers.iter().copied().collect())
        .copied()
        .collect();
    while !common_rows.is_empty() && !common_cols.is_empty() {
        if s2.row_numbers.len() < s2.column_numbers.len() {
            let (Some(&min_c), Some(&max_c)) = (
                s2.column_numbers.iter().min(),
                s2.column_numbers.iter().max(),
            ) else {
                break;
            };
            if common_cols.remove(&max_c) {
                s2.column_numbers.retain(|&c| c != max_c);
            } else if common_cols.remove(&min_c) {
                s2.column_numbers.retain(|&c| c != min_c);
            } else {
                s2.column_numbers.clear();
                common_cols.clear();
            }
        } else {
            let (Some(&min_r), Some(&max_r)) = (
                s2.row_numbers.iter().min(),
                s2.row_numbers.iter().max(),
            ) else {
                break;
            };
            if common_rows.remove(&max_r) {
                s2.row_numbers.retain(|&r| r != max_r);
            } else if common_rows.remove(&min_r) {
                s2.row_numbers.retain(|&r| r != min_r);
            } else {
                s2.row_numbers.clear();
                common_rows.clear();
            }
        }
    }
}

/// Suppress overlapping supercells, shrinking first (`nms_supercells`).
fn nms_supercells(supercells: &[PotatrObject]) -> Vec<PotatrObject> {
    let mut sorted: Vec<PotatrObject> = supercells.to_vec();
    sort_by_score_desc(&mut sorted);
    let n = sorted.len();
    let mut suppressed = vec![false; n];
    for i2 in 1..n {
        for i1 in 0..i2 {
            // Same-vec borrow: clone the winner's grid (tiny vecs).
            let s1 = sorted[i1].clone();
            remove_supercell_overlap(&s1, &mut sorted[i2]);
        }
        let s2 = &sorted[i2];
        if (s2.row_numbers.len() < 2 && s2.column_numbers.len() < 2)
            || s2.row_numbers.is_empty()
            || s2.column_numbers.is_empty()
        {
            suppressed[i2] = true;
        }
    }
    sorted
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !suppressed[*i])
        .map(|(_, o)| o)
        .collect()
}

/// Remove header supercells that break the header tree (`header_supercell_tree`):
/// each must sit below exactly one ancestor covering its columns in every row above.
fn header_supercell_tree(supercells: &mut Vec<PotatrObject>) {
    let header: Vec<PotatrObject> = supercells
        .iter()
        .filter(|s| s.header)
        .cloned()
        .collect();
    let mut ordered = header;
    sort_by_score_desc(&mut ordered);
    let mut drop_idx: Vec<usize> = Vec::new();
    for (hi, hs) in ordered.iter().enumerate() {
        let min_row = hs.row_numbers.iter().copied().min().unwrap_or(0);
        let mut ancestors: std::collections::HashMap<usize, usize> = Default::default();
        for hs2 in &ordered {
            let max2 = hs2.row_numbers.iter().copied().max().unwrap_or(0);
            if max2 < min_row {
                let cols: std::collections::HashSet<usize> =
                    hs.column_numbers.iter().copied().collect();
                let cols2: std::collections::HashSet<usize> =
                    hs2.column_numbers.iter().copied().collect();
                if cols.is_subset(&cols2) {
                    for &r2 in &hs2.row_numbers {
                        *ancestors.entry(r2).or_insert(0) += 1;
                    }
                }
            }
        }
        for r in 0..min_row {
            if ancestors.get(&r).copied().unwrap_or(0) != 1 {
                drop_idx.push(hi);
                break;
            }
        }
    }
    // Map back to `supercells` positions via (bbox, rows, cols) identity.
    let mut remove: Vec<usize> = Vec::new();
    for hi in drop_idx {
        let hs = &ordered[hi];
        if let Some(pos) = supercells.iter().position(|s| {
            s.header
                && s.bbox == hs.bbox
                && s.row_numbers == hs.row_numbers
                && s.column_numbers == hs.column_numbers
        }) {
            if !remove.contains(&pos) {
                remove.push(pos);
            }
        }
    }
    remove.sort_unstable();
    for pos in remove.into_iter().rev() {
        supercells.remove(pos);
    }
}

/// Rotate boxes 90° clockwise with the page, in place, stripping " rotated"
/// from labels (`rotate_objects_90`); `img_size` is the page's (width, height).
fn rotate_objects_90(_img_w: f32, img_h: f32, objs: &mut [PotatrObject]) {
    for o in objs.iter_mut() {
        o.label = o.label.replace(" rotated", "");
        o.bbox = [img_h - o.bbox[3], o.bbox[0], img_h - o.bbox[1], o.bbox[2]];
    }
}

/// Undo `rotate_objects_90` (`rotate_objects_270`); size is the *rotated*
/// page's (width, height). Labels untouched. Object version: unused on the
/// inference path (cells take `rotate_objects_270_cells`); kept for port
/// completeness — the reference shares one function for both.
#[allow(dead_code)]
fn rotate_objects_270(img_w: f32, _img_h: f32, objs: &mut [PotatrObject]) {
    for o in objs.iter_mut() {
        o.bbox = [o.bbox[1], img_w - o.bbox[2], o.bbox[3], img_w - o.bbox[0]];
    }
}

/// One header box covering the header rows from the top (`align_headers`);
/// marks each row's `header` flag. Only one header per table.
fn align_headers(headers: &[PotatrObject], rows: &mut [PotatrObject]) -> Vec<PotatrObject> {
    for r in rows.iter_mut() {
        r.header = false;
    }
    let mut header_row_nums: Vec<usize> = Vec::new();
    for h in headers {
        for (row_num, row) in rows.iter().enumerate() {
            let row_h = row.bbox[3] - row.bbox[1];
            let raw_oh = row.bbox[3].min(h.bbox[3]) - row.bbox[1].max(h.bbox[1]);
            // Reference divides unguarded: raw_oh / row_h. Zero-height rows
            // with zero overlap crash it (0.0/0.0 ZeroDivisionError); Rust
            // yields NaN, which compares false — identical everywhere else
            // (positive overlap over zero height is inf >= 0.5 in both).
            let ratio = if row_h == 0.0 && raw_oh == 0.0 {
                f32::NAN
            } else {
                raw_oh / row_h
            };
            if ratio >= 0.5 {
                header_row_nums.push(row_num);
            }
        }
    }
    if header_row_nums.is_empty() {
        return Vec::new();
    }
    let mut rect: Option<BBox> = None;
    if header_row_nums[0] > 0 {
        header_row_nums = (0..=header_row_nums[0]).chain(header_row_nums).collect();
    }
    let mut last_row: isize = -1;
    for &row_num in &header_row_nums {
        if row_num as isize == last_row + 1 {
            rows[row_num].header = true;
            rect = Some(bbox_include(rect, &rows[row_num].bbox));
            last_row = row_num as isize;
        } else {
            break; // later header rows dropped — one header per table
        }
    }
    vec![PotatrObject {
        bbox: rect.unwrap_or([0.0, 0.0, 0.0, 0.0]),
        ..Default::default()
    }]
}

/// Refined table structures: thresholded rows/columns/headers/supercells.
#[derive(Debug, Clone, Default)]
pub struct TableStructures {
    pub rows: Vec<PotatrObject>,
    pub columns: Vec<PotatrObject>,
    pub headers: Vec<PotatrObject>,
    pub supercells: Vec<PotatrObject>,
}

/// Threshold, align and suppress headers and supercells (`refine_table_structures`).
fn refine_table_structures(mut structs: TableStructures) -> TableStructures {
    let mut headers = apply_threshold(&structs.headers, class_threshold("table column header"));
    headers = nms(&headers, MatchCriteria::Object2Overlap, 0.05, true);
    headers = align_headers(&headers, &mut structs.rows);
    let mut supercells: Vec<PotatrObject> = structs
        .supercells
        .iter()
        .filter(|s| !s.subheader)
        .cloned()
        .collect();
    let subheaders: Vec<PotatrObject> = structs
        .supercells
        .iter()
        .filter(|s| s.subheader)
        .cloned()
        .collect();
    supercells = apply_threshold(&supercells, class_threshold("table spanning cell"));
    let mut subheaders = apply_threshold(&subheaders, class_threshold("table projected row header"));
    supercells.append(&mut subheaders);
    supercells = align_supercells(&supercells, &structs.rows, &structs.columns);
    supercells = nms_supercells(&supercells);
    let mut supercells = supercells;
    header_supercell_tree(&mut supercells);
    structs.supercells = supercells;
    structs.headers = headers;
    structs
}

/// Refined, consistent structures → text-bearing cells (`table_structure_to_cells`).
/// Returns (cells, confidence score from word slotting).
fn table_structure_to_cells(
    structs: &TableStructures,
    table_spans: &[PotatrObject],
) -> (Vec<PotatrCell>, f32) {
    let columns = &structs.columns;
    let rows = &structs.rows;
    let supercells = &structs.supercells;
    let mut cells: Vec<PotatrCell> = Vec::new();
    let mut subcells: Vec<PotatrCell> = Vec::new();
    for (column_num, column) in columns.iter().enumerate() {
        for (row_num, row) in rows.iter().enumerate() {
            let cell_rect = bbox_intersect(&row.bbox, &column.bbox);
            let header = row.header;
            let mut cell = PotatrCell {
                bbox: cell_rect,
                column_nums: vec![column_num],
                row_nums: vec![row_num],
                header,
                ..Default::default()
            };
            let area = bbox_area(&cell_rect);
            let mut is_sub = false;
            if area > 0.0 {
                for sc in supercells {
                    if bbox_area(&bbox_intersect(&sc.bbox, &cell_rect)) / area > 0.5 {
                        is_sub = true;
                        break;
                    }
                }
            }
            if is_sub {
                subcells.push(cell);
            } else {
                cell.subheader = false;
                cells.push(cell);
            }
        }
    }
    for sc in supercells {
        let mut cell_rows: std::collections::HashSet<usize> = Default::default();
        let mut cell_cols: std::collections::HashSet<usize> = Default::default();
        let mut sc_rect: Option<BBox> = None;
        let mut header = true;
        for sub in &subcells {
            let sub_area = bbox_area(&sub.bbox);
            if sub_area > 0.0
                && bbox_area(&bbox_intersect(&sub.bbox, &sc.bbox)) / sub_area > 0.5
            {
                sc_rect = Some(bbox_include(sc_rect, &sub.bbox));
                cell_rows.extend(sub.row_nums.iter().copied());
                cell_cols.extend(sub.column_nums.iter().copied());
                header = header && sub.header;
            }
        }
        if sc_rect.is_some() && !cell_rows.is_empty() && !cell_cols.is_empty() {
            cells.push(PotatrCell {
                bbox: sc_rect.unwrap(),
                column_nums: cell_cols.into_iter().collect(),
                row_nums: cell_rows.into_iter().collect(),
                header,
                subheader: sc.subheader,
                ..Default::default()
            });
        }
    }
    // Confidence from word slotting.
    let (_, _, match_scores) = slot_into_containers(
        &cells_as_objects(&cells),
        table_spans,
        0.5,
        true,
        false,
    );
    let confidence = if match_scores.is_empty() {
        0.0
    } else {
        let mean = match_scores.iter().sum::<f32>() / match_scores.len() as f32;
        let min = match_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        (mean + min) / 2.0
    };
    // Dilate rows/columns before final extraction.
    let mut dilated_rows = rows.clone();
    let mut dilated_columns = columns.clone();
    for cell in &mut cells {
        let mut col_rect: Option<BBox> = None;
        for &cn in &cell.column_nums {
            col_rect = Some(bbox_include(col_rect, &dilated_columns[cn].bbox));
        }
        let mut row_rect: Option<BBox> = None;
        for &rn in &cell.row_nums {
            row_rect = Some(bbox_include(row_rect, &dilated_rows[rn].bbox));
        }
        // Every cell has ≥ 1 row and column (reference asserts).
        cell.bbox = bbox_intersect(
            &col_rect.expect("cell column"),
            &row_rect.expect("cell row"),
        );
    }
    let (by_cell, _, _) = slot_into_containers(
        &cells_as_objects(&cells),
        table_spans,
        0.001,
        true,
        false,
    );
    for (cell, span_nums) in cells.iter_mut().zip(by_cell.iter()) {
        cell.spans = span_nums.iter().map(|&n| table_spans[n].clone()).collect();
        cell.cell_text = extract_text_from_spans(&cell.spans, true);
    }
    // Adjust row/column/cell boxes to the extracted text.
    let num_rows = dilated_rows.len();
    sort_top_to_bottom(&mut dilated_rows);
    let num_columns = dilated_columns.len();
    sort_left_to_right(&mut dilated_columns);
    let mut min_x_by_col: Vec<Vec<f32>> = (0..num_columns).map(|_| Vec::new()).collect();
    let mut max_x_by_col: Vec<Vec<f32>> = (0..num_columns).map(|_| Vec::new()).collect();
    let mut min_y_by_row: Vec<Vec<f32>> = (0..num_rows).map(|_| Vec::new()).collect();
    let mut max_y_by_row: Vec<Vec<f32>> = (0..num_rows).map(|_| Vec::new()).collect();
    // NOTE: each cell's spans were stored above; the adjustment reads them.
    for cell in &cells {
        let min_row = cell.row_nums.iter().copied().min().unwrap_or(0);
        let max_row = cell.row_nums.iter().copied().max().unwrap_or(0);
        let min_col = cell.column_nums.iter().copied().min().unwrap_or(0);
        let max_col = cell.column_nums.iter().copied().max().unwrap_or(0);
        for s in &cell.spans {
            min_x_by_col[min_col].push(s.bbox[0]);
            min_y_by_row[min_row].push(s.bbox[1]);
            max_x_by_col[max_col].push(s.bbox[2]);
            max_y_by_row[max_row].push(s.bbox[3]);
        }
    }
    for (row_num, row) in dilated_rows.iter_mut().enumerate() {
        if !min_x_by_col[0].is_empty() {
            row.bbox[0] = min_x_by_col[0].iter().fold(f32::INFINITY, |a, &b| a.min(b));
        }
        if !min_y_by_row[row_num].is_empty() {
            row.bbox[1] = min_y_by_row[row_num].iter().fold(f32::INFINITY, |a, &b| a.min(b));
        }
        if !max_x_by_col[num_columns - 1].is_empty() {
            row.bbox[2] = max_x_by_col[num_columns - 1]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        }
        if !max_y_by_row[row_num].is_empty() {
            row.bbox[3] = max_y_by_row[row_num].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        }
    }
    for (col_num, col) in dilated_columns.iter_mut().enumerate() {
        if !min_x_by_col[col_num].is_empty() {
            col.bbox[0] = min_x_by_col[col_num].iter().fold(f32::INFINITY, |a, &b| a.min(b));
        }
        if !min_y_by_row[0].is_empty() {
            col.bbox[1] = min_y_by_row[0].iter().fold(f32::INFINITY, |a, &b| a.min(b));
        }
        if !max_x_by_col[col_num].is_empty() {
            col.bbox[2] = max_x_by_col[col_num].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        }
        if !max_y_by_row[num_rows - 1].is_empty() {
            col.bbox[3] = max_y_by_row[num_rows - 1]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        }
    }
    for cell in &mut cells {
        let mut row_rect: Option<BBox> = None;
        for &rn in &cell.row_nums {
            row_rect = Some(bbox_include(row_rect, &dilated_rows[rn].bbox));
        }
        let mut col_rect: Option<BBox> = None;
        for &cn in &cell.column_nums {
            col_rect = Some(bbox_include(col_rect, &dilated_columns[cn].bbox));
        }
        if let (Some(rr), Some(cr)) = (row_rect, col_rect) {
            let rect = bbox_intersect(&rr, &cr);
            if bbox_area(&rect) > 0.0 {
                cell.bbox = rect;
            }
        }
    }
    (cells, confidence)
}

/// View cells as slot-containers (bbox only; text kept on the cells).
fn cells_as_objects(cells: &[PotatrCell]) -> Vec<PotatrObject> {
    cells
        .iter()
        .map(|c| PotatrObject {
            bbox: c.bbox,
            ..Default::default()
        })
        .collect()
}

/// Softmax over the last axis of a 2-D logits matrix (rows independent).
fn softmax_rows(logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
    logits
        .iter()
        .map(|row| {
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|&e| e / sum).collect()
        })
        .collect()
}

/// One image's raw outputs → detected objects (`potatr_outputs_to_objects`):
/// queries whose most likely class isn't "no object" with probability > 0.5,
/// boxes rescaled from normalized cxcyhw to pixels (`rescale_bboxes`).
pub fn potatr_outputs_to_objects(
    node_logits: &ndarray::ArrayD<f32>,
    boxes: &ndarray::ArrayD<f32>,
    img_w: f32,
    img_h: f32,
) -> Vec<PotatrObject> {
    let node_shape = node_logits.shape().to_vec();
    let box_shape = boxes.shape().to_vec();
    assert_eq!(node_shape.len(), 3, "node logits [1,Q,17]");
    assert_eq!(box_shape.len(), 3, "boxes [1,Q,4]");
    let (batch, n_queries, n_classes) = (node_shape[0], node_shape[1], node_shape[2]);
    assert_eq!(batch, 1);
    assert_eq!(box_shape, vec![1, n_queries, 4]);
    let mut objects = Vec::new();
    for q in 0..n_queries {
        let logits: Vec<f32> = (0..n_classes).map(|c| node_logits[[0, q, c]]).collect();
        let probs = softmax_rows(std::slice::from_ref(&logits))[0].clone();
        let (mut best_c, mut best_p) = (0, probs[0]);
        for (c, &p) in probs.iter().enumerate().skip(1) {
            if p > best_p {
                best_c = c;
                best_p = p;
            }
        }
        // Reference: torch.max over classes, first max wins on ties; the
        // strict `>` scan above reproduces first-max-wins.
        let label = CLASS_IDX2NAME.get(best_c).copied().unwrap_or("no object");
        if label == "no object" || best_p <= 0.5 {
            continue;
        }
        let cx = boxes[[0, q, 0]];
        let cy = boxes[[0, q, 1]];
        let w = boxes[[0, q, 2]];
        let h = boxes[[0, q, 3]];
        objects.push(PotatrObject {
            idx: q,
            label: label.to_string(),
            score: best_p,
            bbox: [
                (cx - 0.5 * w) * img_w,
                (cy - 0.5 * h) * img_h,
                (cx + 0.5 * w) * img_w,
                (cy + 0.5 * h) * img_h,
            ],
            ..Default::default()
        });
    }
    objects
}

/// Objects in one table → consistent structures (`potatr_objects_to_table_structures`).
fn potatr_objects_to_table_structures(
    objects_in_table: &[PotatrObject],
    tokens_in_table: &[PotatrObject],
) -> TableStructures {
    let columns: Vec<PotatrObject> = objects_in_table
        .iter()
        .filter(|o| o.label == "table column")
        .cloned()
        .collect();
    let mut rows: Vec<PotatrObject> = objects_in_table
        .iter()
        .filter(|o| o.label == "table row")
        .cloned()
        .collect();
    let headers: Vec<PotatrObject> = objects_in_table
        .iter()
        .filter(|o| o.label == "table column header")
        .cloned()
        .collect();
    let mut supercells: Vec<PotatrObject> = objects_in_table
        .iter()
        .filter(|o| o.label == "table spanning cell")
        .cloned()
        .collect();
    for s in supercells.iter_mut() {
        s.subheader = false;
    }
    let mut subheaders: Vec<PotatrObject> = objects_in_table
        .iter()
        .filter(|o| o.label == "table projected row header")
        .cloned()
        .collect();
    for s in subheaders.iter_mut() {
        s.subheader = true;
    }
    supercells.append(&mut subheaders);
    for row in rows.iter_mut() {
        row.header = false;
        for h in &headers {
            if iob(&row.bbox, &h.bbox) >= 0.5 {
                row.header = true;
            }
        }
    }
    let mut rows = refine_rows(&rows, tokens_in_table);
    let mut columns = refine_columns(&columns, tokens_in_table);
    // Shrink the table box to the rows' height × columns' width.
    let mut row_rect: Option<BBox> = None;
    for r in &rows {
        row_rect = Some(bbox_include(row_rect, &r.bbox));
    }
    let mut col_rect: Option<BBox> = None;
    for c in &columns {
        col_rect = Some(bbox_include(col_rect, &c.bbox));
    }
    let row_bbox = row_rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
    let col_bbox = col_rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
    let table_bbox = [col_bbox[0], row_bbox[1], col_bbox[2], row_bbox[3]];
    align_columns(&mut columns, &table_bbox);
    align_rows(&mut rows, &table_bbox);
    let mut structs = TableStructures {
        rows,
        columns,
        headers,
        supercells,
    };
    if !structs.rows.is_empty() && structs.columns.len() > 1 {
        structs = refine_table_structures(structs);
    }
    structs
}

/// Objects + words in one table → cells (`potatr_objects_to_cells`).
/// Tables with no rows or no columns get no cells and score 0.
fn potatr_objects_to_cells(
    objects_in_table: &[PotatrObject],
    tokens_in_table: &[PotatrObject],
) -> (TableStructures, Vec<PotatrCell>, f32) {
    let structs = potatr_objects_to_table_structures(objects_in_table, tokens_in_table);
    if structs.columns.is_empty() || structs.rows.is_empty() {
        (structs, Vec::new(), 0.0)
    } else {
        let (cells, score) = table_structure_to_cells(&structs, tokens_in_table);
        (structs, cells, score)
    }
}

/// One image's outputs + page words → each table's cells
/// (`potatr_outputs_to_tables_cells`), handling rotated tables by rotating
/// copies (the caller's words keep page coordinates).
fn potatr_outputs_to_tables_cells(
    node_logits: &ndarray::ArrayD<f32>,
    boxes: &ndarray::ArrayD<f32>,
    tokens: &[PotatrObject],
    img_w: f32,
    img_h: f32,
) -> Vec<Vec<PotatrCell>> {
    let objects = potatr_outputs_to_objects(node_logits, boxes, img_w, img_h);
    let table_objs: Vec<PotatrObject> = objects
        .iter()
        .filter(|o| o.label == "table" || o.label == "table rotated")
        .cloned()
        .collect();
    let mut out = Vec::new();
    for table in &table_objs {
        let in_table: Vec<PotatrObject> = objects
            .iter()
            .filter(|o| iob(&o.bbox, &table.bbox) >= 0.5)
            .cloned()
            .collect();
        let rotated = table.label == "table rotated";
        // Owned per-branch buffers (rotated branches work on copies so the
        // caller's words keep page coordinates).
        let (objects_ref, tokens_ref, table_bbox) = if rotated {
            let mut rot_table = vec![table.clone()];
            let mut rot_objs = in_table;
            let mut rot_tokens: Vec<PotatrObject> = tokens.to_vec();
            rotate_objects_90(img_w, img_h, &mut rot_table);
            rotate_objects_90(img_w, img_h, &mut rot_objs);
            rotate_objects_90(img_w, img_h, &mut rot_tokens);
            let bbox = rot_table[0].bbox;
            (rot_objs, rot_tokens, bbox)
        } else {
            (in_table, tokens.to_vec(), table.bbox)
        };
        let in_tokens: Vec<PotatrObject> = tokens_ref
            .iter()
            .filter(|t| iob(&t.bbox, &table_bbox) >= 0.5)
            .cloned()
            .collect();
        let (_, cells, _) = potatr_objects_to_cells(&objects_ref, &in_tokens);
        let mut cells = cells;
        if rotated {
            // Rotated page is (img_h, img_w): undo with swapped dims.
            rotate_objects_270_cells(img_h, img_w, &mut cells);
        }
        out.push(cells);
    }
    out
}

/// Undo `rotate_objects_90` on cells (labels untouched).
fn rotate_objects_270_cells(img_w: f32, img_h: f32, cells: &mut [PotatrCell]) {
    let _ = img_h;
    for c in cells.iter_mut() {
        c.bbox = [c.bbox[1], img_w - c.bbox[2], c.bbox[3], img_w - c.bbox[0]];
    }
}

/// Cells → HTML table (`metrics.cells_to_html`): first header row goes in
/// `<thead>` as `<th>`, the rest in `<tbody>` as `<td>`, with rowspan/colspan
/// for multi-grid cells. Row order = min row, then min column.
///
/// Fidelity notes: the *section* (thead/tbody) follows each row's first
/// cell, but the *tag* is per cell (a body-flagged cell in a header row
/// still emits `<td>`); empty text serializes as self-closing (`<th />`)
/// exactly like ElementTree; `& < >` are escaped, everything else raw.
fn potatr_cells_to_html(cells: &[PotatrCell]) -> String {
    if cells.is_empty() {
        return String::new();
    }
    let mut ordered: Vec<&PotatrCell> = cells.iter().collect();
    // Reference sorts by column first (stable), then by row (stable): net
    // (row, column). The single tuple sort below is the same order.
    ordered.sort_by(|a, b| {
        let ka = (
            a.row_nums.iter().copied().min().unwrap_or(0),
            a.column_nums.iter().copied().min().unwrap_or(0),
        );
        let kb = (
            b.row_nums.iter().copied().min().unwrap_or(0),
            b.column_nums.iter().copied().min().unwrap_or(0),
        );
        ka.cmp(&kb)
    });
    let mut html = String::from("<html><table>");
    let mut current_row = -1i64;
    let mut section: Option<bool> = None; // true = thead, false = tbody
    let mut tr_open = false;
    for cell in ordered {
        let this_row = cell.row_nums.iter().copied().min().unwrap_or(0) as i64;
        if this_row > current_row {
            current_row = this_row;
            if tr_open {
                html.push_str("</tr>");
            }
            if section != Some(cell.header) {
                match section {
                    Some(true) => html.push_str("</thead>"),
                    Some(false) => html.push_str("</tbody>"),
                    None => {}
                }
                html.push_str(if cell.header { "<thead>" } else { "<tbody>" });
                section = Some(cell.header);
            }
            html.push_str("<tr>");
            tr_open = true;
        }
        let tag = if cell.header { "th" } else { "td" };
        if cell.cell_text.is_empty() {
            // ElementTree self-closing form (`<th />`), byte parity.
            html.push('<');
            html.push_str(tag);
            html.push_str(" />");
            continue;
        }
        html.push('<');
        html.push_str(tag);
        if cell.column_nums.len() > 1 {
            html.push_str(&format!(" colspan=\"{}\"", cell.column_nums.len()));
        }
        if cell.row_nums.len() > 1 {
            html.push_str(&format!(" rowspan=\"{}\"", cell.row_nums.len()));
        }
        html.push('>');
        html.push_str(&escape_html(&cell.cell_text));
        html.push_str("</");
        html.push_str(tag);
        html.push('>');
    }
    if tr_open {
        html.push_str("</tr>");
    }
    match section {
        Some(true) => html.push_str("</thead>"),
        Some(false) => html.push_str("</tbody>"),
        None => {}
    }
    html.push_str("</table></html>");
    html
}

fn escape_html(s: &str) -> String {
    // ElementTree serializes text with & < > escaped (and no quotes in text
    // nodes). Mirror that so byte parity holds.
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

fn table_labels() -> [&'static str; 2] {
    ["table", "table rotated"]
}

fn caption_labels() -> [&'static str; 2] {
    ["table caption", "table caption rotated"]
}

fn footer_labels() -> [&'static str; 2] {
    ["table footer", "table footer rotated"]
}

/// Text of the objects of the given classes linked to a table
/// (`inference.linked_text_parts`): an object links when the mean of the two
/// edge probabilities is ≥ threshold; textless objects skipped; top-to-bottom.
fn linked_text_parts(
    table: &PotatrObject,
    objects: &[PotatrObject],
    edge_probs: &ndarray::ArrayD<f32>,
    tokens: &[PotatrObject],
    classes: &[&str],
    edge_threshold: f32,
) -> Vec<LinkedText> {
    let mut parts = Vec::new();
    for obj in objects {
        if !classes.contains(&obj.label.as_str()) {
            continue;
        }
        let p = (edge_probs[[table.idx, obj.idx]] + edge_probs[[obj.idx, table.idx]]) / 2.0;
        if p < edge_threshold {
            continue;
        }
        let (text, _) = extract_text_inside_bbox(tokens, &obj.bbox);
        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        parts.push(LinkedText {
            text,
            bbox: obj.bbox,
            score: obj.score,
            edge_score: p,
        });
    }
    parts.sort_by(|a, b| {
        a.bbox[1]
            .partial_cmp(&b.bbox[1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    parts
}

/// Edge probabilities per query pair: most-likely edge class over classes 1..
/// (`tables_from_outputs`: `softmax(-1)[:, :, 1:].max(-1)`). Class 0 is no-edge.
fn edge_probabilities(edge_logits: &ndarray::ArrayD<f32>) -> ndarray::ArrayD<f32> {
    let shape = edge_logits.shape().to_vec();
    assert_eq!(shape.len(), 4, "edge logits [1,Q,Q,2]");
    assert_eq!(shape[0], 1);
    let n = shape[1];
    assert_eq!(shape, vec![1, n, n, 2]);
    let mut probs = ndarray::ArrayD::<f32>::zeros(vec![n, n]);
    for i in 0..n {
        for j in 0..n {
            let l0 = edge_logits[[0, i, j, 0]];
            let l1 = edge_logits[[0, i, j, 1]];
            let m = l0.max(l1);
            let e0 = (l0 - m).exp();
            let e1 = (l1 - m).exp();
            // max over classes 1.. of a 2-class softmax = e1/(e0+e1).
            probs[[i, j]] = e1 / (e0 + e1);
        }
    }
    probs
}

/// Raw outputs + page words → page tables (`inference.tables_from_outputs`).
/// Each table has its HTML, its cells, caption/footer text with their source
/// objects, and its box and score. Tables follow object order (the reference
/// asserts `tables_cells` aligns with the table objects — `assert!` here too).
pub fn tables_from_outputs(
    node_logits: &ndarray::ArrayD<f32>,
    boxes: &ndarray::ArrayD<f32>,
    edge_logits: &ndarray::ArrayD<f32>,
    tokens: &[PotatrObject],
    img_w: f32,
    img_h: f32,
    edge_threshold: f32,
) -> Vec<PotatrTable> {
    let objects = potatr_outputs_to_objects(node_logits, boxes, img_w, img_h);
    let edge_probs = edge_probabilities(edge_logits);
    let tables_cells = potatr_outputs_to_tables_cells(node_logits, boxes, tokens, img_w, img_h);
    let table_objs: Vec<&PotatrObject> = objects
        .iter()
        .filter(|o| table_labels().contains(&o.label.as_str()))
        .collect();
    assert_eq!(
        table_objs.len(),
        tables_cells.len(),
        "tables_cells aligns with table objects"
    );
    let captions = caption_labels();
    let footers = footer_labels();
    table_objs
        .into_iter()
        .zip(tables_cells)
        .map(|(table, cells)| {
            let caption_parts =
                linked_text_parts(table, &objects, &edge_probs, tokens, &captions, edge_threshold);
            let footer_parts =
                linked_text_parts(table, &objects, &edge_probs, tokens, &footers, edge_threshold);
            let join = |parts: &[LinkedText]| {
                parts
                    .iter()
                    .map(|p| p.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            PotatrTable {
                html: if cells.is_empty() {
                    String::new()
                } else {
                    potatr_cells_to_html(&cells)
                },
                cells,
                caption: join(&caption_parts),
                footer: join(&footer_parts),
                caption_parts,
                footer_parts,
                label: table.label.clone(),
                bbox: table.bbox,
                score: table.score,
            }
        })
        .collect()
}

#[cfg(test)]
mod p1_tests {
    //! Ported from the reference `tests/test_postprocess.py` +
    //! `tests/test_inference.py` (same fixtures, same expectations).
    use super::*;
    use ndarray::ArrayD;

    const PAGE_W: f32 = 1000.0;
    const PAGE_H: f32 = 800.0;

    fn class_idx(label: &str) -> usize {
        CLASS_IDX2NAME.iter().position(|&l| l == label).unwrap()
    }

    /// Confident model outputs for (class, box, logit) objects + linked pairs,
    /// mirroring the reference `_outputs` helpers.
    fn make_outputs(
        objects: &[(&str, BBox, f32)],
        edges: &[(usize, usize)],
    ) -> (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) {
        let n = objects.len();
        let mut logits = ArrayD::<f32>::zeros(vec![1, n, CLASS_IDX2NAME.len()]);
        let mut boxes = ArrayD::<f32>::zeros(vec![1, n, 4]);
        for (i, (label, b, logit)) in objects.iter().enumerate() {
            logits[[0, i, class_idx(label)]] = *logit;
            boxes[[0, i, 0]] = (b[0] + b[2]) / 2.0 / PAGE_W;
            boxes[[0, i, 1]] = (b[1] + b[3]) / 2.0 / PAGE_H;
            boxes[[0, i, 2]] = (b[2] - b[0]) / PAGE_W;
            boxes[[0, i, 3]] = (b[3] - b[1]) / PAGE_H;
        }
        let mut edge = ArrayD::<f32>::zeros(vec![1, n, n, 2]);
        for i in 0..n {
            for j in 0..n {
                edge[[0, i, j, 0]] = 5.0;
            }
        }
        for &(i, j) in edges {
            edge[[0, i, j, 1]] = 10.0;
            edge[[0, j, i, 1]] = 10.0;
        }
        (logits, boxes, edge)
    }

    fn word(text: &str, bbox: BBox) -> PotatrObject {
        PotatrObject::token(text, bbox, 0, 0, 0)
    }

    fn approx(a: &BBox, b: &BBox) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-3)
    }

    #[test]
    fn rotated_table_keeps_caller_words_and_reads_cell() {
        // Single-cell rotated table; the rotated copies must not leak back.
        // (Rust borrows, so non-mutation is structural; the clone-equality
        // assertion documents the contract anyway.)
        let table_box = [100.0, 100.0, 300.0, 500.0];
        let (logits, boxes, _) = make_outputs(
            &[
                ("table rotated", table_box, 10.0),
                ("table row rotated", table_box, 10.0),
                ("table column rotated", table_box, 10.0),
            ],
            &[],
        );
        let tokens = vec![
            word("Revenue", [150.0, 200.0, 250.0, 300.0]),
            word("2024", [700.0, 300.0, 740.0, 320.0]),
        ];
        let before: Vec<(String, BBox)> =
            tokens.iter().map(|t| (t.text.clone(), t.bbox)).collect();
        let tables = potatr_outputs_to_tables_cells(&logits, &boxes, &tokens, PAGE_W, PAGE_H);
        let after: Vec<(String, BBox)> =
            tokens.iter().map(|t| (t.text.clone(), t.bbox)).collect();
        assert_eq!(before, after, "caller words unchanged");
        assert_eq!(tables.len(), 1);
        let texts: Vec<&str> = tables[0].iter().map(|c| c.cell_text.as_str()).collect();
        assert_eq!(texts, vec!["Revenue"]);
    }

    #[test]
    fn outputs_to_objects_keeps_confident_objects_in_pixels() {
        let (logits, boxes, _) = make_outputs(
            &[
                ("table", [100.0, 80.0, 500.0, 400.0], 10.0),
                ("no object", [0.0, 0.0, 10.0, 10.0], 10.0),
                ("table row", [100.0, 80.0, 500.0, 160.0], 0.0),
            ],
            &[],
        );
        // Third object: every class equally likely — nothing above 0.5.
        let mut logits = logits;
        for c in 0..CLASS_IDX2NAME.len() {
            logits[[0, 2, c]] = 0.0;
        }
        let objects = potatr_outputs_to_objects(&logits, &boxes, PAGE_W, PAGE_H);
        assert_eq!(
            objects.iter().map(|o| (o.idx, o.label.as_str())).collect::<Vec<_>>(),
            vec![(0, "table")]
        );
        assert!(objects[0].score > 0.99);
        assert!(approx(&objects[0].bbox, &[100.0, 80.0, 500.0, 400.0]));
    }

    #[test]
    fn table_structure_to_cells_2x2_with_header() {
        let (logits, boxes, _) = make_outputs(
            &[
                ("table", [100.0, 100.0, 500.0, 300.0], 10.0),
                ("table row", [100.0, 100.0, 500.0, 200.0], 10.0),
                ("table row", [100.0, 200.0, 500.0, 300.0], 10.0),
                ("table column", [100.0, 100.0, 300.0, 300.0], 10.0),
                ("table column", [300.0, 100.0, 500.0, 300.0], 10.0),
                ("table column header", [100.0, 100.0, 500.0, 200.0], 10.0),
            ],
            &[],
        );
        let tokens = vec![
            word("Name", [150.0, 140.0, 250.0, 160.0]),
            word("Value", [350.0, 140.0, 450.0, 160.0]),
            word("a", [150.0, 240.0, 170.0, 260.0]),
            word("1", [350.0, 240.0, 370.0, 260.0]),
        ];
        let tables = potatr_outputs_to_tables_cells(&logits, &boxes, &tokens, PAGE_W, PAGE_H);
        assert_eq!(tables.len(), 1);
        let got: std::collections::HashMap<(usize, usize), (&str, bool)> = tables[0]
            .iter()
            .map(|c| ((c.row_nums[0], c.column_nums[0]), (c.cell_text.as_str(), c.header)))
            .collect();
        assert_eq!(
            got,
            [
                ((0, 0), ("Name", true)),
                ((0, 1), ("Value", true)),
                ((1, 0), ("a", false)),
                ((1, 1), ("1", false)),
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn each_table_gets_its_own_caption() {
        // The second table is more confident; tables must not swap captions.
        let (logits, boxes, edge) = make_outputs(
            &[
                ("table", [100.0, 100.0, 400.0, 300.0], 4.0),
                ("table caption", [100.0, 60.0, 400.0, 90.0], 10.0),
                ("table", [500.0, 400.0, 900.0, 700.0], 10.0),
                ("table caption", [500.0, 360.0, 900.0, 390.0], 10.0),
            ],
            &[(0, 1), (2, 3)],
        );
        let tokens = vec![
            word("First", [150.0, 65.0, 250.0, 85.0]),
            word("Second", [550.0, 365.0, 650.0, 385.0]),
        ];
        let tables = tables_from_outputs(&logits, &boxes, &edge, &tokens, PAGE_W, PAGE_H, 0.5);
        assert_eq!(
            tables.iter().map(|t| t.caption.as_str()).collect::<Vec<_>>(),
            vec!["First", "Second"]
        );
        assert!(approx(&tables[0].bbox, &[100.0, 100.0, 400.0, 300.0]));
        assert!(tables[0].score < tables[1].score);
        assert!(tables[0].caption_parts[0].edge_score > 0.99);
    }
}
