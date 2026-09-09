// HybridConverter — pdf_oxide fast path + ONNX heavy passes.
//
// Matches Python `bobine/converter.py` line-for-line where possible.

use std::path::{Path, PathBuf};

use image::DynamicImage;
use pdf_oxide::{api::Pdf, geometry::Rect};
use tracing::{info, warn};

use crate::config::{ConverterConfig, RoutingMode};
use crate::engine::OnnxEngine;
use crate::error::{BobineError, Result};
use crate::pdf_source::{PdfSource, SourceChar};

/// Progress / cancellation hooks for long-running conversions.
///
/// - `should_continue`: return `false` to stop after the current page.
/// - `on_page`: called as `(page_index, page_count)` (0-based index).
pub struct ProgressHooks<'a> {
    pub should_continue: Box<dyn Fn() -> bool + 'a>,
    pub on_page: Box<dyn Fn(usize, usize) + 'a>,
}

impl Default for ProgressHooks<'_> {
    fn default() -> Self {
        Self {
            should_continue: Box::new(|| true),
            on_page: Box::new(|_, _| {}),
        }
    }
}

// ---------------------------------------------------------------------------
// Constants

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MATH_FONT_KEYWORDS: &[&str] = &[
    "cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol", "mathjax", "stix", "xits", "asana",
    "euclid",
];

const MATH_UNICODE_RANGES: &[(u32, u32)] = &[
    (0x0370, 0x03FF),
    (0x2200, 0x22FF),
    (0x2A00, 0x2AFF),
    (0x27C0, 0x27EF),
    (0x2980, 0x29FF),
    (0x1D400, 0x1D7FF),
];

const MONO_FONT_KEYWORDS: &[&str] = &[
    "mono",
    "courier",
    "consol",
    "menlo",
    "inconsolata",
    "sourcecode",
    "dejavu sans mono",
    "fixed",
    "terminal",
];

/// Region claiming priority for overlap dedup: lower claims first, so
/// specific routing wins the text under overlapping boxes. Must mirror the
/// routing order in `full_structure_page_markdown` (caption checked before
/// table/math, so `table_caption`/`formula_caption` route to plain text).
const CLAIM_MATH: u8 = 0;
const CLAIM_TABLE: u8 = 1;
const CLAIM_CAPTION: u8 = 2;
const CLAIM_TEXT: u8 = 3;
const CLAIM_NONE: u8 = 9; // figure/image: emits crops, never claims text

fn region_claim_priority(label: &str) -> u8 {
    let l = label.to_lowercase();
    if l.contains("caption") || l.contains("footnote") {
        CLAIM_CAPTION
    } else if l.contains("table") {
        CLAIM_TABLE
    } else if MATH_LAYOUT_LABELS.iter().any(|k| l.contains(k)) {
        CLAIM_MATH
    } else if l.contains("figure") || l.contains("image") {
        CLAIM_NONE
    } else {
        CLAIM_TEXT
    }
}

const MATH_LAYOUT_LABELS: &[&str] = &[
    "equation",
    "display_formula",
    "inline_formula",
    "isolate_formula",
    "formula",
];

// ---------------------------------------------------------------------------
// Glyph ownership (extracted from `full_structure_page_markdown`)
//
// Overlap dedup happens at GLYPH level, not rectangle level: every
// character of the page is assigned to exactly one owning region — the
// region with the largest overlap over that glyph, ties broken by claim
// priority (specific routing before generic: math > table > caption >
// text), then reading order. Figure regions never own glyphs: they emit
// crops, not text. See the longer rationale at the call site.
// ---------------------------------------------------------------------------

/// Claim order + rank lookup for a sorted region list.
/// Earlier claim_order position = more specific routing = wins ties.
fn claim_ranks(sorted: &[crate::rapid_layout::LayoutRegion]) -> (Vec<usize>, Vec<usize>) {
    let mut claim_order: Vec<usize> = (0..sorted.len()).collect();
    claim_order.sort_by(|&a, &b| {
        region_claim_priority(&sorted[a].label)
            .cmp(&region_claim_priority(&sorted[b].label))
            .then_with(|| {
                sorted[a]
                    .y0
                    .partial_cmp(&sorted[b].y0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        sorted[a]
                            .x0
                            .partial_cmp(&sorted[b].x0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
    });
    let mut rank = vec![0usize; sorted.len()];
    for (pos, &i) in claim_order.iter().enumerate() {
        rank[i] = pos;
    }
    (claim_order, rank)
}

/// Each char belongs to its argmax-overlap non-figure region; ties break
/// toward the earlier claim_order position (`rank`). Layout coords are
/// render pixels; char boxes are scaled up by `scale` (dpi / 72).
fn assign_glyph_owners(
    page_chars: &[SourceChar],
    sorted: &[crate::rapid_layout::LayoutRegion],
    rank: &[usize],
    scale: f32,
) -> Vec<Option<usize>> {
    let mut glyph_owner: Vec<Option<usize>> = vec![None; page_chars.len()];
    for (ci, c) in page_chars.iter().enumerate() {
        let mut best: Option<(f32, usize)> = None; // (overlap area px², region idx)
        for (ri, region) in sorted.iter().enumerate() {
            if region_claim_priority(&region.label) == CLAIM_NONE {
                continue;
            }
            let iw = region.x1.min((c.bbox.x + c.bbox.width) * scale)
                - region.x0.max(c.bbox.x * scale);
            if iw <= 0.0 {
                continue;
            }
            let ih = region.y1.min((c.bbox.y + c.bbox.height) * scale)
                - region.y0.max(c.bbox.y * scale);
            if ih <= 0.0 {
                continue;
            }
            let area = iw * ih;
            let take = match best {
                None => true,
                Some((ba, bri)) => {
                    area > ba || ((area - ba).abs() <= ba * 1e-6 && rank[ri] < rank[bri])
                }
            };
            if take {
                best = Some((area, ri));
            }
        }
        if let Some((_, ri)) = best {
            glyph_owner[ci] = Some(ri);
        }
    }
    glyph_owner
}

/// Caption/footnote pre-route (extracted from the dispatch loop).
/// Caption labels describe OTHER regions, so they always render as plain
/// text. Returns `Some(text)` when the region owns non-empty text (caller
/// pushes it and `continue`s); `None` lets dispatch fall through to the
/// label chain — this preserves the inline behavior, including the edge
/// case where an empty caption (e.g. `table_caption`) still reaches the
/// table arm below. The check must precede the math/table/figure routes
/// because e.g. "formula" substring-matches "formula_caption".
fn caption_block(page_chars: &[SourceChar], lines: &[Vec<usize>], lab: &str) -> Option<String> {
    if !(lab.contains("caption") || lab.contains("footnote")) {
        return None;
    }
    let text = region_lines_to_text(page_chars, lines);
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Figure/image arm (extracted from the dispatch loop). Returns the markdown
/// image link (`Some`) or `None` when neither an embedded raster nor a
/// render crop could be produced. Terminal: no fall-through — `None` simply
/// emits nothing for this region.
#[allow(clippy::too_many_arguments)]
fn figure_block(
    config: &ConverterConfig,
    sorted: &[crate::rapid_layout::LayoutRegion],
    region_lines: &[Vec<Vec<usize>>],
    page_chars: &[SourceChar],
    bbox: Rect,
    scale: f32,
    median_char_h: f32,
    assets: &[EmbeddedAsset],
    img: &DynamicImage,
    dpi: u32,
    work_dir: &Path,
    page_index: usize,
    block_no: usize,
) -> Option<String> {
    // Caption association: the nearest caption region (below or above
    // within ~3 line heights) becomes the alt text.
    let cap_alt = nearby_caption_text(
        sorted,
        region_lines,
        page_chars,
        &bbox,
        scale,
        3.0 * median_char_h,
    )
    .and_then(|t| caption_alt(&t, 120));
    let mk_img = |src: &str| match &cap_alt {
        Some(a) => format!("![{}]({})", a, src),
        None => format!("![]({})", src),
    };
    // Prefer the ORIGINAL embedded raster when this region covers it
    // (≥ 60% of the raster inside the box): lossless bytes at native
    // resolution beat a 300-dpi render crop that bakes in surrounding
    // content. Crops remain for vector/mixed art. Decoration-sized
    // assets are excluded from matching.
    let matched = best_embedded_match(&bbox, assets, 0.6)
        .and_then(|ai| assets.get(ai))
        .filter(|a| {
            a.bbox_pts.map_or(false, |b| {
                (b.width as f64) * (b.height as f64) >= config.min_figure_area_pts
            })
        });
    if let Some(a) = matched {
        info!(
            "page {}: figure → embedded asset {}",
            page_index + 1,
            a.rel_path
        );
        return Some(mk_img(&a.rel_path));
    }
    if let Some(crop) = crop_image(img, bbox, dpi, 0.0) {
        // Page-scoped structured name: every page restarts its block
        // counter and all pages share one asset tree.
        let p = work_dir
            .join(&config.image_output_dir)
            .join(format!("p{}", page_index))
            .join(format!("crop{}.png", block_no));
        std::fs::create_dir_all(p.parent().unwrap()).ok();
        if crop.save(&p).is_ok() {
            let rel = format!(
                "{}/p{}/{}",
                config.image_output_dir,
                page_index,
                p.file_name().unwrap().to_string_lossy()
            );
            return Some(mk_img(&rel));
        }
    }
    None
}

/// Seam repair: glyphs covered by NO emitting region (seams between
/// layout boxes, dropped by argmax ownership) are clustered into lines
/// and offered to the nearest region whose padded box contains them.
/// Only truly homeless lines are appended to `blocks` directly.
///
/// NOTE (behavior quirk, preserved): this runs AFTER the per-region
/// dispatch loop, so lines adopted into `region_lines` here are never
/// re-emitted into `blocks` — only homeless lines reach the output.
/// Moving this before dispatch would change output (goldens); do that
/// as a deliberate behavior fix, not a refactor.
fn repair_seams(
    page_chars: &[SourceChar],
    sorted: &[crate::rapid_layout::LayoutRegion],
    glyph_owner: &[Option<usize>],
    region_lines: &mut [Vec<Vec<usize>>],
    blocks: &mut Vec<String>,
    scale: f32,
    page_index: usize,
) {
    let unowned: Vec<usize> = (0..page_chars.len())
        .filter(|&i| glyph_owner[i].is_none())
        .collect();
    if unowned.is_empty() {
        return;
    }
    let pad_px = 8.0 * scale; // tolerance around each layout box (px)
    let mut homeless = 0usize;
    for line in cluster_lines(page_chars, &unowned) {
        let lx0 = line
            .iter()
            .map(|&i| page_chars[i].bbox.x)
            .fold(f32::INFINITY, f32::min);
        let ly0 = line
            .iter()
            .map(|&i| page_chars[i].bbox.y)
            .fold(f32::INFINITY, f32::min);
        let lx1 = line
            .iter()
            .map(|&i| page_chars[i].bbox.x + page_chars[i].bbox.width)
            .fold(f32::NEG_INFINITY, f32::max);
        let ly1 = line
            .iter()
            .map(|&i| page_chars[i].bbox.y + page_chars[i].bbox.height)
            .fold(f32::NEG_INFINITY, f32::max);
        let cx = (lx0 + lx1) * 0.5 * scale;
        let cy = (ly0 + ly1) * 0.5 * scale;
        // Regions whose PADDED box contains the line center; nearest
        // vertical edge wins (seams are mostly horizontal slivers).
        let mut best: Option<(f32, usize)> = None;
        for (ri, region) in sorted.iter().enumerate() {
            if region_claim_priority(&region.label) == CLAIM_NONE {
                continue;
            }
            if cx >= region.x0 - pad_px
                && cx <= region.x1 + pad_px
                && cy >= region.y0 - pad_px
                && cy <= region.y1 + pad_px
            {
                let d = if cy < region.y0 {
                    region.y0 - cy
                } else if cy > region.y1 {
                    cy - region.y1
                } else {
                    0.0
                };
                if best.map_or(true, |(bd, _)| d < bd) {
                    best = Some((d, ri));
                }
            }
        }
        match best {
            Some((_, ri)) => {
                region_lines[ri].push(line);
                // Re-sort this region's lines back into reading order.
                region_lines[ri].sort_by(|a, b| {
                    let ay = a
                        .iter()
                        .map(|&i| page_chars[i].bbox.y)
                        .fold(f32::INFINITY, f32::min);
                    let by = b
                        .iter()
                        .map(|&i| page_chars[i].bbox.y)
                        .fold(f32::INFINITY, f32::min);
                    ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            None => {
                homeless += line.len();
                let t = region_lines_to_text(page_chars, &[line]);
                if !t.trim().is_empty() {
                    blocks.push(t);
                }
            }
        }
    }
    if homeless > 0 {
        info!(
            "page {}: {} chars outside every layout region (appended)",
            page_index + 1,
            homeless
        );
    }
}
/// Assign WHOLE LINES, not glyphs: cluster all owned glyphs into visual
/// lines first, then give each line to its majority owner. Per-glyph
/// ownership alone can switch regions mid-line where two model boxes
/// disagree, shredding words across output blocks. Lines come back in
/// reading order within each region.
fn assign_region_lines(
    page_chars: &[SourceChar],
    glyph_owner: &[Option<usize>],
    rank: &[usize],
    num_regions: usize,
) -> Vec<Vec<Vec<usize>>> {
    let owned: Vec<usize> = (0..page_chars.len())
        .filter(|&i| glyph_owner[i].is_some())
        .collect();
    let mut region_lines: Vec<Vec<Vec<usize>>> = vec![Vec::new(); num_regions];
    for line in cluster_lines(page_chars, &owned) {
        let mut tally: std::collections::HashMap<usize, usize> = Default::default();
        for &i in &line {
            // `owned` only holds `Some` indices, but never panic on a
            // corrupt page — skip instead of shredding the line.
            let Some(owner) = glyph_owner[i] else { continue };
            *tally.entry(owner).or_default() += 1;
        }
        let winner = tally
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| rank[b.0].cmp(&rank[a.0])));
        if let Some((ri, _)) = winner {
            region_lines[ri].push(line);
        }
    }
    for rl in region_lines.iter_mut() {
        rl.sort_by(|a, b| {
            let ay = a
                .iter()
                .map(|&i| page_chars[i].bbox.y)
                .fold(f32::INFINITY, f32::min);
            let by = b
                .iter()
                .map(|&i| page_chars[i].bbox.y)
                .fold(f32::INFINITY, f32::min);
            ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    region_lines
}

// ---------------------------------------------------------------------------
// HybridConverter
// ---------------------------------------------------------------------------

pub struct HybridConverter {
    #[allow(dead_code)]
    pub config: ConverterConfig,
    pub engine: OnnxEngine,
}

impl HybridConverter {
    pub fn new(config: ConverterConfig, cache_dir: &Path) -> Self {
        let engine = OnnxEngine::new(&config, cache_dir);
        Self { config, engine }
    }

    // ==================================================================
    // Top-level dispatch
    // ==================================================================

    pub fn convert(&mut self, input: &Path, work_dir: &Path) -> Result<String> {
        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        match ext.as_str() {
            "pdf" => self.convert_pdf(input, work_dir),
            "docx" | "xlsx" | "pptx" | "doc" | "xls" | "ppt" => self.convert_office(input),
            _ => std::fs::read_to_string(input).map_err(BobineError::Io),
        }
    }

    // ==================================================================
    // PDF conversion
    // ==================================================================

    pub fn convert_pdf(&mut self, path: &Path, work_dir: &Path) -> Result<String> {
        self.convert_pdf_with(path, work_dir, &ProgressHooks::default())
    }

    /// Like [`convert_pdf`] but with progress/cancellation hooks.
    pub fn convert_pdf_with(
        &mut self,
        path: &Path,
        work_dir: &Path,
        hooks: &ProgressHooks<'_>,
    ) -> Result<String> {
        let mut pdf = Pdf::open(path).map_err(|e| BobineError::PdfOxide(format!("open: {e}")))?;
        self.convert_pdf_source(&mut pdf, work_dir, hooks)
    }

    /// Convert an already-opened document. Generic over any [`PdfSource`] —
    /// pass a real `pdf_oxide::api::Pdf`, or a custom source for testing /
    /// alternative PDF backends.
    pub fn convert_pdf_source<S: PdfSource>(
        &mut self,
        pdf: &mut S,
        work_dir: &Path,
        hooks: &ProgressHooks<'_>,
    ) -> Result<String> {
        let n_pages = pdf.page_count()?;

        std::fs::create_dir_all(work_dir)?;
        let mut blocks: Vec<String> = Vec::with_capacity(n_pages);

        for i in 0..n_pages {
            if !(hooks.should_continue)() {
                info!("conversion cancelled at page {} of {}", i + 1, n_pages);
                break;
            }
            (hooks.on_page)(i, n_pages);
            // Embedded-image inventory: built once per page so the fast-path
            // interleave, the figure branch and the gallery tail all see the
            // same placements. Paths come straight from the pre-pass (which
            // picks the extension), bboxes from the same extraction order.
            let assets = if self.config.extract_images {
                match self.extract_page_images(pdf, i, work_dir) {
                    Ok(paths) => {
                        let infos = pdf.images(i).unwrap_or_default();
                        paths
                            .into_iter()
                            .zip(infos)
                            .map(|(path, im)| EmbeddedAsset {
                                rel_path: format!(
                                    "{}/p{}/{}",
                                    self.config.image_output_dir,
                                    i,
                                    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                                ),
                                bbox_pts: im.bbox,
                            })
                            .collect()
                    }
                    Err(e) => {
                        warn!("image extraction failed on page {}: {e}", i + 1);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            };
            let page_md = self.route_page(pdf, i, work_dir, &assets)?;
            let final_md = if self.config.extract_images && self.config.append_unreferenced_images {
                maybe_append_gallery(page_md, &assets)
            } else {
                page_md
            };
            blocks.push(final_md);
        }

        let joined = blocks.join("\n\n---\n\n");
        Ok(if self.config.promote_title {
            promote_document_title(&joined)
        } else {
            joined
        })
    }

    // ==================================================================
    // Per-page routing
    // ==================================================================

    fn route_page(
        &mut self,
        pdf: &mut dyn PdfSource,
        index: usize,
        work_dir: &Path,
        assets: &[EmbeddedAsset],
    ) -> Result<String> {
        let md = self.route_page_inner(pdf, index, work_dir, assets)?;
        // Pattern-based section-heading promotion runs on every mode's output
        // (font heuristics miss numbered sections in all of them).
        let md = if self.config.promote_headings {
            promote_headings(&md)
        } else {
            md
        };
        // Code-block detection is a pure char scan — no ONNX needed.
        if self.config.detect_code_blocks {
            let mut md = md;
            for block in wrap_code_blocks(pdf, index) {
                md.push_str("\n\n");
                md.push_str(&block);
            }
            return Ok(md);
        }
        Ok(md)
    }

    fn route_page_inner(
        &mut self,
        pdf: &mut dyn PdfSource,
        index: usize,
        work_dir: &Path,
        assets: &[EmbeddedAsset],
    ) -> Result<String> {
        if !self.config.use_onnx || self.config.routing_mode == RoutingMode::Never {
            return self.fast_markdown_placed(pdf, index, assets);
        }

        if self.config.routing_mode == RoutingMode::Surgical {
            if is_scanned(pdf, index, self.config.scanned_text_threshold) {
                self.engine.ensure_models()?;
                match self.full_structure_page_markdown(pdf, index, work_dir, assets) {
                    Ok(Some(md)) if !md.trim().is_empty() => {
                        info!("page {}: scanned → ONNX layout+OCR", index + 1);
                        return Ok(md);
                    }
                    Err(e) => warn!(
                        "page {}: ONNX full-structure failed ({}); falling back to fast path",
                        index + 1,
                        e
                    ),
                    _ => {}
                }
                return self.fast_markdown_placed(pdf, index, assets);
            }
            let md = self.surgical_page_markdown(pdf, index, work_dir)?;
            return Ok(interleave_images(
                pdf,
                index,
                self.config.min_figure_area_pts,
                &md,
                assets,
            ));
        }

        // AUTO / ALWAYS
        if needs_onnx(pdf, index, &self.config) {
            match self.full_structure_page_markdown(pdf, index, work_dir, assets) {
                Ok(Some(md)) if !md.trim().is_empty() => {
                    info!("page {} → ONNX layout+OCR", index + 1);
                    return Ok(md);
                }
                Err(e) => tracing::warn!(
                    "page {}: ONNX full-structure failed ({}); falling back to fast path",
                    index + 1,
                    e
                ),
                _ => {}
            }
        }

        self.fast_markdown_placed(pdf, index, assets)
    }

    /// Fast-path markdown with embedded figures interleaved at their visual
    /// positions.
    fn fast_markdown_placed(
        &mut self,
        pdf: &mut dyn PdfSource,
        index: usize,
        assets: &[EmbeddedAsset],
    ) -> Result<String> {
        let md = fast_page_markdown(pdf, index)?;
        Ok(interleave_images(
            pdf,
            index,
            self.config.min_figure_area_pts,
            &md,
            assets,
        ))
    }

    // ==================================================================
    // SURGICAL pipeline
    // ==================================================================

    fn surgical_page_markdown(
        &mut self,
        pdf: &mut dyn PdfSource,
        index: usize,
        work_dir: &Path,
    ) -> Result<String> {
        let fast_md = fast_page_markdown(pdf, index)?;

        let mut boxes =
            math_boxes_from_chars(pdf, index, self.config.min_formula_math_chars, 1.5, 1.5);
        if boxes.is_empty() && self.config.formula_layout_fallback {
            // P2 fallback: use RapidLayout to find equations when text-layer
            // has no math fonts (Word/InDesign/OCR output).
            let dpi = self.config.formula_dpi;
            if let Ok(img) = render_page_image(pdf, index, dpi) {
                let scale = dpi as f32 / 72.0;
                if let Ok(regions) = self.engine.layout_regions(&img) {
                    boxes = regions
                        .into_iter()
                        .filter(|r| {
                            MATH_LAYOUT_LABELS
                                .iter()
                                .any(|k| r.label.to_lowercase().contains(k))
                        })
                        .map(|r| {
                            Rect::new(
                                r.x0 / scale,
                                r.y0 / scale,
                                (r.x1 - r.x0) / scale,
                                (r.y1 - r.y0) / scale,
                            )
                        })
                        .collect();
                }
            }
        }
        if boxes.is_empty() {
            return Ok(fast_md);
        }

        let dpi = self.config.formula_dpi;
        let img = render_page_image(pdf, index, dpi)?;
        let media = page_media_box(pdf, index)?;
        let page_h = media[3];

        let line_height = estimate_line_height(pdf, index, page_h);

        // Crop + OCR each box
        let mut crops: Vec<(Rect, PathBuf)> = Vec::new();
        for (j, &bbox) in boxes.iter().enumerate() {
            if let Some(crop) = crop_image(&img, bbox, dpi, self.config.formula_pad_pts) {
                if crop.width() >= 4 && crop.height() >= 4 {
                    let p = work_dir.join(format!("_formula_p{}_{}.png", index, j));
                    if crop.save(&p).is_ok() {
                        crops.push((bbox, p));
                    }
                }
            }
        }
        if crops.is_empty() {
            return Ok(fast_md);
        }

        info!(
            "page {}: {} formula region(s) → OCR",
            index + 1,
            crops.len()
        );

        let mut replacements: Vec<(String, String)> = Vec::new();
        for (bbox, crop_path) in &crops {
            if let Some(latex) = self.engine.recognize_formula(crop_path)? {
                let needle = region_text(pdf, index, *bbox);
                let display = (bbox.height) > 1.6 * line_height
                    || bbox.width > self.config.formula_inline_max_width_pts as f32;
                let wrapped = if display {
                    format!("$$\n{}\n$$", latex)
                } else {
                    format!("${}$", latex)
                };
                replacements.push((needle, wrapped));
            }
        }

        Ok(splice(&fast_md, &replacements))
    }

    // ==================================================================
    // Helpers
    // ==================================================================

    fn extract_page_images(
        &self,
        pdf: &mut dyn PdfSource,
        index: usize,
        dir: &Path,
    ) -> Result<Vec<std::path::PathBuf>> {
        // Structured layout: <work>/<image_output_dir>/p{page}/img{k}.{ext}
        let out_dir = dir
            .join(&self.config.image_output_dir)
            .join(format!("p{}", index));
        pdf.extract_image_files(index, &out_dir, "img")
    }

    pub fn convert_office(&self, path: &Path) -> Result<String> {
        use office_oxide::Document;
        let doc =
            Document::open(path).map_err(|e| BobineError::OfficeOxide(format!("open: {e}")))?;
        Ok(doc.to_markdown())
    }
}

// ======================================================================
// Free functions
// ======================================================================

/// Append a gallery of UNREFERENCED embedded images at the end of the page.
///
/// Unlike the previous all-or-nothing rule (any `![` suppressed the whole
/// gallery, losing figures that the layout model missed), each asset is now
/// judged individually: assets already referenced inline are skipped, and
/// decoration-sized placements (below `min_figure_area_pts`) are dropped.
/// Assets without a known bbox cannot be size-judged and stay candidates —
/// they may be content the source could not localize.
fn maybe_append_gallery(md: String, assets: &[EmbeddedAsset]) -> String {
    let mut gallery = String::new();
    for a in assets {
        if md.contains(&format!("({})", a.rel_path)) {
            continue; // already referenced inline by figure branch / interleave
        }
        if let Some(b) = a.bbox_pts {
            if f64::from(b.width) * f64::from(b.height) < 100.0 {
                continue; // decoration (logo/rule/bullet)
            }
        }
        gallery.push_str(&format!("![]({})\n", a.rel_path));
    }
    if gallery.is_empty() {
        md
    } else {
        format!("{}\n\n{}", md, gallery)
    }
}

// ======================================================================
// Fast-path image interleaving
// ======================================================================

/// Collapse runs of whitespace so md text and char-line text compare equal
/// despite reflow; returns the normalized string plus the original byte
/// offset of every surviving char.
fn normalize_with_offsets(s: &str) -> (String, Vec<usize>) {
    let mut norm = String::with_capacity(s.len());
    let mut offsets = Vec::with_capacity(s.len());
    let mut last_ws = true; // skip leading
    for (i, ch) in s.char_indices() {
        if ch.is_whitespace() {
            if !last_ws {
                norm.push(' ');
                offsets.push(i);
            }
            last_ws = true;
        } else {
            norm.push(ch);
            offsets.push(i);
            last_ws = false;
        }
    }
    while norm.ends_with(' ') {
        norm.pop();
        offsets.pop();
    }
    (norm, offsets)
}

/// Insert inline-image links into fast-path markdown at their visual
/// positions.
///
/// Anchoring works on the TEXT LAYER, not the markdown: chars are clustered
/// into lines with [`cluster_lines`], each asset is placed before the first
/// line that starts below its bbox bottom, and that line's text is located
/// in the markdown with a monotone substring search (pdf_oxide emits text in
/// reading order, so line order and markdown order agree). Lines whose text
/// can't be found — e.g. swallowed by paragraph reflow or table detection —
/// are skipped entirely — [`maybe_append_gallery`] is the single tail
/// fallback for them (and for unplaced assets).
///
/// Limitation: on multi-column pages an anchor may resolve into the other
/// column's same-height text. Acceptable v1 trade-off vs. losing placement
/// entirely.
/// If `text` looks like a figure/table caption ("Figure 3: …", "Fig. S1 …",
/// "Table 2 …"), return a collapsed, truncated single-line version suitable
/// for image alt text. Non-captions yield `None` — empty alt is better than
/// a wrong one.
fn caption_alt(text: &str, max_chars: usize) -> Option<String> {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut words = collapsed.split(' ');
    let first = words.next()?.trim_end_matches(['.', ':']).to_lowercase();
    if !matches!(first.as_str(), "figure" | "fig" | "table" | "abb" | "plate") {
        return None;
    }
    let rest = words.next()?;
    if rest.is_empty() {
        return None;
    }
    // The token after the keyword should identify the figure: a number,
    // a supplementary tag ("S1"), or nothing but punctuation.
    // (`rest` is non-empty per the check above; `map_or` avoids an unwrap
    // on untrusted caption text.)
    if !rest.chars().next().map_or(false, |c| c.is_ascii_digit())
        && !rest.chars().any(|c| c.is_ascii_alphanumeric())
    {
        return None;
    }
    let mut out = String::new();
    for w in collapsed.split(' ') {
        if out.len() + w.len() + 1 > max_chars {
            out.push('…');
            break;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    // Alt-text syntax: square brackets would terminate the ![...] run.
    Some(out.replace(['[', ']'], ""))
}

// ======================================================================
// Heading promotion (pattern-based scholarly outline)
// ======================================================================

/// Promote scholarly section headings that font heuristics leave as bold or
/// plain text ("I. INTRODUCTION", "**3.1** **Encoder** **and** **Decoder**
/// **Stacks**") to markdown headings. Applied to every mode's output —
/// pdf_oxide's `detect_headings` only catches display titles, and DocLayout's
/// single `title` class gives no hierarchy.
fn promote_headings(md: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in md.split('\n') {
        let t = line.trim();
        if t.starts_with("```") {
            in_fence = !in_fence;
            out.push(line.to_string());
            continue;
        }
        if !in_fence {
            if let Some(h) = promote_heading_line(t) {
                out.push(h);
                continue;
            }
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

/// Promote a single line if it matches a guarded section-heading pattern.
fn promote_heading_line(t: &str) -> Option<String> {
    // Reject list/quote/table/image lines; note "**" bold runs are fine,
    // only single-star bullets are excluded.
    if t.starts_with(['#', '|', '>', '-', '!'])
        || t.starts_with("* ")
        || t == "*"
    {
        return None;
    }
    let bare = t.replace("**", "");
    let bare = bare.trim();
    let (level, rest, list_risk) = split_section_prefix(bare)?;
    // Uppercase/bold headings pass outright; otherwise accept short,
    // title-like remainders ("Encoder and Decoder Stacks"). Single-number
    // prefixes ("1.") are excluded from that relaxation because they are
    // indistinguishable from markdown ordered-list items.
    let ok = plausible_heading_text(rest)
        || bold_dominant(t)
        || (!list_risk && short_title_like(rest));
    if !ok {
        return None;
    }
    Some(format!("{} {}", "#".repeat(level), rest))
}

/// Split a leading section-number/letter prefix from the heading text:
/// roman ("I." → H2), decimal ("3.1" → H3, "3.1.1" → H4), plain number
/// ("3." → H2, flagged as list-risky), capital letter ("A." → H3).
fn split_section_prefix(s: &str) -> Option<(usize, &str, bool)> {
    let b = s.as_bytes();
    if b.is_empty() {
        return None;
    }
    // Decimal / numbered sections.
    if b[0].is_ascii_digit() {
        let mut i = 0;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let mut parts = 1;
        loop {
            if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
                parts += 1;
                i += 1;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
            } else {
                break;
            }
        }
        if i < b.len() && b[i] == b'.' {
            i += 1;
        }
        if i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
            let rest = s[i..].trim();
            // Section numbering starts at 1 — "0 H" is figure-axis junk.
            let leading_ok = s[..i].trim_end_matches('.').parse::<u32>().map_or(true, |v| v > 0);
            if !rest.is_empty() && leading_ok {
                let level = match parts {
                    1 => 2,
                    2 => 3,
                    _ => 4,
                };
                // "1. item" is exactly ordered-list syntax — list-risky.
                return Some((level, rest, parts == 1));
            }
        }
        return None;
    }
    // Roman numerals, longest first.
    const ROMANS: [&str; 12] = [
        "XII", "XI", "IX", "VIII", "VII", "VI", "IV", "X", "V", "III", "II", "I",
    ];
    for r in ROMANS {
        if let Some(after_dot) = s.strip_prefix(r).and_then(|x| x.strip_prefix(". ")) {
            let rest = after_dot.trim();
            if !rest.is_empty() {
                return Some((2, rest, true));
            }
        }
    }
    // Single capital letter.
    let c = b[0] as char;
    if c.is_ascii_uppercase() {
        if let Some(after_dot) = s[1..].strip_prefix(". ") {
            let rest = after_dot.trim();
            if !rest.is_empty() {
                return Some((3, rest, false));
            }
        }
    }
    None
}

/// Short, title-shaped remainder: few words (≤8), starts capitalized, no
/// commas or trailing sentence punctuation.
fn short_title_like(rest: &str) -> bool {
    if rest.ends_with(['.', ',', ';', ':']) || rest.contains(',') || rest.contains('=') {
        return false;
    }
    // Numeric tokens ("0.00", "5000") smell of axis labels, not titles.
    if rest.split_whitespace().any(|w| w.parse::<f64>().is_ok()) {
        return false;
    }
    let starts_capital = match rest.split_whitespace().next() {
        Some(w) => w.chars().next().map_or(false, |c| c.is_uppercase()),
        None => false,
    };
    starts_capital && rest.split_whitespace().count() <= 8
}

/// Text guards against promoting prose: short, few words, no trailing
/// sentence punctuation, and either mostly uppercase or (checked separately)
/// predominantly bold-wrapped by the source markdown.
fn plausible_heading_text(rest: &str) -> bool {
    if rest.is_empty() || rest.len() > 100 {
        return false;
    }
    if rest.ends_with(['.', ',', ';', '!']) {
        return false;
    }
    if rest.split_whitespace().count() > 14 {
        return false;
    }
    // Axis labels and equation debris ("Time t = 0.00", "H") are not
    // headings.
    if rest.contains('=') {
        return false;
    }
    if rest
        .split_whitespace()
        .any(|w| w.parse::<f64>().is_ok())
    {
        return false;
    }
    let letters = rest.chars().filter(|c| c.is_ascii_alphabetic()).count();
    if letters < 3 {
        return false;
    }
    let upper = rest.chars().filter(|c| c.is_ascii_uppercase()).count();
    upper * 10 >= letters * 7
}

/// True when >60% of the line's non-whitespace chars sit inside `**…**`
/// runs — pdf_oxide wraps heading-font text in bold markers.
fn bold_dominant(line: &str) -> bool {
    let segs: Vec<&str> = line.split("**").collect();
    if segs.len() < 3 {
        return false;
    }
    let nsw = |x: &str| x.chars().filter(|c| !c.is_whitespace()).count();
    let total: usize = segs.iter().map(|x| nsw(x)).sum();
    let bold: usize = segs
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, x)| nsw(x))
        .sum();
    total > 0 && bold * 10 >= total * 6
}

/// Document-level title promotion: if the very first content block is a short
/// non-sentence line, make it the `#` title; otherwise promote the first
/// heading found near the top of the document to level 1.
fn promote_document_title(md: &str) -> String {
    let lines: Vec<&str> = md.split('\n').collect();
    let is_content = |l: &str| {
        let t = l.trim();
        !t.is_empty() && t != "---"
    };
    // Case 1: first content line is itself title-like and not yet a heading.
    if let Some(i) = lines.iter().position(|l| is_content(l)) {
        let t = lines[i].trim();
        if !t.starts_with('#') && title_like(t) && !t.ends_with(['.', ',', ';']) {
            let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            out[i] = format!("# {t}");
            return out.join("\n");
        }
    }
    // Case 2: first title-like heading on PAGE 1 (before the first page
    // separator) → level 1. Page scoping beats a line-count window: ONNX
    // reading order can scramble blocks, and the document title always
    // lives on the first page.
    let mut out: Option<Vec<String>> = None;
    for (i, l) in lines.iter().enumerate() {
        if l.trim() == "---" {
            break;
        }
        if !is_content(l) {
            continue;
        }
        let t = l.trim_start();
        let hashes = t.chars().take_while(|&c| c == '#').count();
        // Layout models sometimes label email/prose fragments "title" —
        // require actual title shape.
        if hashes >= 2 && title_like(t[hashes..].trim()) {
            let mut o: Vec<String> = lines.iter().map(|x| x.to_string()).collect();
            o[i] = format!("# {}", t[hashes..].trim());
            out = Some(o);
            break;
        }
    }
    if let Some(o) = out {
        return o.join("\n");
    }
    md.to_string()
}

/// Sanity gate for document-title promotion: rejects bullet/equation debris
/// ("∗ yiping.ma@… Here we focus on … o = O …") while accepting real titles.
fn title_like(t: &str) -> bool {
    if t.starts_with(['*', '•', '\u{2217}', '\u{2212}', '-']) {
        return false;
    }
    if t.contains('=') || t.contains('+') || t.contains(")(") || t.contains('@') {
        return false;
    }
    // Author lists (≥2 commas) and dates ("August 7, 2026" — bare numeric
    // tokens) are not titles.
    if t.matches(',').count() >= 2 {
        return false;
    }
    if t.split_whitespace().any(|w| w.trim_end_matches(',').parse::<f64>().is_ok()) {
        return false;
    }
    t.len() <= 120 && t.split_whitespace().count() <= 25
}

/// Text of the caption region nearest to figure rect `fig` (both in PDF
/// points): below or above within `max_gap_pts`, with some horizontal
/// overlap. Deterministic counterpart of the interleave path's caption
/// heuristic — DocLayout emits separate `figure_caption` regions.
fn nearby_caption_text(
    sorted: &[crate::rapid_layout::LayoutRegion],
    region_lines: &[Vec<Vec<usize>>],
    chars: &[SourceChar],
    fig: &Rect,
    scale: f32,
    max_gap_pts: f32,
) -> Option<String> {
    let mut best: Option<(f32, usize)> = None;
    for (j, r) in sorted.iter().enumerate() {
        if !r.label.to_lowercase().contains("caption") {
            continue;
        }
        let cr = Rect::new(
            r.x0 / scale,
            r.y0 / scale,
            (r.x1 - r.x0) / scale,
            (r.y1 - r.y0) / scale,
        );
        // Horizontal overlap required (≥ 30% of the narrower box).
        let ix0 = fig.x.max(cr.x);
        let ix1 = (fig.x + fig.width).min(cr.x + cr.width);
        let min_w = fig.width.min(cr.width);
        if ix1 - ix0 < 0.3 * min_w {
            continue;
        }
        let gap = if cr.y >= fig.y + fig.height {
            cr.y - (fig.y + fig.height) // caption below figure
        } else if fig.y >= cr.y + cr.height {
            fig.y - (cr.y + cr.height) // caption above figure
        } else {
            // Vertically overlapping: accept a SIDE-BY-SIDE caption (wide
            // two-column figures sometimes carry the caption beside them),
            // requiring substantial vertical co-extent and small horizontal
            // separation so we don't steal another figure's caption.
            let hgap = if cr.x >= fig.x + fig.width {
                cr.x - (fig.x + fig.width)
            } else if fig.x >= cr.x + cr.width {
                fig.x - (cr.x + cr.width)
            } else {
                0.0 // horizontally inside the figure box — caption on the image
            };
            let vov0 = fig.y.max(cr.y);
            let vov1 = (fig.y + fig.height).min(cr.y + cr.height);
            let voverlap = (vov1 - vov0).max(0.0);
            if hgap <= max_gap_pts * 2.0 && voverlap >= 0.5 * fig.height.min(cr.height) {
                hgap
            } else {
                continue;
            }
        };
        if gap <= max_gap_pts && best.map_or(true, |(bg, _)| gap < bg) {
            best = Some((gap, j));
        }
    }
    best.and_then(|(_, j)| {
        let t = region_lines_to_text(chars, &region_lines[j]);
        if t.trim().is_empty() { None } else { Some(t) }
    })
}

fn interleave_images(
    pdf: &mut dyn PdfSource,
    index: usize,
    min_area_pts: f64,
    md: &str,
    assets: &[EmbeddedAsset],
) -> String {
    // Candidates: placed, content-sized, not already referenced.
    let mut candidates: Vec<&EmbeddedAsset> = assets
        .iter()
        .filter(|a| {
            a.bbox_pts.is_some()
                && a.area_pts() >= min_area_pts
                && !md.contains(&format!("({})", a.rel_path))
        })
        .collect();
    if candidates.is_empty() {
        return md.to_string();
    }
    candidates.sort_by(|a, b| {
        let ay = a.bbox_pts.map_or(0.0, |b2| b2.y);
        let by = b.bbox_pts.map_or(0.0, |b2| b2.y);
        ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Text-layer lines in reading order (y ascending).
    let Ok(chars) = pdf.chars(index) else { return md.to_string() };
    if chars.is_empty() {
        return md.to_string();
    }
    let all: Vec<usize> = (0..chars.len()).collect();
    let mut lines: Vec<(f32, f32, String)> = cluster_lines(&chars, &all)
        .into_iter()
        .map(|l| {
            let y0 = l.iter().map(|&i| chars[i].bbox.y).fold(f32::INFINITY, f32::min);
            let y1 = l
                .iter()
                .map(|&i| chars[i].bbox.y + chars[i].bbox.height)
                .fold(f32::NEG_INFINITY, f32::max);
            (y0, y1, line_string(&chars, &l))
        })
        .collect();
    lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let (norm, offsets) = normalize_with_offsets(md);
    let mut insertions: Vec<(usize, String)> = Vec::new(); // (orig byte pos, link)
    let mut cursor = 0usize; // index into `norm`
    let mut line_i = 0usize;

    for asset in &candidates {
        // Candidates are pre-filtered by `bbox_pts.is_some()`, but re-assert
        // rather than unwrap: a future filter change must not panic on
        // unplaced assets (gallery picks them up instead).
        let Some(b) = asset.bbox_pts else { continue };
        let bottom = b.y + b.height;
        // Index of the first line starting below the asset's bottom edge.
        let anchor = (line_i..lines.len()).find(|&li| lines[li].0 >= bottom - 0.5);
        let Some(li) = anchor else {
            continue; // below all text lines — gallery will pick it up
        };
        line_i = li + 1;
        // Captions sit under (sometimes over) the figure, and rasters often
        // include whitespace padding, so scan a small window of lines that
        // STRADDLES the bbox bottom and take the caption-pattern line whose
        // edge is closest to it.
        let lo = li.saturating_sub(4);
        let hi = (li + 2).min(lines.len());
        let mut alt: Option<String> = None;
        let mut best_dist = f32::INFINITY;
        for j in lo..hi {
            let Some(c) = caption_alt(&lines[j].2, 120) else { continue };
            let dist = if lines[j].0 >= bottom {
                lines[j].0 - bottom
            } else {
                bottom - lines[j].1
            };
            if dist < best_dist {
                best_dist = dist;
                alt = Some(c);
            }
        }
        let link = match alt {
            Some(a) => format!("\n\n![{}]({})\n\n", a, asset.rel_path),
            None => format!("\n\n![]({})\n\n", asset.rel_path),
        };
        // Locate that line's text in the markdown after the previous anchor.
        let needle_full = normalize_with_offsets(&lines[li].2).0;
        let words: Vec<&str> = needle_full.split(' ').collect();
        let try8 = words.iter().take(8).copied().collect::<Vec<&str>>().join(" ");
        let try4 = words.iter().take(4).copied().collect::<Vec<&str>>().join(" ");
        let tries = [needle_full.as_str(), try8.as_str(), try4.as_str()];
        let mut hit: Option<(usize, usize)> = None; // (norm pos, matched len)
        for needle in tries.iter() {
            if needle.is_empty() {
                continue;
            }
            if let Some(pos) = norm[cursor..].find(needle) {
                hit = Some((cursor + pos, needle.len()));
                break;
            }
        }
        match hit {
            Some((pos, matched_len)) => {
                // Original byte offset of the match start, snapped to the
                // beginning of its markdown line.
                let orig = offsets.get(pos).copied().unwrap_or(0);
                let line_start = md[..orig].rfind('\n').map_or(0, |p| p + 1);
                insertions.push((line_start, link));
                cursor = pos + matched_len;
            }
            None => {} // unmatchable line — gallery fallback handles it
        }
    }

    if insertions.is_empty() {
        return md.to_string();
    }
    let mut out = String::with_capacity(md.len() + 128 * insertions.len());
    let mut last = 0usize;
    for (pos, link) in insertions {
        out.push_str(&md[last..pos]);
        out.push_str(&link);
        last = pos;
    }
    out.push_str(&md[last..]);
    out
}

fn fast_page_markdown(pdf: &mut dyn PdfSource, index: usize) -> Result<String> {
    match pdf.page_markdown(index) {
        Ok(md) => Ok(md),
        Err(_) => {
            let r = Rect::new(0.0, 0.0, f32::MAX, f32::MAX);
            Ok(pdf.text_in_rect(index, r).unwrap_or_default())
        }
    }
}

fn render_page_image(pdf: &mut dyn PdfSource, index: usize, dpi: u32) -> Result<DynamicImage> {
    let png = pdf.render_png(index, dpi)?;
    image::load_from_memory(&png).map_err(|e| BobineError::Ort(format!("decode render: {e}")))
}

fn page_media_box(pdf: &mut dyn PdfSource, index: usize) -> Result<[f32; 4]> {
    pdf.media_box(index)
}

fn estimate_line_height(pdf: &mut dyn PdfSource, index: usize, page_h: f32) -> f32 {
    if let Ok(chars) = pdf.chars(index) {
        let mut heights: Vec<f32> = chars
            .iter()
            .map(|c| c.bbox.height)
            .filter(|&h| h > 0.0)
            .collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !heights.is_empty() {
            return heights[heights.len() / 2];
        }
    }
    page_h / 50.0
}

// ======================================================================
// Math / scan detection
// ======================================================================

fn is_math_unicode(ch: char) -> bool {
    let cp = ch as u32;
    MATH_UNICODE_RANGES
        .iter()
        .any(|&(lo, hi)| cp >= lo && cp <= hi)
}

/// Heuristic quality gate for TexTeller output on hybrid-refined crops.
/// Fragment/prose crops produce recognizable garbage: English words,
/// mbox/text wrappers, or single symbol-free tokens. Real display
/// equations carry structure (operators, sub/superscripts, fractions).
fn plausible_display_latex(latex: &str) -> bool {
    let l = latex.to_lowercase();
    if l.contains("\\mbox{") || l.contains("\\text{") {
        return false;
    }
    if l.contains(" the ") || l.contains(" with ") || l.contains(" and ") {
        return false;
    }
    // Degenerate repetition (e.g. \mathscr{M} repeated 13x) is a classic
    // autoregressive hallucination on garbage crops.
    let words: std::collections::HashMap<&str, usize> =
        l.split_whitespace().fold(Default::default(), |mut m, w| {
            *m.entry(w).or_default() += 1;
            m
        });
    if words.values().any(|&c| c > 4) {
        return false;
    }
    ["=", "+", "^", "_", "\\frac", "\\sum", "\\int", "\\sqrt"]
        .iter()
        .any(|m| l.contains(m))
}

fn is_math_font(font_name: &str) -> bool {
    let lower = font_name.to_lowercase();
    MATH_FONT_KEYWORDS.iter().any(|k| lower.contains(k))
}

fn page_math_signal(pdf: &mut dyn PdfSource, index: usize) -> (usize, usize) {
    let chars = match pdf.chars(index) {
        Ok(c) => c,
        Err(_) => return (0, 0),
    };
    let total = chars.len();
    let math = chars
        .iter()
        .filter(|c| is_math_font(&c.font_name) || is_math_unicode(c.char))
        .count();
    (math, total)
}

fn is_scanned(pdf: &mut dyn PdfSource, index: usize, threshold: usize) -> bool {
    let chars = match pdf.chars(index) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let has_images = pdf.image_count(index).unwrap_or(0) > 0;
    chars.len() < threshold && has_images
}

fn needs_onnx(pdf: &mut dyn PdfSource, index: usize, config: &ConverterConfig) -> bool {
    if !config.use_onnx || config.routing_mode == RoutingMode::Never {
        return false;
    }
    if config.routing_mode == RoutingMode::Always {
        return true;
    }
    let (math, total) = page_math_signal(pdf, index);
    if math > config.math_char_threshold && (total == 0 || math as f64 / total as f64 > 0.02) {
        return true;
    }
    is_scanned(pdf, index, config.scanned_text_threshold)
}

// ======================================================================
// Formula box detection
// ======================================================================

#[allow(clippy::too_many_arguments)]
fn math_boxes_from_chars(
    pdf: &mut dyn PdfSource,
    index: usize,
    min_chars: usize,
    hgap_mult: f32,
    vgap_mult: f32,
) -> Vec<Rect> {
    let chars = match pdf.chars(index) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let items: Vec<&SourceChar> = chars
        .iter()
        .filter(|c| is_math_font(&c.font_name) || is_math_unicode(c.char))
        .collect();

    if items.is_empty() {
        return vec![];
    }

    let mut heights: Vec<f32> = items.iter().map(|c| c.bbox.height).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_h = heights[heights.len() / 2].max(10.0);
    let line_tol = 0.8 * med_h;
    let hgap = hgap_mult * med_h;
    let vgap_max = vgap_mult * med_h;

    // Group by baseline
    let mut sorted: Vec<&&SourceChar> = items.iter().collect();
    sorted.sort_by(|a, b| {
        (a.bbox.y + a.bbox.height / 2.0)
            .partial_cmp(&(b.bbox.y + b.bbox.height / 2.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines: Vec<Vec<Rect>> = vec![];
    for c in sorted {
        let r = c.bbox;
        let cy = r.y + r.height / 2.0;
        if let Some(last) = lines.last() {
            let last_cy = last[0].y + last[0].height / 2.0;
            if (cy - last_cy).abs() <= line_tol {
                lines.last_mut().unwrap().push(r);
                continue;
            }
        }
        lines.push(vec![r]);
    }

    // Horizontal merge per line
    let mut boxes: Vec<(Rect, usize)> = vec![];
    for ln in &lines {
        let mut sorted_ln = ln.clone();
        sorted_ln.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        let mut runs: Vec<(Rect, usize)> = vec![];
        for r in &sorted_ln {
            let mut merged = false;
            for (run_r, count) in &mut runs {
                if r.x <= run_r.x + run_r.width + hgap && r.x + r.width >= run_r.x - hgap {
                    let new_x = run_r.x.min(r.x);
                    let new_y = run_r.y.min(r.y);
                    let new_x1 = (run_r.x + run_r.width).max(r.x + r.width);
                    let new_y1 = (run_r.y + run_r.height).max(r.y + r.height);
                    *run_r = Rect::new(new_x, new_y, new_x1 - new_x, new_y1 - new_y);
                    *count += 1;
                    merged = true;
                    break;
                }
            }
            if !merged {
                runs.push((*r, 1));
            }
        }
        boxes.extend(runs);
    }

    // Vertical merge
    boxes.sort_by(|(a, _), (b, _)| {
        a.y.partial_cmp(&b.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut changed = true;
    while changed {
        changed = false;
        let mut out: Vec<(Rect, usize)> = vec![];
        for (b, n) in &boxes {
            let mut merged = false;
            for (u, count) in &mut out {
                let hx = !(b.x > u.x + u.width + hgap || b.x + b.width < u.x - hgap);
                if !hx {
                    continue;
                }
                let gap = (b.y - (u.y + u.height))
                    .max(u.y - (b.y + b.height))
                    .max(0.0);
                if gap > vgap_max {
                    continue;
                }
                let new_x = u.x.min(b.x);
                let new_y = u.y.min(b.y);
                let new_x1 = (u.x + u.width).max(b.x + b.width);
                let new_y1 = (u.y + u.height).max(b.y + b.height);
                *u = Rect::new(new_x, new_y, new_x1 - new_x, new_y1 - new_y);
                *count += n;
                merged = true;
                changed = true;
                break;
            }
            if !merged {
                out.push((*b, *n));
            }
        }
        boxes = out;
    }

    boxes
        .into_iter()
        .filter(|(_, n)| *n >= min_chars)
        .map(|(r, _)| r)
        .collect()
}

// ======================================================================
// Image cropping
// ======================================================================

/// An embedded raster asset extracted for a page, with its placement.
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedAsset {
    /// Work-dir-relative link target with forward slashes
    /// (`assets/p{n}/img{k}.{ext}`) — safe to embed in markdown on any OS.
    pub rel_path: String,
    /// Placement bbox in PDF points (None when not localizable).
    pub bbox_pts: Option<Rect>,
}

impl EmbeddedAsset {
    fn area_pts(&self) -> f64 {
        self.bbox_pts
            .map_or(0.0, |b| f64::from(b.width) * f64::from(b.height))
    }
}

fn rect_iou(a: &Rect, b: &Rect) -> f32 {
    let ix0 = a.x.max(b.x);
    let iy0 = a.y.max(b.y);
    let ix1 = (a.x + a.width).min(b.x + b.width);
    let iy1 = (a.y + a.height).min(b.y + b.height);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    if inter <= 0.0 {
        return 0.0;
    }
    let union = a.width * a.height + b.width * b.height - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// The embedded raster best covered by `region`, if any.
///
/// Uses COVERAGE (intersection / raster area) rather than symmetric IoU:
/// DocLayout figure boxes are routinely coarser or offset from the true
/// bitmap placement, so we only ask "does the figure region subsume this
/// raster?" before preferring its original bytes over a render crop.
/// Decoration-sized assets are filtered by the caller via area thresholds.
fn best_embedded_match(
    region: &Rect,
    assets: &[EmbeddedAsset],
    min_coverage: f32,
) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, a) in assets.iter().enumerate() {
        let Some(b) = a.bbox_pts else { continue };
        let ix0 = region.x.max(b.x);
        let iy0 = region.y.max(b.y);
        let ix1 = (region.x + region.width).min(b.x + b.width);
        let iy1 = (region.y + region.height).min(b.y + b.height);
        let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
        let area = b.width * b.height;
        if area <= 0.0 {
            continue;
        }
        let cov = inter / area;
        if cov >= min_coverage && best.map_or(true, |(bv, _)| cov > bv) {
            best = Some((cov, i));
        }
    }
    best.map(|(_, i)| i)
}

fn is_figure_label(label: &str) -> bool {
    let l = label.to_lowercase();
    l.contains("figure") || l.contains("image")
}

/// Drop figure regions that heavily overlap an earlier figure region — the
/// layout model sometimes draws two boxes over one figure, and each box
/// would emit a near-identical render crop. Text/table/caption regions are
/// never dropped here: their duplication is handled by glyph ownership.
fn dedup_figure_regions(regions: Vec<crate::rapid_layout::LayoutRegion>) -> Vec<crate::rapid_layout::LayoutRegion> {
    let mut kept: Vec<crate::rapid_layout::LayoutRegion> = Vec::with_capacity(regions.len());
    for r in regions {
        let dup = is_figure_label(&r.label)
            && kept.iter().any(|k| {
                is_figure_label(&k.label)
                    && rect_iou(
                        &Rect::new(k.x0, k.y0, k.x1 - k.x0, k.y1 - k.y0),
                        &Rect::new(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0),
                    ) >= 0.5
            });
        if !dup {
            kept.push(r);
        }
    }
    kept
}

fn crop_image(img: &DynamicImage, bbox: Rect, dpi: u32, pad_pts: f64) -> Option<DynamicImage> {
    let s = dpi as f32 / 72.0;
    let pad = pad_pts as f32;
    let left = ((bbox.x - pad) * s).max(0.0) as u32;
    let right = ((bbox.x + bbox.width + pad) * s) as u32;
    let top = ((bbox.y - pad) * s).max(0.0) as u32;
    let bottom = ((bbox.y + bbox.height + pad) * s) as u32;
    if right <= left || bottom <= top {
        return None;
    }
    Some(img.crop_imm(left, top, right - left, bottom - top))
}

// ======================================================================
// Region text
// ======================================================================

fn region_text(pdf: &mut dyn PdfSource, index: usize, bbox: Rect) -> String {
    pdf.text_in_rect(index, bbox).unwrap_or_default()
}

/// Assemble text from a region's OWNED glyphs (indices into `chars`).
///
/// Glyph ownership is resolved once per page (see the dedup block in
/// `full_structure_page_markdown`): each character belongs to exactly one
/// region, so lines can be emitted whole without any duplicate or shredded
/// fragments — which per-piece `text_in_rect` extraction could not
/// guarantee (pdf_oxide filters at *span* granularity under `Intersects`,
/// so words straddling a boundary were emitted by both sides).
fn line_string(chars: &[SourceChar], idxs: &[usize]) -> String {
    let mut idxs = idxs.to_vec();
    idxs.sort_by(|&a, &b| {
        chars[a].bbox.x.partial_cmp(&chars[b].bbox.x).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut text = String::new();
    let mut prev_end = f32::NAN;
    for &i in &idxs {
        let g = &chars[i].bbox;
        // Insert a space when the gap between glyphs is wide relative to the
        // glyph's own advance (~0.25x) — cheap word segmentation without a
        // full layout pass.
        if !prev_end.is_nan() && g.x - prev_end > g.width * 0.25 {
            text.push(' ');
        }
        text.push(chars[i].char);
        prev_end = g.x + g.width;
    }
    text.trim().to_string()
}

/// Cluster glyphs into visual lines by centre-y. Tolerance is the median
/// advance (heights are noisy due to ascenders/descenders).
fn cluster_lines(chars: &[SourceChar], glyph_ids: &[usize]) -> Vec<Vec<usize>> {
    if glyph_ids.is_empty() {
        return vec![];
    }
    let gyc = |i: usize| chars[i].bbox.y + chars[i].bbox.height / 2.0;
    let mut ws: Vec<f32> = glyph_ids.iter().map(|&i| chars[i].bbox.width).collect();
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let tol = ws[ws.len() / 2].max(0.5);

    let mut order = glyph_ids.to_vec();
    order.sort_by(|&a, &b| {
        gyc(a).partial_cmp(&gyc(b)).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                chars[a].bbox.x.partial_cmp(&chars[b].bbox.x).unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut lines: Vec<Vec<usize>> = vec![vec![]];
    let mut line_y = gyc(order[0]);
    for &i in &order {
        if (gyc(i) - line_y).abs() > tol {
            lines.push(vec![]);
            line_y = gyc(i);
        }
        // Split at wide horizontal gaps (column gutters): same-y glyphs from
        // adjacent text columns must not join one line, or the assembler
        // would interleave the columns.
        if let Some(&last) = lines.last().unwrap().last() {
            let gap = chars[i].bbox.x - (chars[last].bbox.x + chars[last].bbox.width);
            if gap > (chars[i].bbox.width * 2.0).max(12.0) {
                lines.push(vec![]);
            }
        }
        lines.last_mut().unwrap().push(i);
    }
    lines
}

/// Assemble text from a region's OWNED lines (each pre-clustered, glyph
/// indices into `chars`). Lines are emitted top-to-bottom.
///
/// Ownership is resolved once per page at LINE granularity (see the dedup
/// block in `full_structure_page_markdown`): each visual line belongs to
/// exactly one region, so lines are emitted whole — never duplicated and
/// never shredded mid-word, which per-piece `text_in_rect` extraction could
/// not guarantee (pdf_oxide filters at *span* granularity under
/// `Intersects`, so words straddling a strip boundary were emitted by both
/// sides).
fn region_lines_to_text(chars: &[SourceChar], lines: &[Vec<usize>]) -> String {
    let mut out = String::new();
    for line in lines {
        let t = line_string(chars, line);
        if !t.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&t);
        }
    }
    out
}
/// `base` minus the union of `subs`, as a set of disjoint rectangles
/// (standard strip decomposition). Degenerate pieces are dropped; if the
/// piece count runs away (pathological overlap) the whole base is kept.
///
/// No longer used by the region-dedup path (glyph ownership replaced rect
/// subtraction — see `full_structure_page_markdown`); kept for the geometry
/// unit tests and any future per-rectangle work.
#[cfg(test)]
fn rect_subtract(base: Rect, subs: &[Rect]) -> Vec<Rect> {
    const EPS: f32 = 1e-3;
    let mut pieces: Vec<Rect> = vec![base];
    for s in subs {
        if s.width <= 0.0 || s.height <= 0.0 {
            continue;
        }
        let mut next: Vec<Rect> = Vec::with_capacity(pieces.len() + 4);
        for p in pieces.drain(..) {
            if s.x >= p.x + p.width - EPS
                || s.x + s.width <= p.x + EPS
                || s.y >= p.y + p.height - EPS
                || s.y + s.height <= p.y + EPS
            {
                next.push(p);
                continue;
            }
            // Top strip: full width above `s`
            if s.y > p.y + EPS {
                next.push(Rect::new(p.x, p.y, p.width, s.y - p.y));
            }
            // Bottom strip: full width below `s`
            if s.y + s.height < p.y + p.height - EPS {
                next.push(Rect::new(
                    p.x,
                    s.y + s.height,
                    p.width,
                    p.y + p.height - s.y - s.height,
                ));
            }
            // Left/right strips only within `s`'s y-span (no corner overlap)
            let sy0 = s.y.max(p.y);
            let sy1 = (s.y + s.height).min(p.y + p.height);
            if s.x > p.x + EPS {
                next.push(Rect::new(p.x, sy0, s.x - p.x, sy1 - sy0));
            }
            if s.x + s.width < p.x + p.width - EPS {
                let x0 = (s.x + s.width).max(p.x);
                next.push(Rect::new(x0, sy0, p.x + p.width - x0, sy1 - sy0));
            }
        }
        pieces = next
            .into_iter()
            .filter(|r| r.width > 0.5 && r.height > 0.5)
            .collect();
        if pieces.len() > 64 {
            return vec![base];
        }
    }
    pieces
}

// ======================================================================
// Text splicing
// ======================================================================

fn ws_replace(md: &str, needle: &str, block: &str) -> Option<String> {
    let tokens: Vec<&str> = needle
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return None;
    }
    let escaped: Vec<String> = tokens.iter().map(|t| regex::escape(t)).collect();
    let pattern = escaped.join(r"\s+");
    let re = regex::Regex::new(&pattern).ok()?;
    re.find(md).map(|m| {
        let mut s = String::with_capacity(md.len());
        s.push_str(&md[..m.start()]);
        s.push_str(block);
        s.push_str(&md[m.end()..]);
        s
    })
}

fn splice(md: &str, replacements: &[(String, String)]) -> String {
    let mut result = md.to_string();
    let mut leftovers: Vec<&str> = vec![];
    for (needle, block) in replacements {
        let needle = needle.trim();
        if !needle.is_empty() && result.contains(needle) {
            result = result.replacen(needle, block, 1);
            continue;
        }
        if !needle.is_empty() {
            if let Some(new) = ws_replace(&result, needle, block) {
                result = new;
                continue;
            }
        }
        leftovers.push(block);
    }
    if !leftovers.is_empty() {
        result.push_str("\n\n");
        result.push_str(&leftovers.join("\n\n"));
    }
    result
}

// ======================================================================
// Full ONNX structure page (AUTO / ALWAYS modes)
// ======================================================================

impl HybridConverter {
    fn full_structure_page_markdown(
        &mut self,
        pdf: &mut dyn PdfSource,
        index: usize,
        work_dir: &Path,
        assets: &[EmbeddedAsset],
    ) -> Result<Option<String>> {
        let dpi = self.config.render_dpi;
        let img = match render_page_image(pdf, index, dpi) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        let media = page_media_box(pdf, index)?;
        let page_h = media[3];
        let scale = dpi as f32 / 72.0;

        // Text-layer math boxes (hybrid formula refinement). On born-digital
        // pages these localize display equations far more precisely than the
        // layout model, whose formula head only draws paragraph-sized blobs
        // on dense math pages. Boxes are consumed by the first region that
        // contains them so overlapping regions cannot emit duplicates.
        // hgap 2.5: TeX \quad spacing inside display equations fragments runs
        // at 1.5; vgap 0.9: never merge math from adjacent text lines.
        let mut page_math_boxes =
            math_boxes_from_chars(pdf, index, self.config.min_formula_math_chars, 2.5, 0.9);
        let line_height = estimate_line_height(pdf, index, page_h);

        // Layout analysis
        let regions = self.engine.layout_regions(&img)?;
        if regions.is_empty() {
            return Ok(None);
        }

        // Sort by reading order: top-to-bottom, left-to-right
        let mut sorted = regions.clone();
        sorted.sort_by(|a, b| {
            a.y0.partial_cmp(&b.y0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal))
        });
        // The layout model sometimes draws two heavily-overlapping boxes over
        // one figure; each box used to emit its own near-identical render crop.
        sorted = dedup_figure_regions(sorted);

        // Overlap dedup: the layout model emits heavily overlapping regions
        // (giant pseudo-formula boxes over text columns), and every text-route
        // region used to emit its whole rectangle — on a dense math paper each
        // character was emitted up to 4 times.
        //
        // Dedup happens at GLYPH level, not rectangle level: every character
        // of the page is assigned to exactly one owning region — the region
        // with the largest overlap over that glyph, ties broken by claim
        // priority (specific routing before generic: math > table > caption >
        // text), then reading order. Figure regions never own glyphs: they
        // emit crops, not text.
        //
        // Earlier revisions subtracted claimed rectangles from later regions
        // (strip decomposition) and extracted text per piece. That fragments
        // columns into slivers whose line-level extraction either duplicated
        // lines (span-granularity `Intersects` filtering in pdf_oxide bleeds
        // across shared strip edges) or shredded them mid-word (glyph-centre
        // rules on thin strips). Owning glyphs directly keeps every emitted
        // line intact while guaranteeing each glyph is emitted once.
        // Glyph ownership is resolved once per page (see `claim_ranks` /
        // `assign_glyph_owners` / `assign_region_lines` above for the
        // rationale). `owners` was removed: nothing ever read it.
        let (_claim_order, rank) = claim_ranks(&sorted);
        // One chars() pass for all glyph-exact region extraction below.
        let page_chars = pdf.chars(index).unwrap_or_default();
        let glyph_owner = assign_glyph_owners(&page_chars, &sorted, &rank, scale);
        let mut region_lines =
            assign_region_lines(&page_chars, &glyph_owner, &rank, sorted.len());

        // Median glyph height ≈ line height — used as the caption-association
        // distance budget.
        let mut char_heights: Vec<f32> =
            page_chars.iter().map(|c| c.bbox.height).collect();
        char_heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_char_h = char_heights.get(char_heights.len() / 2).copied().unwrap_or(11.0);

        let mut blocks: Vec<String> = Vec::new();
        let dbg_blocks = std::env::var("BOB_DEBUG_BLOCKS").is_ok();
        // Areas already OCR'd by an earlier region's scanned-page fallback.
        let mut ocr_claimed: Vec<Rect> = Vec::new();
        for (ri, region) in sorted.iter().enumerate() {
            if dbg_blocks {
                eprintln!(
                    "DBG p{} region[{}] {:?} prio={} eff_pieces={}",
                    index + 1,
                    ri,
                    region.label,
                    region_claim_priority(&region.label),
                    region_lines[ri].len()
                );
            }
            // Convert layout coords (render pixels) → PDF points
            let bbox = Rect::new(
                region.x0 / scale,
                region.y0 / scale,
                (region.x1 - region.x0) / scale,
                (region.y1 - region.y0) / scale,
            );
            let lab = region.label.to_lowercase();
            if let Some(text) = caption_block(&page_chars, &region_lines[ri], &lab) {
                blocks.push(text);
                continue;
            }

            if lab.contains("table") {
                // Stage 1 — structured extraction from the text layer:
                // Tagged-PDF structure tree or ruled-grid detection scoped
                // to this region. Preserves cell/column structure that a
                // plain glyph dump destroys (borderless tables fall through
                // — the spatial detector rejects prose-shaped candidates).
                if self.config.structured_tables {
                    match pdf.tables_in_rect(index, bbox) {
                        Ok(tables) => {
                            let md: Vec<String> = tables
                                .iter()
                                .filter(|t| crate::tables::is_plausible_table(t))
                                .filter_map(crate::tables::source_table_markdown)
                                .collect();
                            if !md.is_empty() {
                                info!(
                                    "page {}: {} structured table(s) in region",
                                    index + 1,
                                    md.len()
                                );
                                blocks.push(md.join("\n\n"));
                                continue;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("page {}: structured table probe failed: {e}", index + 1)
                        }
                    }
                }
                // Stage 2 — unstructured dump of the region's text lines.
                let text = region_lines_to_text(&page_chars, &region_lines[ri]);
                if !text.trim().is_empty() {
                    let t = if self.config.convert_html_tables {
                        crate::tables::html_tables_to_gfm(&text)
                    } else {
                        text
                    };
                    blocks.push(t);
                } else if let Some(crop) = crop_image(&img, bbox, dpi, 0.0) {
                    // Scanned table: OCR the crop, then SLANet-plus structure.
                    match self
                        .engine
                        .ocr_lines(&crop)
                        .map(|lines| self.engine.recognize_table(&crop, &lines))
                    {
                        Ok(Ok(Some(html))) => {
                            info!(
                                "page {}: table recognized ({}px crop)",
                                index + 1,
                                crop.width()
                            );
                            let t = if self.config.convert_html_tables {
                                crate::tables::html_tables_to_gfm(&html)
                            } else {
                                html
                            };
                            blocks.push(t);
                        }
                        Ok(Ok(None)) | Ok(Err(_)) => {
                            blocks.push(format!("[table: {}]", region.label));
                        }
                        Err(_) => {
                            blocks.push(format!("[table: {}]", region.label));
                        }
                    }
                } else {
                    blocks.push(format!("[table: {}]", region.label));
                }
            } else if MATH_LAYOUT_LABELS.iter().any(|k| lab.contains(k)) {
                let page_h_px = img.height() as f64;
                let page_px = img.width() as f64 * page_h_px;
                // region coords are already render pixels - no extra scale
                let w_px = (region.x1 - region.x0) as f64;
                let h_px = (region.y1 - region.y0) as f64;
                let plausible = h_px <= 0.25 * page_h_px && w_px * h_px <= 0.15 * page_px;

                // Hybrid refinement (born-digital pages): intersect the
                // layout formula region with text-layer math boxes. The
                // layout model localizes formulas only at paragraph
                // granularity; glyph geometry pins the actual equation
                // strips. Region prose is kept and formulas are spliced in,
                // mirroring SURGICAL mode.
                let margin = 6.0_f32;
                let refined: Vec<Rect> = page_math_boxes
                    .iter()
                    .copied()
                    .filter(|b| {
                        let cx = b.x + b.width / 2.0;
                        let cy = b.y + b.height / 2.0;
                        cx >= bbox.x - margin
                            && cx <= bbox.x + bbox.width + margin
                            && cy >= bbox.y - margin
                            && cy <= bbox.y + bbox.height + margin
                    })
                    .collect();
                if !refined.is_empty() {
                    // consume so overlapping regions cannot re-emit them
                    let taken = refined.clone();
                    page_math_boxes.retain(|b| !taken.iter().any(|t| t.x == b.x && t.y == b.y));
                }

                if std::env::var("BOB_DEBUG_HYBRID").is_ok() {
                    eprintln!(
                        "HYBRID p{} region pt=({:.0},{:.0},{:.0},{:.0}) mathboxes={} refined={}",
                        index + 1,
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height,
                        page_math_boxes.len(),
                        refined.len()
                    );
                    for b in page_math_boxes.iter().take(10) {
                        eprintln!(
                            "  mb x={:.0} y={:.0} w={:.0} h={:.0}",
                            b.x, b.y, b.width, b.height
                        );
                    }
                }
                if !refined.is_empty() {
                    let text = region_lines_to_text(&page_chars, &region_lines[ri]);
                    let mut reps: Vec<(String, String)> = Vec::new();
                    for rb in &refined {
                        // Gate 1: skip labels ("(4)"), artifacts and tiny
                        // inline fragments - splicing them is pure noise.
                        if rb.width < 40.0 {
                            continue;
                        }
                        // Gate 2: display-style boxes only. Boxes taller
                        // than ~3 lines are merged blobs of display
                        // equations interleaved with inline-math prose
                        // lines; cropping them yields garbage.
                        let display = rb.height > 1.6 * line_height
                            || rb.width > self.config.formula_inline_max_width_pts as f32;
                        if !display || rb.height > 3.0 * line_height {
                            continue;
                        }
                        // Glyph-tight boxes: no padding, padding only
                        // bleeds neighbouring text lines into the crop.
                        let t_rec = std::time::Instant::now();
                        let crop_res = crop_image(&img, *rb, dpi, 0.0);
                        if std::env::var("BOB_DEBUG_HYBRID").is_ok() {
                            eprintln!(
                                "HYBRID-crop p{} w={:.0} h={:.0} (lh={:.1})",
                                index + 1,
                                rb.width,
                                rb.height,
                                line_height
                            );
                        }
                        if let Some(crop) = crop_res {
                            if crop.width() < 4 || crop.height() < 4 {
                                continue;
                            }
                            let p = work_dir.join(format!(
                                "_reg_hf_{}_{}.png",
                                index,
                                blocks.len() + reps.len()
                            ));
                            if crop.save(&p).is_ok() {
                                // Token budget proportional to crop area: a
                                // mis-cropped sliver cannot contain a large
                                // equation, and the cap turns runaway
                                // decodes (measured: 2600 chars / 28 s from
                                // a 153x30pt fragment) into fast failures.
                                let budget =
                                    ((rb.width * rb.height) / 40.0).clamp(64.0, 512.0) as usize;
                                if let Some(latex) =
                                    self.engine.recognize_formula_capped(&p, budget)?
                                {
                                    if std::env::var("BOB_DEBUG_HYBRID").is_ok() {
                                        eprintln!("HYBRID-png {}", p.display());
                                        eprintln!(
                                            "HYBRID-ocr p{} {} chars in {:?}",
                                            index + 1,
                                            latex.len(),
                                            t_rec.elapsed()
                                        );
                                    }
                                    if plausible_display_latex(&latex) {
                                        let needle = region_text(pdf, index, *rb);
                                        reps.push((needle, format!("$$\n{}\n$$", latex)));
                                    } else if std::env::var("BOB_DEBUG_HYBRID").is_ok() {
                                        eprintln!("HYBRID-reject p{} {:?}", index + 1, latex);
                                    }
                                }
                            }
                        }
                    }
                    if !text.trim().is_empty() {
                        blocks.push(splice(&text, &reps));
                        continue;
                    }
                    if !reps.is_empty() {
                        for (_, wrapped) in reps {
                            blocks.push(wrapped);
                        }
                        continue;
                    }
                }

                // Fallback (scans / no text layer): TexTeller on the whole
                // region crop, unless the box is implausibly large - a full
                // autoregressive decode of body text costs tens of seconds
                // and yields garbage LaTeX.
                if !plausible {
                    tracing::warn!(
                        "page {}: implausible {} region ({:.0}x{:.0}px, {:.0}% of page); treating as text",
                        index + 1,
                        region.label,
                        w_px,
                        h_px,
                        100.0 * (w_px * h_px) / page_px
                    );
                    let text = region_lines_to_text(&page_chars, &region_lines[ri]);
                    if !text.trim().is_empty() {
                        blocks.push(text);
                    }
                    continue;
                }
                // Born-digital shortcut: if the text layer has content here
                // but the heuristic found no math boxes, the region
                // provably contains no equation - do not spend a multi-second
                // decode on prose. TexTeller only when the text layer is
                // empty (scans, image-only equations), and even then the
                // decode is budgeted by crop area.
                let text = region_lines_to_text(&page_chars, &region_lines[ri]);
                if text.trim().is_empty() {
                    if let Some(crop) = crop_image(&img, bbox, dpi, self.config.formula_pad_pts) {
                        if crop.width() >= 4 && crop.height() >= 4 {
                            let p = work_dir.join(format!("_reg_f_{}.png", blocks.len()));
                            if crop.save(&p).is_ok() {
                                let budget =
                                    ((bbox.width * bbox.height) / 40.0).clamp(64.0, 512.0) as usize;
                                if let Some(latex) =
                                    self.engine.recognize_formula_capped(&p, budget)?
                                {
                                    blocks.push(format!("$$\n{}\n$$", latex));
                                    continue;
                                }
                            }
                        }
                    }
                } else {
                    blocks.push(text);
                }
            } else if lab.contains("figure") || lab.contains("image") {
                if let Some(link) = figure_block(
                    &self.config,
                    &sorted,
                    &region_lines,
                    &page_chars,
                    bbox,
                    scale,
                    median_char_h,
                    assets,
                    &img,
                    dpi,
                    work_dir,
                    index,
                    blocks.len(),
                ) {
                    blocks.push(link);
                }
            } else {
                // Text region: prefer PDF text layer, fall back to OCR
                let text = region_lines_to_text(&page_chars, &region_lines[ri]);
                if !text.trim().is_empty() {
                    let mut t = text;
                    if lab.contains("title") {
                        t = format!("## {}", t);
                    }
                    blocks.push(t);
                } else {
                    // OCR fallback for scanned pages (no text layer → no
                    // glyph ownership). OCR the region's base bbox at most
                    // once: skip when a previous region already OCR'd most
                    // of this area, else OCR the whole box and record it.
                    // Re-OCR per region used to emit every line once per
                    // overlapping layout box.
                    let base = Rect::new(
                        region.x0,
                        region.y0,
                        region.x1 - region.x0,
                        region.y1 - region.y0,
                    );
                    let base_area = (base.width * base.height).max(1e-6);
                    let already = ocr_claimed.iter().any(|r| {
                        let iw = (r.x + r.width).min(base.x + base.width) - r.x.max(base.x);
                        let ih = (r.y + r.height).min(base.y + base.height) - r.y.max(base.y);
                        iw > 0.0 && ih > 0.0 && (iw * ih) / base_area > 0.25
                    });
                    if already {
                        continue;
                    }
                    if let Some(crop) = crop_image(&img, base, dpi, 0.0) {
                        if let Ok(lines) = self.engine.ocr_lines(&crop) {
                            let t: String = lines
                                .into_iter()
                                .map(|l| l.text)
                                .filter(|t| !t.is_empty())
                                .collect::<Vec<_>>()
                                .join(" ");
                            if !t.trim().is_empty() {
                                ocr_claimed.push(base);
                                let mut t = t;
                                if lab.contains("title") {
                                    t = format!("## {}", t);
                                }
                                blocks.push(t);
                            }
                        }
                    }
                }
            }
        }

        // Seam repair (see `repair_seams`): homeless glyph lines are
        // appended; region-adopted lines rejoin `region_lines`.
        repair_seams(
            &page_chars,
            &sorted,
            &glyph_owner,
            &mut region_lines,
            &mut blocks,
            scale,
            index,
        );

        let md = blocks.join("\n\n");
        if dbg_blocks {
            for (bi, b) in blocks.iter().enumerate() {
                eprintln!("DBG p{} block[{}] {:?}...", index + 1, bi, b.chars().take(60).collect::<String>());
            }
        }
        if md.trim().is_empty() {
            Ok(None)
        } else {
            Ok(Some(md))
        }
    }
}

// ======================================================================
// Code block detection (monospace font runs)
// ======================================================================

#[allow(dead_code)]
fn wrap_code_blocks(pdf: &mut dyn PdfSource, index: usize) -> Vec<String> {
    let chars = match pdf.chars(index) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    let mut lines: std::collections::BTreeMap<i32, String> = std::collections::BTreeMap::new();
    for c in &chars {
        if is_mono_font(&c.font_name) {
            let y_key = (c.bbox.y * 10.0) as i32;
            lines.entry(y_key).or_default().push(c.char);
        }
    }
    let mut blocks: Vec<String> = vec![];
    for text in lines.values() {
        let trimmed: String = text.chars().collect();
        if trimmed.len() > 4 {
            blocks.push(format!("```\n{}\n```", trimmed));
        }
    }
    blocks
}

fn is_mono_font(font_name: &str) -> bool {
    let lower = font_name.to_lowercase();
    MONO_FONT_KEYWORDS.iter().any(|k| lower.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdf_source::fake::{FakePage, FakePdf};

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bobine_conv_{}", name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // ==================================================================
    // PdfSource seam end-to-end routing (legacy test_converter_flow port)
    // ==================================================================

    #[test]
    fn never_mode_joins_pages_and_fires_hooks() {
        let mut pdf = FakePdf::new(vec![
            FakePage::from_text("Page one text", "Helvetica"),
            FakePage::from_text("Page two text", "Helvetica"),
            FakePage::from_text("Page three text", "Helvetica"),
        ]);
        let mut conv = HybridConverter::new(
            ConverterConfig {
                routing_mode: RoutingMode::Never,
                ..Default::default()
            },
            &temp_dir("never"),
        );
        let pages = std::cell::RefCell::new(0usize);
        let hooks = ProgressHooks {
            should_continue: Box::new(|| true),
            on_page: Box::new(|_, _| *pages.borrow_mut() += 1),
        };
        let out = conv
            .convert_pdf_source(&mut pdf, Path::new("/tmp"), &hooks)
            .unwrap();
        assert_eq!(*pages.borrow(), 3);
        assert_eq!(out.matches("\n\n---\n\n").count(), 2);
        assert!(out.contains("Page one text"));
        assert!(out.contains("Page three text"));
    }

    #[test]
    fn cancellation_stops_mid_document() {
        let mut pdf = FakePdf::new(vec![
            FakePage::from_text("first page words", "Helvetica"),
            FakePage::from_text("second page words", "Helvetica"),
            FakePage::from_text("third page words", "Helvetica"),
        ]);
        let mut conv = HybridConverter::new(
            ConverterConfig {
                routing_mode: RoutingMode::Never,
                ..Default::default()
            },
            &temp_dir("cancel"),
        );
        let n = std::cell::Cell::new(0usize);
        let hooks = ProgressHooks {
            should_continue: Box::new(|| n.get() < 2), // stop before page 3
            on_page: Box::new(|_, _| n.set(n.get() + 1)),
        };
        let out = conv
            .convert_pdf_source(&mut pdf, Path::new("/tmp"), &hooks)
            .unwrap();
        assert_eq!(n.get(), 2);
        assert!(out.contains("first page"));
        assert!(out.contains("second page"));
        assert!(!out.contains("third page"));
    }

    #[test]
    fn rect_iou_basics() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!((rect_iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0); // 25 overlap, 175 union
        assert!((rect_iou(&a, &b) - 25.0 / 175.0).abs() < 1e-6);
        let c = Rect::new(100.0, 100.0, 5.0, 5.0);
        assert_eq!(rect_iou(&a, &c), 0.0);
    }

    #[test]
    fn embedded_match_prefers_best_coverage() {
        let region = Rect::new(100.0, 100.0, 200.0, 100.0);
        let assets = vec![
            EmbeddedAsset {
                rel_path: "weak.png".to_string(),
                bbox_pts: Some(Rect::new(90.0, 90.0, 40.0, 30.0)), // mostly outside
            },
            EmbeddedAsset {
                rel_path: "strong.png".to_string(),
                bbox_pts: Some(Rect::new(105.0, 105.0, 190.0, 92.0)), // ~92% covered
            },
            EmbeddedAsset {
                rel_path: "unplaced.png".to_string(),
                bbox_pts: None,
            },
        ];
        let hit = best_embedded_match(&region, &assets, 0.6).unwrap();
        assert_eq!(assets[hit].rel_path, "strong.png");
        // No match at all when nothing is sufficiently covered.
        let far = Rect::new(500.0, 500.0, 50.0, 50.0);
        assert!(best_embedded_match(&far, &assets, 0.6).is_none());
        // Coverage (not IoU): a region that fully subsumes a small raster
        // matches even though symmetric overlap would be low.
        let big_region = Rect::new(0.0, 0.0, 400.0, 300.0);
        let small = vec![EmbeddedAsset {
            rel_path: "small.png".to_string(),
            bbox_pts: Some(Rect::new(10.0, 10.0, 30.0, 20.0)), // IoU ≈ 0.005
        }];
        assert!(best_embedded_match(&big_region, &small, 0.6).is_some());
    }

    #[test]
    fn figure_region_dedup_drops_overlap_keeps_text() {
        use crate::rapid_layout::LayoutRegion;
        let mk = |x0: f32, y0: f32, x1: f32, y1: f32, label: &str| LayoutRegion {
            x0,
            y0,
            x1,
            y1,
            label: label.into(),
            confidence: 0.9,
        };
        let regions = vec![
            mk(50.0, 50.0, 250.0, 250.0, "figure"),
            mk(55.0, 55.0, 255.0, 255.0, "figure"), // IoU ≈ 0.82 with first
            mk(300.0, 50.0, 500.0, 200.0, "plain text"), // overlaps nothing
            mk(310.0, 60.0, 510.0, 210.0, "plain text"), // text dup: NOT dropped here
        ];
        let kept = dedup_figure_regions(regions);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept.iter().filter(|r| r.label == "figure").count(), 1);
    }

    #[test]
    fn gallery_appends_unreferenced_images() {
        let img = image::DynamicImage::new_rgb8(4, 4);
        let with_img = FakePage::from_text("caption-less page", "Helvetica").with_image(img);
        let dir = temp_dir("gallery");
        let mut conv = HybridConverter::new(ConverterConfig::default(), &dir);
        let mut pdf = FakePdf::new(vec![with_img]);
        let out = conv
            .convert_pdf_source(&mut pdf, &dir.join("work"), &ProgressHooks::default())
            .unwrap();
        assert!(out.contains("![](assets/p0/img0.png)"), "{}", out);
        assert!(out.contains("assets"), "structured path expected: {}", out);

        // a page already referencing an image inline still gets its OTHER
        // embedded images appended (per-asset rule, not all-or-nothing)
        let md_page = FakePage::from_text("", "Helvetica")
            .with_markdown("![inline](x.png)")
            .with_image(image::DynamicImage::new_rgb8(4, 4));
        let dir2 = temp_dir("gallery2");
        let mut pdf2 = FakePdf::new(vec![md_page]);
        let out2 = conv
            .convert_pdf_source(&mut pdf2, &dir2.join("work"), &ProgressHooks::default())
            .unwrap();
        assert!(out2.contains("![inline](x.png)"), "{}", out2);
        assert!(out2.contains("![](assets/p0/img0.png)"), "{}", out2);

        // a page that already references ALL its images gets no gallery tail
        let md_page3 = FakePage::from_text("", "Helvetica")
            .with_markdown("see ![](assets/p0/img0.png) above")
            .with_image(image::DynamicImage::new_rgb8(4, 4));
        let dir3 = temp_dir("gallery3");
        let mut pdf3 = FakePdf::new(vec![md_page3]);
        let out3 = conv
            .convert_pdf_source(&mut pdf3, &dir3.join("work"), &ProgressHooks::default())
            .unwrap();
        assert_eq!(out3.matches("img0.png").count(), 1, "{}", out3);
    }

    #[test]
    fn promote_headings_patterns() {
        let md = "**I.** **INTRODUCTION**\n\nsome prose paragraph.\n\n**3.1** **Encoder** **and** **Decoder** **Stacks**\n\n3.1.1 Sub-sub section title\n\nA. Setup details";
        let out = promote_headings(md);
        assert!(out.contains("## INTRODUCTION"), "{out}");
        assert!(out.contains("### Encoder and Decoder Stacks"), "{out}");
        assert!(out.contains("#### Sub-sub section title"), "{out}");
        assert!(out.contains("### Setup details"), "{out}");
        // Prose must not be touched.
        assert!(out.contains("some prose paragraph."));
    }

    #[test]
    fn promote_headings_guards() {
        // Numbered list items in sentence case stay untouched.
        let md = "1. Set up the environment before running.\n2. Install dependencies";
        assert_eq!(promote_headings(md), md);
        // Tables, existing headings, code fences untouched.
        let md = "| a | b |\n### already\n```\nI. NOT A HEADING\n```";
        assert_eq!(promote_headings(md), md);
        // Long sentence-case line after a number → prose, not heading.
        let md = "4. We now turn to the analysis of the general case with several more words following it here";
        assert_eq!(promote_headings(md), md);
    }

    #[test]
    fn promote_document_title_cases() {
        // Case 1: first block is a bare title line.
        let md = "Vector Edge Solitons and Domain Walls\n\nDavid Snee\n\n## I. Intro";
        let out = promote_document_title(md);
        assert!(out.starts_with("# Vector Edge Solitons and Domain Walls"), "{out}");
        // Case 2: junk first (ends with '.'), first heading promoted to #.
        let md = "Provided proper attribution is provided, Google grants permission.\n\n## Attention Is All You Need\n\ntext";
        let out = promote_document_title(md);
        assert!(out.contains("# Attention Is All You Need"), "{out}");
        // No promotion when the candidate heading sits beyond page 1.
        let md = "Long introductory prose sentence ends here.\n\nbody continues\n\nmore prose.\n\n---\n\nmore pages follow here.\n\n### Deep heading";
        assert_eq!(promote_document_title(md), md);
        // Layout-model junk labelled "title" (email + equation debris) must
        // NOT be promoted.
        let md = "## \u{2217} yiping.ma@northumbria.ac.uk Here we focus on high-frequ\n\n## Vector Edge Solitons and Domain Walls";
        let out = promote_document_title(md);
        assert!(out.starts_with("## \u{2217}"), "{out}");
    }

    #[test]
    fn caption_alt_detection() {
        assert_eq!(
            caption_alt("Figure 1: Temporal L2 error at time T=1", 120).as_deref(),
            Some("Figure 1: Temporal L2 error at time T=1")
        );
        assert_eq!(caption_alt("Fig. S1  setup", 120).as_deref(), Some("Fig. S1 setup"));
        assert_eq!(caption_alt("Table 2: results", 120).as_deref(), Some("Table 2: results"));
        // Non-captions → None (empty alt is better than a wrong one).
        assert_eq!(caption_alt("We show that the scheme converges.", 120), None);
        assert_eq!(caption_alt("configuration of the model", 120), None);
        // Truncation at word boundary with ellipsis.
        let long = caption_alt("Figure 3: a very long caption that keeps going and going beyond the limit", 40).unwrap();
        assert!(long.ends_with('…') && long.chars().count() <= 41, "{long}");
        // Square brackets stripped for ![...] syntax safety.
        assert_eq!(caption_alt("Figure 4: [normalized] error", 120).as_deref(),
            Some("Figure 4: normalized error"));
        // Whitespace collapsed.
        assert_eq!(caption_alt("Figure   5:\ttitle", 120).as_deref(), Some("Figure 5: title"));
    }

    #[test]
    fn interleave_places_asset_between_text_bands() {
        // Two text bands at y=100 and y=300; an asset whose bbox bottom sits
        // between them must appear between the two texts in the markdown.
        let mut page = FakePage::default();
        page.media_box = [0.0, 0.0, 612.0, 792.0];
        page.md = Some(
            "First paragraph text.

Figure 1: Test caption here

Second paragraph text."
                .into(),
        );
        for (i, ch) in "First paragraph text".chars().enumerate() {
            page.chars.push(SourceChar::new(ch, 50.0 + i as f32 * 8.0, 100.0, 7.0, 11.0, "Helvetica"));
        }
        for (i, ch) in "Figure 1: Test caption here".chars().enumerate() {
            page.chars.push(SourceChar::new(ch, 50.0 + i as f32 * 8.0, 255.0, 7.0, 11.0, "Helvetica"));
        }
        for (i, ch) in "Second paragraph text".chars().enumerate() {
            page.chars.push(SourceChar::new(ch, 50.0 + i as f32 * 8.0, 300.0, 7.0, 11.0, "Helvetica"));
        }
        let img = image::DynamicImage::new_rgb8(10, 10);
        page.images.push(img);
        page.image_bboxes.push(Some(Rect::new(50.0, 150.0, 200.0, 100.0))); // bottom=250
        let dir = temp_dir("interleave");
        let mut conv = HybridConverter::new(ConverterConfig::default(), &dir);
        let mut pdf = FakePdf::new(vec![page]);
        let out = conv
            .convert_pdf_source(&mut pdf, &dir.join("work"), &ProgressHooks::default())
            .unwrap();
        let first = out.find("First paragraph").unwrap();
        let link = out.find("![").unwrap();
        let second = out.find("Second paragraph").unwrap();
        assert!(first < link && link < second, "link must sit between bands: {out}");
        // The caption line below the asset becomes the alt text.
        assert!(
            out.contains("![Figure 1: Test caption here](assets/p0/img0.png)"),
            "alt text expected: {out}"
        );
    }

    #[test]
    fn surgical_without_math_returns_fast_markdown() {
        let mut pdf = FakePdf::new(vec![FakePage::from_text(
            "plain body prose only",
            "Helvetica",
        )]);
        let mut conv = HybridConverter::new(
            ConverterConfig {
                routing_mode: RoutingMode::Surgical,
                ..Default::default()
            },
            &temp_dir("surgical_plain"),
        );
        let out = conv
            .convert_pdf_source(
                &mut pdf,
                Path::new("/tmp/nowhere"),
                &ProgressHooks::default(),
            )
            .unwrap();
        assert!(out.contains("plain body prose"), "{}", out);
    }

    #[test]
    fn auto_mode_plain_ascii_page_stays_on_fast_path() {
        let mut pdf = FakePdf::new(vec![FakePage::from_text(
            "just regular sentences here",
            "Helvetica",
        )]);
        let mut conv = HybridConverter::new(ConverterConfig::default(), &temp_dir("auto_plain"));
        let out = conv
            .convert_pdf_source(
                &mut pdf,
                Path::new("/tmp/nowhere"),
                &ProgressHooks::default(),
            )
            .unwrap();
        assert!(out.contains("regular sentences"), "{}", out);
    }

    #[test]
    fn fast_path_falls_back_to_char_join_when_md_errors() {
        let mut page = FakePage::from_text("fallback chars", "Helvetica");
        page.md = None; // simulate to_markdown failure
        let mut pdf = FakePdf::new(vec![page]);
        let mut conv = HybridConverter::new(
            ConverterConfig {
                routing_mode: RoutingMode::Never,
                ..Default::default()
            },
            &temp_dir("fallback"),
        );
        let out = conv
            .convert_pdf_source(&mut pdf, Path::new("/tmp"), &ProgressHooks::default())
            .unwrap();
        assert!(out.contains("fallback chars"), "{}", out);
    }

    // ==================================================================
    // math_boxes_from_chars grouping (free fn, no ONNX)
    // ==================================================================

    #[test]
    fn math_boxes_group_lines_and_respect_min_chars() {
        // one line of 6 math-font chars at y=100
        let mut chars: Vec<SourceChar> = (0..6)
            .map(|i| SourceChar::new('x', 10.0 + i as f32 * 8.0, 100.0, 7.0, 11.0, "cmmi"))
            .collect();
        // a second math line just below -> merges vertically into the same box
        for i in 0..4 {
            chars.push(SourceChar::new(
                'y',
                12.0 + i as f32 * 8.0,
                112.0,
                7.0,
                11.0,
                "msam",
            ));
        }
        // body-font chars elsewhere are ignored entirely
        for i in 0..20 {
            chars.push(SourceChar::new(
                'a',
                10.0 + i as f32 * 8.0,
                400.0,
                7.0,
                11.0,
                "Helvetica",
            ));
        }
        let mut fake = FakePdf::default();
        fake.pages[0].chars = chars;

        let boxes = math_boxes_from_chars(&mut fake, 0, 5, 1.5, 1.5);
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        let b = boxes[0];
        assert!(b.y <= 100.0 && b.y + b.height >= 123.0, "{b:?}");

        // raising min_chars above both line sizes filters everything
        let none = math_boxes_from_chars(&mut fake, 0, 11, 1.5, 1.5);
        assert!(none.is_empty());
    }

    // ==================================================================
    // is_math_unicode / is_math_font
    // ==================================================================

    #[test]
    fn greek_is_math() {
        assert!(is_math_unicode('α'));
        assert!(is_math_unicode('β'));
        assert!(is_math_unicode('Σ'));
    }

    #[test]
    fn math_operator_is_math() {
        assert!(is_math_unicode('∀')); // U+2200
        assert!(is_math_unicode('∫')); // U+222B
        assert!(is_math_unicode('∑')); // U+2211
    }

    #[test]
    fn ascii_letter_not_math() {
        assert!(!is_math_unicode('a'));
        assert!(!is_math_unicode('Z'));
        assert!(!is_math_unicode('5'));
    }

    #[test]
    fn math_font_keyword_match() {
        assert!(is_math_font("cmmi10"));
        assert!(is_math_font("STIXMath"));
        assert!(is_math_font("XITS Math"));
    }

    #[test]
    fn body_font_not_math() {
        assert!(!is_math_font("Times New Roman"));
        assert!(!is_math_font("Arial"));
        assert!(!is_math_font("cmr10")); // cmr is roman, not math
    }

    #[test]
    fn mono_font_keyword_match() {
        assert!(is_mono_font("Courier New"));
        assert!(is_mono_font("Consolas"));
        assert!(is_mono_font("DejaVu Sans Mono"));
    }

    #[test]
    fn proportional_font_not_mono() {
        assert!(!is_mono_font("Times New Roman"));
        assert!(!is_mono_font("Helvetica"));
    }

    // ==================================================================
    // ws_replace
    // ==================================================================

    #[test]
    fn ws_replace_simple() {
        let result = ws_replace("Hello   world", "Hello world", "Goodbye");
        assert_eq!(result, Some("Goodbye".into()));
    }

    #[test]
    fn ws_replace_preserves_surrounding() {
        let result = ws_replace("aa hello  world bb", "hello world", "X");
        assert_eq!(result, Some("aa X bb".into()));
    }

    #[test]
    fn ws_replace_not_found() {
        let result = ws_replace("hello world", "goodbye", "X");
        assert!(result.is_none());
    }

    #[test]
    fn ws_replace_empty_needle() {
        assert!(ws_replace("text", "   ", "X").is_none());
    }

    // ==================================================================
    // splice
    // ==================================================================

    #[test]
    fn splice_exact_match() {
        let md = "The formula is x = y";
        let repls = vec![("x = y".into(), "$x = y$".into())];
        assert_eq!(splice(md, &repls), "The formula is $x = y$");
    }

    #[test]
    fn splice_ws_tolerant() {
        let md = "The formula is x   =   y";
        let repls = vec![("x = y".into(), "$x = y$".into())];
        assert_eq!(splice(md, &repls), "The formula is $x = y$");
    }

    #[test]
    fn splice_leftover_appended() {
        let md = "some text";
        let repls = vec![("not found".into(), "replacement".into())];
        let result = splice(md, &repls);
        assert!(result.contains("some text"));
        assert!(result.contains("replacement"));
    }

    #[test]
    fn splice_empty_needle_appends() {
        let md = "base";
        let repls = vec![("".into(), "extra".into())];
        let result = splice(md, &repls);
        assert!(result.contains("base"));
        assert!(result.contains("extra"));
    }

    #[test]
    fn needs_onnx_flags_scanned_pages_in_auto_mode() {
        let scanned_page = FakePage::from_text("tiny", "Helvetica") // < threshold chars
            .with_image(image::DynamicImage::new_rgb8(2, 2));
        let mut scanned = FakePdf::new(vec![scanned_page]);
        let cfg = ConverterConfig::default();
        assert!(needs_onnx(&mut scanned, 0, &cfg));

        let textual_page = FakePage::from_text("word word word word word word", "Helvetica");
        let mut textual = FakePdf::new(vec![textual_page]);
        assert!(!needs_onnx(&mut textual, 0, &cfg));

        // Always overrides everything; Never suppresses everything
        let cfg_always = ConverterConfig {
            routing_mode: RoutingMode::Always,
            ..Default::default()
        };
        assert!(needs_onnx(&mut textual, 0, &cfg_always));
        let cfg_never = ConverterConfig {
            routing_mode: RoutingMode::Never,
            ..Default::default()
        };
        assert!(!needs_onnx(&mut scanned, 0, &cfg_never));
    }

    #[test]
    fn wrap_code_blocks_emits_fence_for_mono_lines() {
        let mut page = FakePage::from_text("", "Helvetica");
        page.chars.clear();
        for (i, ch) in "let x = 42;".chars().enumerate() {
            page.chars.push(SourceChar::new(
                ch,
                50.0 + i as f32 * 8.0,
                300.0,
                7.0,
                11.0,
                "Consolas",
            ));
        }
        let mut fake = FakePdf::new(vec![page]);
        let blocks = wrap_code_blocks(&mut fake, 0);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].starts_with("```"), "{}", blocks[0]);
        assert!(blocks[0].contains("let x = 42;"));

        // proportional font means no block
        let prop = FakePage::from_text("hello world", "Georgia");
        let mut prop_doc = FakePdf::new(vec![prop]);
        assert!(wrap_code_blocks(&mut prop_doc, 0).is_empty());
    }

    // ==================================================================
    // Overlap dedup: rect_subtract + claim priority
    // ==================================================================

    #[test]
    fn rect_subtract_no_overlap_returns_base() {
        let base = Rect::new(0.0, 0.0, 100.0, 100.0);
        let subs = vec![Rect::new(200.0, 200.0, 10.0, 10.0)];
        let out = rect_subtract(base, &subs);
        assert_eq!(out.len(), 1);
        assert!((out[0].x - 0.0).abs() < 1e-6);
        assert!((out[0].width - 100.0).abs() < 1e-6);
        assert!((out[0].height - 100.0).abs() < 1e-6);
    }

    #[test]
    fn rect_subtract_full_cover_is_empty() {
        let base = Rect::new(10.0, 10.0, 50.0, 50.0);
        let subs = vec![Rect::new(0.0, 0.0, 100.0, 100.0)];
        let out = rect_subtract(base, &subs);
        assert!(out.is_empty());
    }

    #[test]
    fn rect_subtract_central_hole_yields_four_pieces() {
        let base = Rect::new(0.0, 0.0, 100.0, 100.0);
        let subs = vec![Rect::new(40.0, 40.0, 20.0, 20.0)];
        let out = rect_subtract(base, &subs);
        assert_eq!(out.len(), 4);
        let area: f32 = out.iter().map(|r| r.width * r.height).sum();
        assert!((area - 9600.0).abs() < 1e-3, "area {area}");
        // No piece overlaps the hole
        for r in &out {
            let overlaps = !(r.x >= 60.0 - 1e-3
                || r.x + r.width <= 40.0 + 1e-3
                || r.y >= 60.0 - 1e-3
                || r.y + r.height <= 40.0 + 1e-3);
            assert!(!overlaps, "piece {:?} overlaps hole", r);
        }
    }

    #[test]
    fn rect_subtract_left_third_strips_to_one_piece() {
        let base = Rect::new(0.0, 0.0, 90.0, 30.0);
        let subs = vec![Rect::new(0.0, 0.0, 30.0, 30.0)];
        let out = rect_subtract(base, &subs);
        assert_eq!(out.len(), 1);
        assert!((out[0].x - 30.0).abs() < 1e-6);
        assert!((out[0].width - 60.0).abs() < 1e-6);
        assert!((out[0].height - 30.0).abs() < 1e-6);
    }

    #[test]
    fn rect_subtract_two_subs_preserves_area() {
        // Base 100x100; sub A = full-height left quarter; sub B = 20x20
        // hole in the remaining 3/4. Area must be 10000 - 2500 - 400.
        let base = Rect::new(0.0, 0.0, 100.0, 100.0);
        let subs = vec![
            Rect::new(0.0, 0.0, 25.0, 100.0),
            Rect::new(60.0, 40.0, 20.0, 20.0),
        ];
        let out = rect_subtract(base, &subs);
        let area: f32 = out.iter().map(|r| r.width * r.height).sum();
        assert!((area - 7100.0).abs() < 1e-3, "area {area}");
        // Pieces are pairwise disjoint
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                let (a, b) = (&out[i], &out[j]);
                let disjoint = a.x >= b.x + b.width - 1e-3
                    || b.x >= a.x + a.width - 1e-3
                    || a.y >= b.y + b.height - 1e-3
                    || b.y >= a.y + a.height - 1e-3;
                assert!(disjoint, "pieces {i} and {j} overlap");
            }
        }
    }

    #[test]
    fn claim_priority_mirrors_routing() {
        assert_eq!(region_claim_priority("isolate_formula"), CLAIM_MATH);
        assert_eq!(region_claim_priority("formula"), CLAIM_MATH);
        assert_eq!(region_claim_priority("equation"), CLAIM_MATH);
        // Caption substring wins over the math/table substrings
        assert_eq!(region_claim_priority("formula_caption"), CLAIM_CAPTION);
        assert_eq!(region_claim_priority("table_caption"), CLAIM_CAPTION);
        assert_eq!(region_claim_priority("table_footnote"), CLAIM_CAPTION);
        assert_eq!(region_claim_priority("table"), CLAIM_TABLE);
        assert_eq!(region_claim_priority("plain text"), CLAIM_TEXT);
        assert_eq!(region_claim_priority("title"), CLAIM_TEXT);
        assert_eq!(region_claim_priority("figure"), CLAIM_NONE);
        assert_eq!(region_claim_priority("image"), CLAIM_NONE);
    }
}
