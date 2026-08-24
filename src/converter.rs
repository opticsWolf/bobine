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
    "cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol", "mathjax",
    "stix", "xits", "asana", "euclid",
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
    "mono", "courier", "consol", "menlo", "inconsolata", "sourcecode",
    "dejavu sans mono", "fixed", "terminal",
];

const MATH_LAYOUT_LABELS: &[&str] = &[
    "equation", "display_formula", "inline_formula", "isolate_formula", "formula",
];

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
        let mut pdf = Pdf::open(path)
            .map_err(|e| BobineError::PdfOxide(format!("open: {e}")))?;
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
            if self.config.extract_images {
                if let Err(e) = self.extract_page_images(pdf, i, work_dir) {
                    warn!("image extraction failed on page {}: {e}", i + 1);
                }
            }
            let page_md = self.route_page(pdf, i, work_dir)?;
            let final_md =
                if self.config.extract_images && self.config.append_unreferenced_images {
                    self.maybe_append_gallery(page_md, work_dir, i)
                } else {
                    page_md
                };
            blocks.push(final_md);
        }

        Ok(blocks.join("\n\n---\n\n"))
    }

    // ==================================================================
    // Per-page routing
    // ==================================================================

    fn route_page(&mut self, pdf: &mut dyn PdfSource, index: usize, work_dir: &Path) -> Result<String> {
        let md = self.route_page_inner(pdf, index, work_dir)?;
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
    ) -> Result<String> {
        if !self.config.use_onnx || self.config.routing_mode == RoutingMode::Never {
            return fast_page_markdown(pdf, index);
        }

        if self.config.routing_mode == RoutingMode::Surgical {
            if is_scanned(pdf, index, self.config.scanned_text_threshold) {
                self.engine.ensure_models()?;
                if let Ok(Some(md)) =
                    self.full_structure_page_markdown(pdf, index, work_dir)
                {
                    if !md.trim().is_empty() {
                        info!("page {}: scanned → ONNX layout+OCR", index + 1);
                        return Ok(md);
                    }
                }
                return fast_page_markdown(pdf, index);
            }
            return self.surgical_page_markdown(pdf, index, work_dir);
        }

        // AUTO / ALWAYS
        if needs_onnx(pdf, index, &self.config) {
            if let Ok(Some(md)) =
                self.full_structure_page_markdown(pdf, index, work_dir)
            {
                if !md.trim().is_empty() {
                    info!("page {} → ONNX layout+OCR", index + 1);
                    return Ok(md);
                }
            }
        }

        fast_page_markdown(pdf, index)
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

        let mut boxes = math_boxes_from_chars(pdf, index, self.config.min_formula_math_chars);
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
                            MATH_LAYOUT_LABELS.iter()
                                .any(|k| r.label.to_lowercase().contains(k))
                        })
                        .map(|r| Rect::new(
                            r.x0 / scale, r.y0 / scale,
                            (r.x1 - r.x0) / scale, (r.y1 - r.y0) / scale,
                        ))
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

        info!("page {}: {} formula region(s) → OCR", index + 1, crops.len());

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

    fn extract_page_images(&self, pdf: &mut dyn PdfSource, index: usize, dir: &Path) -> Result<()> {
        pdf.extract_image_files(index, dir, &format!("p{}_img", index))?;
        Ok(())
    }

    fn maybe_append_gallery(&self, md: String, dir: &Path, page: usize) -> String {
        if md.contains("![") {
            return md;
        }
        let prefix = format!("p{}_img", page);
        let mut gallery = String::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with(&prefix) {
                    gallery.push_str(&format!("![]({})\n", name_str));
                }
            }
        }
        if gallery.is_empty() {
            md
        } else {
            format!("{}\n\n{}", md, gallery)
        }
    }

    pub fn convert_office(&self, path: &Path) -> Result<String> {
        use office_oxide::Document;
        let doc = Document::open(path)
            .map_err(|e| BobineError::OfficeOxide(format!("open: {e}")))?;
        Ok(doc.to_markdown())
    }
}

// ======================================================================
// Free functions
// ======================================================================

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
        let mut heights: Vec<f32> =
            chars.iter().map(|c| c.bbox.height).filter(|&h| h > 0.0).collect();
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
    MATH_UNICODE_RANGES.iter().any(|&(lo, hi)| cp >= lo && cp <= hi)
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

fn math_boxes_from_chars(pdf: &mut dyn PdfSource, index: usize, min_chars: usize) -> Vec<Rect> {
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
    let hgap = 1.5 * med_h;
    let vgap_max = 1.5 * med_h;

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
                if r.x <= run_r.x + run_r.width + hgap
                    && r.x + r.width >= run_r.x - hgap
                {
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
        a.y
            .partial_cmp(&b.y)
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
                let gap = (b.y - (u.y + u.height)).max(u.y - (b.y + b.height)).max(0.0);
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
    ) -> Result<Option<String>> {
        let dpi = self.config.render_dpi;
        let img = match render_page_image(pdf, index, dpi) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        let media = page_media_box(pdf, index)?;
        let _page_h = media[3];
        let scale = dpi as f32 / 72.0;

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

        let mut blocks: Vec<String> = Vec::new();
        for region in &sorted {
            // Convert layout coords (render pixels) → PDF points
            let bbox = Rect::new(
                region.x0 / scale, region.y0 / scale,
                (region.x1 - region.x0) / scale, (region.y1 - region.y0) / scale,
            );
            let lab = region.label.to_lowercase();

            if lab.contains("table") {
                let text = region_text(pdf, index, bbox);
                if !text.trim().is_empty() {
                    let t = if self.config.convert_html_tables {
                        crate::tables::html_tables_to_gfm(&text)
                    } else {
                        text
                    };
                    blocks.push(t);
                } else if let Some(crop) = crop_image(&img, bbox, dpi, 0.0) {
                    // Scanned table: OCR the crop, then SLANet-plus structure.
                    match self.engine.ocr_lines(&crop).map(|lines| {
                        self.engine.recognize_table(&crop, &lines)
                    }) {
                        Ok(Ok(Some(html))) => {
                            info!("page {}: table recognized ({}px crop)", index + 1, crop.width());
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
                // Formula region → TexTeller
                if let Some(crop) = crop_image(&img, bbox, dpi, self.config.formula_pad_pts) {
                    if crop.width() >= 4 && crop.height() >= 4 {
                        let p = work_dir.join(format!("_reg_f_{}.png", blocks.len()));
                        if crop.save(&p).is_ok() {
                            if let Some(latex) = self.engine.recognize_formula(&p)? {
                                blocks.push(format!("$$\n{}\n$$", latex));
                                continue;
                            }
                        }
                    }
                }
            } else if lab.contains("figure") || lab.contains("image") {
                if let Some(crop) = crop_image(&img, bbox, dpi, 0.0) {
                    let p = work_dir.join(format!("_reg_fig_{}.png", blocks.len()));
                    if crop.save(&p).is_ok() {
                        blocks.push(format!("![]({})", p.file_name().unwrap().to_string_lossy()));
                        continue;
                    }
                }
            } else {
                // Text region: prefer PDF text layer, fall back to OCR
                let text = region_text(pdf, index, bbox);
                if !text.trim().is_empty() {
                    let mut t = text;
                    if lab.contains("title") {
                        t = format!("## {}", t);
                    }
                    blocks.push(t);
                } else {
                    // OCR fallback for scanned regions
                    if let Some(crop) = crop_image(&img, bbox, dpi, 0.0) {
                        if let Ok(lines) = self.engine.ocr_lines(&crop) {
                            let text: Vec<String> = lines.into_iter()
                                .map(|l| l.text)
                                .filter(|t| !t.is_empty())
                                .collect();
                            if !text.is_empty() {
                                let mut t = text.join(" ");
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

        let md = blocks.join("\n\n");
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
            ConverterConfig { routing_mode: RoutingMode::Never, ..Default::default() },
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
            ConverterConfig { routing_mode: RoutingMode::Never, ..Default::default() },
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
    fn gallery_appends_unreferenced_images() {
        let img = image::DynamicImage::new_rgb8(4, 4);
        let with_img =
            FakePage::from_text("caption-less page", "Helvetica").with_image(img);
        let dir = temp_dir("gallery");
        let mut conv = HybridConverter::new(ConverterConfig::default(), &dir);
        let mut pdf = FakePdf::new(vec![with_img]);
        let out = conv
            .convert_pdf_source(&mut pdf, &dir.join("work"), &ProgressHooks::default())
            .unwrap();
        assert!(out.contains("![](p0_img0.png)"), "{}", out);

        // a page already referencing an image must NOT get the gallery
        let md_page = FakePage::from_text("", "Helvetica")
            .with_markdown("![inline](x.png)")
            .with_image(image::DynamicImage::new_rgb8(4, 4));
        let dir2 = temp_dir("gallery2");
        let mut pdf2 = FakePdf::new(vec![md_page]);
        let out2 = conv
            .convert_pdf_source(&mut pdf2, &dir2.join("work"), &ProgressHooks::default())
            .unwrap();
        assert!(!out2.contains("p0_img"), "{}", out2);
    }

    #[test]
    fn surgical_without_math_returns_fast_markdown() {
        let mut pdf =
            FakePdf::new(vec![FakePage::from_text("plain body prose only", "Helvetica")]);
        let mut conv = HybridConverter::new(
            ConverterConfig { routing_mode: RoutingMode::Surgical, ..Default::default() },
            &temp_dir("surgical_plain"),
        );
        let out = conv
            .convert_pdf_source(&mut pdf, Path::new("/tmp/nowhere"), &ProgressHooks::default())
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
            .convert_pdf_source(&mut pdf, Path::new("/tmp/nowhere"), &ProgressHooks::default())
            .unwrap();
        assert!(out.contains("regular sentences"), "{}", out);
    }

    #[test]
    fn fast_path_falls_back_to_char_join_when_md_errors() {
        let mut page = FakePage::from_text("fallback chars", "Helvetica");
        page.md = None; // simulate to_markdown failure
        let mut pdf = FakePdf::new(vec![page]);
        let mut conv = HybridConverter::new(
            ConverterConfig { routing_mode: RoutingMode::Never, ..Default::default() },
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
            chars.push(SourceChar::new('y', 12.0 + i as f32 * 8.0, 112.0, 7.0, 11.0, "msam"));
        }
        // body-font chars elsewhere are ignored entirely
        for i in 0..20 {
            chars.push(SourceChar::new('a', 10.0 + i as f32 * 8.0, 400.0, 7.0, 11.0, "Helvetica"));
        }
        let mut fake = FakePdf::default();
        fake.pages[0].chars = chars;

        let boxes = math_boxes_from_chars(&mut fake, 0, 5);
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        let b = boxes[0];
        assert!(b.y <= 100.0 && b.y + b.height >= 123.0, "{b:?}");

        // raising min_chars above both line sizes filters everything
        let none = math_boxes_from_chars(&mut fake, 0, 11);
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
        assert!(is_math_unicode('∀'));  // U+2200
        assert!(is_math_unicode('∫'));  // U+222B
        assert!(is_math_unicode('∑'));  // U+2211
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
        let cfg_always =
            ConverterConfig { routing_mode: RoutingMode::Always, ..Default::default() };
        assert!(needs_onnx(&mut textual, 0, &cfg_always));
        let cfg_never =
            ConverterConfig { routing_mode: RoutingMode::Never, ..Default::default() };
        assert!(!needs_onnx(&mut scanned, 0, &cfg_never));
    }

    #[test]
    fn wrap_code_blocks_emits_fence_for_mono_lines() {
        let mut page = FakePage::from_text("", "Helvetica");
        page.chars.clear();
        for (i, ch) in "let x = 42;".chars().enumerate() {
            page.chars
                .push(SourceChar::new(ch, 50.0 + i as f32 * 8.0, 300.0, 7.0, 11.0, "Consolas"));
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
}
