// PdfSource — the seam between bobine's conversion logic and pdf_oxide.
//
// All converter internals operate on `&mut dyn PdfSource`, so the routing /
// splicing / gallery logic can be exercised against tiny in-memory fakes
// instead of real PDF files (mirrors legacy Python's conftest.py fakes).
//
// The trait deliberately uses bobine-owned types (`SourceChar`, PNG bytes,
// `DynamicImage`) rather than pdf_oxide's — fakes stay decoupled from
// upstream struct churn.

use pdf_oxide::{
    api::Pdf,
    geometry::Rect,
    layout::{RectFilterMode, TextChar},
    rendering::RenderOptions,
};

use crate::error::{BobineError, Result};

/// Minimal per-character info the converter actually consumes.
///
/// (Legacy used pdf_oxide's full `TextChar`; only `char`, `bbox` and
/// `font_name` ever mattered.)
#[derive(Debug, Clone)]
pub struct SourceChar {
    pub char: char,
    pub bbox: Rect,
    pub font_name: String,
}

impl SourceChar {
    pub fn new(ch: char, x: f32, y: f32, width: f32, height: f32, font_name: &str) -> Self {
        Self {
            char: ch,
            bbox: Rect::new(x, y, width, height),
            font_name: font_name.to_string(),
        }
    }
}

/// Bobine-owned mirror of one structured table cell.
///
/// Same decoupling rationale as [`SourceChar`]: pdf_oxide's table types
/// churn upstream, and the fakes must not depend on them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceTableCell {
    pub text: String,
    pub colspan: u32,
    pub rowspan: u32,
}

/// Bobine-owned mirror of one structured table row.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceTableRow {
    pub cells: Vec<SourceTableCell>,
    pub is_header: bool,
}

/// Bobine-owned mirror of a structured table extracted from the text layer
/// (Tagged-PDF structure tree or ruled-grid spatial detection).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceTable {
    pub rows: Vec<SourceTableRow>,
    pub has_header: bool,
    pub col_count: usize,
    /// Bounding box in PDF points, when the detector could localize it.
    pub bbox: Option<Rect>,
}

/// Placement info for one embedded raster image of a page.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SourceImage {
    /// Bounding box in PDF points (None when the source cannot localize it —
    /// e.g. images drawn through exotic XObject chains).
    pub bbox: Option<Rect>,
    /// Pixel dimensions of the stored bitmap.
    pub width: u32,
    pub height: u32,
}

/// Abstract view of an opened PDF document.
pub trait PdfSource {
    fn page_count(&mut self) -> Result<usize>;
    /// Lossless text-layer markdown for a page (fast path).
    fn page_markdown(&mut self, index: usize) -> Result<String>;
    /// Plain text extracted from a rectangle (PDF points, top-left origin).
    fn text_in_rect(&mut self, index: usize, rect: Rect) -> Result<String>;
    fn chars(&mut self, index: usize) -> Result<Vec<SourceChar>>;
    /// Number of embedded raster images on a page.
    fn image_count(&mut self, index: usize) -> Result<usize>;
    /// Save embedded raster images of a page into `dir` as
    /// `<prefix><n>.png`; returns the written paths.
    fn extract_image_files(
        &mut self,
        index: usize,
        dir: &std::path::Path,
        prefix: &str,
    ) -> Result<Vec<std::path::PathBuf>>;
    /// Rasterize a page at the requested DPI; returns PNG bytes.
    fn render_png(&mut self, index: usize, dpi: u32) -> Result<Vec<u8>>;
    /// `[x0, y0, width, height]` of the page in points.
    fn media_box(&mut self, index: usize) -> Result<[f32; 4]>;

    /// Structured tables whose bbox intersects `rect` (PDF points), drawn
    /// from the Tagged-PDF structure tree or ruled-grid spatial detection.
    /// Default: none (sources without grid detection). Used by the
    /// born-digital table cascade in the full-structure path.
    fn tables_in_rect(&mut self, _index: usize, _rect: Rect) -> Result<Vec<SourceTable>> {
        Ok(Vec::new())
    }

    /// Placement info for the page's embedded raster images, in extraction
    /// order (the same order `extract_image_files` writes files in).
    /// Default: none.
    fn images(&mut self, _index: usize) -> Result<Vec<SourceImage>> {
        Ok(Vec::new())
    }
}

impl PdfSource for Pdf {
    fn page_count(&mut self) -> Result<usize> {
        self.page_count()
            .map_err(|e| BobineError::PdfOxide(format!("page_count: {e}")))
    }

    fn page_markdown(&mut self, index: usize) -> Result<String> {
        self.to_markdown(index)
            .map_err(|e| BobineError::PdfOxide(format!("to_markdown: {e}")))
    }

    fn text_in_rect(&mut self, index: usize, rect: Rect) -> Result<String> {
        self.extract_text_in_rect(index, rect, RectFilterMode::Intersects)
            .map_err(|e| BobineError::PdfOxide(format!("extract_text_in_rect: {e}")))
    }

    fn chars(&mut self, index: usize) -> Result<Vec<SourceChar>> {
        let chars: Vec<TextChar> = self
            .extract_chars(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_chars: {e}")))?;
        Ok(chars
            .into_iter()
            .map(|c| SourceChar {
                char: c.char,
                bbox: c.bbox,
                font_name: c.font_name,
            })
            .collect())
    }

    fn image_count(&mut self, index: usize) -> Result<usize> {
        Ok(self
            .extract_images(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))?
            .len())
    }

    fn extract_image_files(
        &mut self,
        index: usize,
        dir: &std::path::Path,
        prefix: &str,
    ) -> Result<Vec<std::path::PathBuf>> {
        let objs = self
            .extract_images(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))?;
        std::fs::create_dir_all(dir)
            .map_err(|e| BobineError::Io(e))?;
        let mut out = Vec::with_capacity(objs.len());
        for (n, obj) in objs.iter().enumerate() {
            // Preserve original bytes for JPEG-encoded images (the dominant
            // embed type): no recompression, smaller files. Everything else
            // is transcoded to lossless PNG.
            let (p, res) = match obj.data() {
                pdf_oxide::extractors::ImageData::Jpeg(bytes) => {
                    let p = dir.join(format!("{prefix}{n}.jpg"));
                    let res = std::fs::write(&p, bytes).map_err(BobineError::Io);
                    (p, res)
                }
                _ => {
                    let p = dir.join(format!("{prefix}{n}.png"));
                    let res = obj
                        .save_as_png(&p)
                        .map_err(|e| BobineError::PdfOxide(format!("save image: {e}")));
                    (p, res)
                }
            };
            res?;
            out.push(p);
        }
        Ok(out)
    }

    fn render_png(&mut self, index: usize, dpi: u32) -> Result<Vec<u8>> {
        let opts = RenderOptions::with_dpi(dpi);
        let rendered = self
            .render_page(index, Some(&opts))
            .map_err(|e| BobineError::PdfOxide(format!("render: {e}")))?;
        Ok(rendered.data)
    }

    fn media_box(&mut self, index: usize) -> Result<[f32; 4]> {
        self.page_media_box(index)
            .map_err(|e| BobineError::PdfOxide(format!("media_box: {e}")))
    }

    fn tables_in_rect(&mut self, index: usize, rect: Rect) -> Result<Vec<SourceTable>> {
        // Whole-page extraction with the balanced default config, then scope
        // to the requested rect by bbox. (extract_tables_in_rect would apply
        // the relaxed text-only strategy, which misses ruled grids.)
        let tables = self
            .extract_tables(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_tables: {e}")))?;
        Ok(tables
            .into_iter()
            .filter(|t| t.bbox.map_or(false, |b| b.intersects(&rect)))
            .map(|t| SourceTable {
                has_header: t.has_header,
                col_count: t.col_count,
                bbox: t.bbox,
                rows: t
                    .rows
                    .into_iter()
                    .map(|r| SourceTableRow {
                        is_header: r.is_header,
                        cells: r
                            .cells
                            .into_iter()
                            .map(|c| SourceTableCell {
                                text: c.text,
                                colspan: c.colspan,
                                rowspan: c.rowspan,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect())
    }

    fn images(&mut self, index: usize) -> Result<Vec<SourceImage>> {
        let objs = self
            .extract_images(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))?;
        Ok(objs
            .iter()
            .map(|o| SourceImage {
                bbox: o.bbox().cloned(),
                width: o.width(),
                height: o.height(),
            })
            .collect())
    }
}

// ======================================================================
// Test fakes
// ======================================================================

#[cfg(test)]
pub(crate) mod fake {
    use super::*;

    /// One synthetic page: canned markdown, chars, images and region texts.
    #[derive(Default, Clone)]
    pub(crate) struct FakePage {
        pub md: Option<String>,
        pub chars: Vec<SourceChar>,
        pub images: Vec<image::DynamicImage>,
        /// Placement bboxes (PDF points) aligned with `images`.
        pub image_bboxes: Vec<Option<Rect>>,
        pub regions: Vec<(Rect, String)>,
        /// Structured tables served by `tables_in_rect` when their bbox
        /// intersects the query rect.
        pub tables: Vec<(Rect, SourceTable)>,
        pub media_box: [f32; 4],
        /// PNG bytes returned by `render_png` (a tiny valid PNG by default).
        pub rendered: Vec<u8>,
    }

    const TINY_PNG: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x62, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    impl FakePage {
        /// A page whose chars mirror `text`, laid out left→right at y=700.
        pub fn from_text(text: &str, font: &str) -> Self {
            let mut page = Self::default();
            page.media_box = [0.0, 0.0, 612.0, 792.0];
            page.md = Some(text.to_string());
            for (i, ch) in text.chars().enumerate() {
                page.chars.push(SourceChar::new(
                    ch,
                    50.0 + i as f32 * 8.0,
                    700.0,
                    7.0,
                    11.0,
                    font,
                ));
            }
            page.rendered = TINY_PNG.to_vec();
            page
        }

        pub fn with_image(mut self, img: image::DynamicImage) -> Self {
            self.images.push(img);
            self.image_bboxes.push(None);
            self
        }

        /// Embedded image with a known placement bbox (PDF points).
        pub fn with_image_at(mut self, img: image::DynamicImage, rect: Rect) -> Self {
            self.images.push(img);
            self.image_bboxes.push(Some(rect));
            self
        }

        pub fn with_region(mut self, rect: Rect, text: &str) -> Self {
            self.regions.push((rect, text.to_string()));
            self
        }

        pub fn with_table(mut self, rect: Rect, table: SourceTable) -> Self {
            self.tables.push((rect, table));
            self
        }

        pub fn with_markdown(mut self, md: &str) -> Self {
            self.md = Some(md.to_string());
            self
        }
    }

    /// A synthetic multi-page document.
    pub(crate) struct FakePdf {
        pub pages: Vec<FakePage>,
    }

    impl FakePdf {
        pub fn new(pages: Vec<FakePage>) -> Self {
            Self { pages }
        }
    }

    impl Default for FakePdf {
        fn default() -> Self {
            Self::new(vec![FakePage::from_text("Hello world", "Helvetica")])
        }
    }

    impl PdfSource for FakePdf {
        fn page_count(&mut self) -> Result<usize> {
            Ok(self.pages.len())
        }

        fn page_markdown(&mut self, index: usize) -> Result<String> {
            self.pages[index]
                .md
                .clone()
                .ok_or_else(|| BobineError::PdfOxide("no markdown".into()))
        }

        fn text_in_rect(&mut self, index: usize, rect: Rect) -> Result<String> {
            // Intersects semantics over the registered regions; fall back to
            // joining chars whose bbox intersects.
            for (r, text) in self.pages[index].regions.iter().rev() {
                let hit = !(rect.x > r.x + r.width
                    || rect.x + rect.width < r.x
                    || rect.y > r.y + r.height
                    || rect.y + rect.height < r.y);
                if hit {
                    return Ok(text.clone());
                }
            }
            Ok(self.pages[index]
                .chars
                .iter()
                .filter(|c| {
                    !(rect.x > c.bbox.x + c.bbox.width
                        || rect.x + rect.width < c.bbox.x
                        || rect.y > c.bbox.y + c.bbox.height
                        || rect.y + rect.height < c.bbox.y)
                })
                .map(|c| c.char)
                .collect())
        }

        fn chars(&mut self, index: usize) -> Result<Vec<SourceChar>> {
            Ok(self.pages[index].chars.clone())
        }

        fn image_count(&mut self, index: usize) -> Result<usize> {
            Ok(self.pages[index].images.len())
        }

        fn extract_image_files(
            &mut self,
            index: usize,
            dir: &std::path::Path,
            prefix: &str,
        ) -> Result<Vec<std::path::PathBuf>> {
            std::fs::create_dir_all(dir).map_err(|e| BobineError::Other(format!("fake mkdir: {e}")))?;
            let mut out = Vec::new();
            for (n, img) in self.pages[index].images.iter().enumerate() {
                let p = dir.join(format!("{prefix}{n}.png"));
                img.save_with_format(&p, image::ImageFormat::Png)
                    .map_err(|e| BobineError::Other(format!("fake image save: {e}")))?;
                out.push(p);
            }
            Ok(out)
        }

        fn render_png(&mut self, index: usize, _dpi: u32) -> Result<Vec<u8>> {
            Ok(self.pages[index].rendered.clone())
        }

        fn media_box(&mut self, index: usize) -> Result<[f32; 4]> {
            Ok(self.pages[index].media_box)
        }

        fn images(&mut self, index: usize) -> Result<Vec<SourceImage>> {
            Ok(self.pages[index]
                .images
                .iter()
                .zip(self.pages[index].image_bboxes.iter())
                .map(|(img, bbox)| SourceImage {
                    bbox: *bbox,
                    width: img.width(),
                    height: img.height(),
                })
                .collect())
        }

        fn tables_in_rect(&mut self, index: usize, rect: Rect) -> Result<Vec<SourceTable>> {
            let hit = |r: &Rect| {
                !(rect.x > r.x + r.width
                    || rect.x + rect.width < r.x
                    || rect.y > r.y + r.height
                    || rect.y + rect.height < r.y)
            };
            Ok(self.pages[index]
                .tables
                .iter()
                .filter(|(r, _)| hit(r))
                .map(|(_, t)| t.clone())
                .collect())
        }
    }
}
