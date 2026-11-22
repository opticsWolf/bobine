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
    document::PdfDocument,
    geometry::Rect,
    layout::{RectFilterMode, TextChar},
    rendering::RenderOptions,
};

use crate::error::{BobineError, Result};
use crate::math_source::{EmbeddedFile, MathOrigin, MathSpan};
use crate::pdf_math_raw::{self, AssociatedFileMetadata, FormulaMetadata};

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

    /// Structured math spans from tagged PDF semantics. Default: none.
    fn structured_math(&mut self, _index: usize) -> Result<Vec<MathSpan>> {
        Ok(Vec::new())
    }

    /// Variant used by the converter to cap its own traversal work. Custom
    /// sources can keep implementing `structured_math` unchanged.
    fn structured_math_with_limit(
        &mut self,
        index: usize,
        _max_nodes: usize,
    ) -> Result<Vec<MathSpan>> {
        self.structured_math(index)
    }

    /// Variant that also passes the formula-associated attachment parsing cap.
    fn structured_math_with_limits(
        &mut self,
        index: usize,
        max_nodes: usize,
        _max_formula_attachment_bytes: usize,
    ) -> Result<Vec<MathSpan>> {
        self.structured_math_with_limit(index, max_nodes)
    }

    /// Filenames associated with tagged Formula elements. Sources that do not
    /// expose raw formula associations return an empty set.
    fn associated_formula_files_with_limit(
        &mut self,
        _max_nodes: usize,
    ) -> Result<std::collections::HashSet<String>> {
        Ok(std::collections::HashSet::new())
    }

    /// Catalog-level PDF attachments. Default: none.
    fn embedded_files(&mut self) -> Result<Vec<EmbeddedFile>> {
        Ok(Vec::new())
    }
}

/// `pdf_oxide::document::PdfDocument` adapter used by Bobine's PDF pipeline.
/// It keeps one backend document open and caches its tagged structure tree so
/// per-page extraction does not reparse the whole tree.
pub struct PdfDocumentSource {
    document: PdfDocument,
    structure_tree:
        Option<std::result::Result<Option<pdf_oxide::structure::StructTreeRoot>, String>>,
    raw_formula_metadata: Option<(usize, std::result::Result<Vec<FormulaMetadata>, String>)>,
    embedded_files_cache: Option<std::result::Result<Vec<EmbeddedFile>, String>>,
}

impl PdfDocumentSource {
    /// Open one PDF through the same `pdf_oxide` document used for text,
    /// rendering, tagged structure, and embedded files.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        crate::input_limits::ensure_input_file_size(path.as_ref())?;
        let document =
            PdfDocument::open(path).map_err(|e| BobineError::PdfOxide(format!("open: {e}")))?;
        Ok(Self::from_document(document))
    }

    /// Wrap an already-open `pdf_oxide` document.
    pub fn from_document(document: PdfDocument) -> Self {
        Self {
            document,
            structure_tree: None,
            raw_formula_metadata: None,
            embedded_files_cache: None,
        }
    }

    fn formula_metadata_with_limit(
        &mut self,
        tree: &pdf_oxide::structure::StructTreeRoot,
        max_nodes: usize,
    ) -> std::result::Result<Vec<FormulaMetadata>, String> {
        if self
            .raw_formula_metadata
            .as_ref()
            .is_none_or(|(cached_limit, _)| *cached_limit != max_nodes)
        {
            self.raw_formula_metadata = Some((
                max_nodes,
                pdf_math_raw::aligned_formula_metadata(&self.document, tree, max_nodes),
            ));
        }
        self.raw_formula_metadata
            .as_ref()
            .map(|(_, result)| result.clone())
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    fn embedded_files_cached(&mut self) -> Result<Vec<EmbeddedFile>> {
        self.ensure_embedded_files_cached();
        match self.embedded_files_cache.as_ref() {
            Some(Ok(files)) => Ok(files.clone()),
            Some(Err(error)) => Err(BobineError::PdfOxide(format!(
                "extract_embedded_files: {error}"
            ))),
            None => Ok(Vec::new()),
        }
    }

    fn ensure_embedded_files_cached(&mut self) {
        if self.embedded_files_cache.is_none() {
            // TODO(resource-limits): pdf_oxide 0.3.78 eagerly decodes every
            // name-tree attachment here and returns all decoded Vecs at once.
            // Bobine's per-file parsing limits run afterward; there is no
            // aggregate byte/count limit or peak decompression bound. Replace
            // this with a bounded pdf_oxide extraction API when available.
            let filename_aliases = pdf_math_raw::embedded_filename_aliases(&self.document);
            self.embedded_files_cache = Some(
                self.document
                    .extract_embedded_files()
                    .map(|files| {
                        files
                            .into_iter()
                            .map(|(filename, bytes)| {
                                let filename =
                                    filename_aliases.get(&filename).cloned().unwrap_or_else(|| {
                                        pdf_math_raw::decode_embedded_filename_from_pdf_oxide(
                                            &filename,
                                        )
                                    });
                                EmbeddedFile {
                                    filename,
                                    bytes,
                                    mime_type: None,
                                    relationship: None,
                                }
                            })
                            .collect()
                    })
                    .map_err(|error| error.to_string()),
            );
        }
    }

    fn associated_formula_files_with_limit(
        &mut self,
        max_nodes: usize,
    ) -> Result<std::collections::HashSet<String>> {
        if max_nodes == 0 {
            return Ok(std::collections::HashSet::new());
        }
        if self.structure_tree.is_none() {
            self.structure_tree = Some(
                self.document
                    .structure_tree_with_budget(Some(std::time::Duration::from_secs(2)))
                    .map_err(|error| error.to_string()),
            );
        }
        let tree = match self.structure_tree.as_ref() {
            Some(Ok(Some(tree))) => tree.clone(),
            Some(Ok(None)) => return Ok(std::collections::HashSet::new()),
            Some(Err(error)) => {
                return Err(BobineError::PdfOxide(format!("structure_tree: {error}")));
            }
            None => return Ok(std::collections::HashSet::new()),
        };
        let metadata = self
            .formula_metadata_with_limit(&tree, max_nodes)
            .map_err(|error| BobineError::PdfOxide(format!("formula metadata: {error}")))?;
        Ok(metadata
            .into_iter()
            .flat_map(|item| item.associated_files)
            .filter_map(|file| file.filename)
            .collect())
    }
}

impl PdfSource for PdfDocumentSource {
    fn page_count(&mut self) -> Result<usize> {
        self.document
            .page_count()
            .map_err(|e| BobineError::PdfOxide(format!("page_count: {e}")))
    }

    fn page_markdown(&mut self, index: usize) -> Result<String> {
        self.document
            .to_markdown(index, &pdf_oxide::converters::ConversionOptions::default())
            .map_err(|e| BobineError::PdfOxide(format!("to_markdown: {e}")))
    }

    fn text_in_rect(&mut self, index: usize, rect: Rect) -> Result<String> {
        self.document
            .extract_text_in_rect(index, rect, RectFilterMode::Intersects)
            .map_err(|e| BobineError::PdfOxide(format!("extract_text_in_rect: {e}")))
    }

    fn chars(&mut self, index: usize) -> Result<Vec<SourceChar>> {
        self.document
            .extract_chars(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_chars: {e}")))
            .map(|chars| {
                chars
                    .into_iter()
                    .map(|c| SourceChar {
                        char: c.char,
                        bbox: c.bbox,
                        font_name: c.font_name,
                    })
                    .collect()
            })
    }

    fn image_count(&mut self, index: usize) -> Result<usize> {
        self.document
            .extract_images(index)
            .map(|images| images.len())
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))
    }

    fn extract_image_files(
        &mut self,
        index: usize,
        dir: &std::path::Path,
        prefix: &str,
    ) -> Result<Vec<std::path::PathBuf>> {
        let images = self
            .document
            .extract_images(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))?;
        std::fs::create_dir_all(dir).map_err(BobineError::Io)?;
        let mut out = Vec::with_capacity(images.len());
        for (n, image) in images.iter().enumerate() {
            let (path, result) = match image.data() {
                pdf_oxide::extractors::ImageData::Jpeg(bytes) => {
                    let path = dir.join(format!("{prefix}{n}.jpg"));
                    let result = std::fs::write(&path, bytes).map_err(BobineError::Io);
                    (path, result)
                }
                _ => {
                    let path = dir.join(format!("{prefix}{n}.png"));
                    let result = image
                        .save_as_png(&path)
                        .map_err(|e| BobineError::PdfOxide(format!("save image: {e}")));
                    (path, result)
                }
            };
            result?;
            out.push(path);
        }
        Ok(out)
    }

    fn render_png(&mut self, index: usize, dpi: u32) -> Result<Vec<u8>> {
        pdf_oxide::rendering::render_page(&self.document, index, &RenderOptions::with_dpi(dpi))
            .map(|rendered| rendered.data)
            .map_err(|e| BobineError::PdfOxide(format!("render: {e}")))
    }

    fn media_box(&mut self, index: usize) -> Result<[f32; 4]> {
        self.document
            .get_page_media_box(index)
            .map(|(x0, y0, x1, y1)| [x0, y0, x1, y1])
            .map_err(|e| BobineError::PdfOxide(format!("media_box: {e}")))
    }

    fn tables_in_rect(&mut self, index: usize, rect: Rect) -> Result<Vec<SourceTable>> {
        let tables = self
            .document
            .extract_tables(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_tables: {e}")))?;
        Ok(tables
            .into_iter()
            .filter(|table| table.bbox.map_or(false, |bbox| bbox.intersects(&rect)))
            .map(|table| SourceTable {
                has_header: table.has_header,
                col_count: table.col_count,
                bbox: table.bbox,
                rows: table
                    .rows
                    .into_iter()
                    .map(|row| SourceTableRow {
                        is_header: row.is_header,
                        cells: row
                            .cells
                            .into_iter()
                            .map(|cell| SourceTableCell {
                                text: cell.text,
                                colspan: cell.colspan,
                                rowspan: cell.rowspan,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect())
    }

    fn images(&mut self, index: usize) -> Result<Vec<SourceImage>> {
        self.document
            .extract_images(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_images: {e}")))
            .map(|images| {
                images
                    .iter()
                    .map(|image| SourceImage {
                        bbox: image.bbox().cloned(),
                        width: image.width(),
                        height: image.height(),
                    })
                    .collect()
            })
    }

    fn structured_math(&mut self, index: usize) -> Result<Vec<MathSpan>> {
        self.structured_math_with_limits(index, 10_000, 1024 * 1024)
    }

    fn structured_math_with_limit(
        &mut self,
        index: usize,
        max_nodes: usize,
    ) -> Result<Vec<MathSpan>> {
        self.structured_math_with_limits(index, max_nodes, 1024 * 1024)
    }

    fn structured_math_with_limits(
        &mut self,
        index: usize,
        max_nodes: usize,
        max_formula_attachment_bytes: usize,
    ) -> Result<Vec<MathSpan>> {
        if max_nodes == 0 {
            return Ok(Vec::new());
        }
        if self.structure_tree.is_none() {
            self.structure_tree = Some(
                self.document
                    .structure_tree_with_budget(Some(std::time::Duration::from_secs(2)))
                    .map_err(|e| e.to_string()),
            );
        }
        let tree = match self.structure_tree.as_ref() {
            Some(Ok(Some(tree))) => tree.clone(),
            Some(Ok(None)) => return Ok(Vec::new()),
            Some(Err(error)) => {
                return Err(BobineError::PdfOxide(format!("structure_tree: {error}")));
            }
            None => return Ok(Vec::new()),
        };
        let raw_metadata = self.formula_metadata_with_limit(&tree, max_nodes);
        let (raw_metadata, raw_warning) = match raw_metadata {
            Ok(metadata) => (Some(metadata), None),
            Err(error) => (None, Some(format!("raw formula metadata skipped: {error}"))),
        };
        let attachment_needed = raw_metadata
            .as_ref()
            .is_some_and(|items| items.iter().any(|item| !item.associated_files.is_empty()));
        let attachment_warning = if attachment_needed {
            self.ensure_embedded_files_cached();
            self.embedded_files_cache
                .as_ref()
                .and_then(|result| result.as_ref().err())
                .map(|error| format!("associated-file extraction skipped: {error}"))
        } else {
            None
        };
        let embedded_files = match self.embedded_files_cache.as_ref() {
            Some(Ok(files)) if attachment_needed => files.as_slice(),
            _ => &[],
        };
        let spans = self
            .document
            .extract_spans(index)
            .map_err(|e| BobineError::PdfOxide(format!("extract_spans: {e}")))?;
        let mut formulas = Vec::new();
        let mut visited = 0usize;
        let mut formula_index = 0usize;
        for root in &tree.root_elements {
            collect_formula_spans(
                root,
                None,
                index,
                &spans,
                &mut visited,
                max_nodes,
                &mut formula_index,
                raw_metadata.as_deref(),
                &embedded_files,
                max_formula_attachment_bytes,
                raw_warning.as_deref(),
                attachment_warning.as_deref(),
                &mut formulas,
            );
            if visited >= max_nodes {
                break;
            }
        }
        Ok(formulas)
    }

    fn embedded_files(&mut self) -> Result<Vec<EmbeddedFile>> {
        self.embedded_files_cached()
    }

    fn associated_formula_files_with_limit(
        &mut self,
        max_nodes: usize,
    ) -> Result<std::collections::HashSet<String>> {
        PdfDocumentSource::associated_formula_files_with_limit(self, max_nodes)
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
        std::fs::create_dir_all(dir).map_err(|e| BobineError::Io(e))?;
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

fn collect_formula_spans(
    elem: &pdf_oxide::structure::StructElem,
    inherited_page: Option<u32>,
    target_page: usize,
    text_spans: &[pdf_oxide::layout::TextSpan],
    visited: &mut usize,
    max_nodes: usize,
    formula_index: &mut usize,
    raw_metadata: Option<&[FormulaMetadata]>,
    embedded_files: &[EmbeddedFile],
    max_formula_attachment_bytes: usize,
    raw_warning: Option<&str>,
    attachment_warning: Option<&str>,
    out: &mut Vec<MathSpan>,
) {
    if *visited >= max_nodes {
        return;
    }
    *visited += 1;
    let page = elem.page.or(inherited_page);
    if elem.struct_type == pdf_oxide::structure::StructType::Formula {
        let this_formula_index = *formula_index;
        *formula_index += 1;
        let formula_page = page.or_else(|| first_marked_page(elem));
        if formula_page == Some(target_page as u32) {
            let mut bbox = None;
            let xml = if elem.actual_text.as_deref().is_some_and(looks_like_mathml) {
                collect_formula_bbox(elem, target_page, text_spans, &mut bbox);
                elem.actual_text.clone().unwrap_or_default()
            } else {
                let mut content = String::new();
                serialize_formula_children(
                    elem,
                    target_page,
                    text_spans,
                    &mut bbox,
                    &mut content,
                    visited,
                    max_nodes,
                );
                format!("<math xmlns=\"http://www.w3.org/1998/Math/MathML\">{content}</math>")
            };
            let metadata = raw_metadata.and_then(|items| items.get(this_formula_index));
            let description = metadata
                .and_then(|metadata| metadata.description.clone())
                .or_else(|| elem.alt_text.clone());
            let mut warnings = Vec::new();
            if let Some(warning) = raw_warning {
                warnings.push(warning.to_string());
            }
            if let Some(warning) = attachment_warning {
                warnings.push(warning.to_string());
            }
            if raw_metadata.is_some() && metadata.is_none() {
                warnings.push(format!(
                    "raw metadata has no aligned entry for Formula {}",
                    this_formula_index
                ));
            }
            if let Some(metadata) = metadata {
                warnings.extend(metadata.warnings.iter().cloned());
            }
            let structure_namespace = metadata.and_then(|metadata| metadata.namespace.clone());
            let structure_attributes = metadata.and_then(|metadata| metadata.attributes.clone());
            match crate::math_source::mathml_to_latex(&xml) {
                Ok(latex) => out.push(MathSpan {
                    page: target_page,
                    bbox,
                    latex,
                    display: true,
                    origin: MathOrigin::TaggedMathMl,
                    confidence: 0.98,
                    tag: Some("Formula".to_string()),
                    warnings,
                    description,
                    structure_namespace,
                    structure_attributes,
                    associated_filename: None,
                    associated_relationship: None,
                    associated_mime_type: None,
                }),
                Err(structure_error) => {
                    warnings.push(structure_error.clone());
                    let associated = metadata.and_then(|metadata| {
                        extract_associated_mathml(
                            metadata,
                            embedded_files,
                            max_formula_attachment_bytes,
                        )
                        .map_err(|errors| warnings.extend(errors))
                        .ok()
                    });
                    if let Some((latex, file)) = associated {
                        out.push(MathSpan {
                            page: target_page,
                            bbox,
                            latex,
                            display: true,
                            origin: MathOrigin::AssociatedFile,
                            confidence: 0.95,
                            tag: Some("Formula".to_string()),
                            warnings,
                            description,
                            structure_namespace,
                            structure_attributes,
                            associated_filename: file.filename.clone(),
                            associated_relationship: file.relationship.clone(),
                            associated_mime_type: file.mime_type.clone(),
                        });
                    } else if description.is_some() {
                        out.push(MathSpan {
                            page: target_page,
                            bbox,
                            latex: String::new(),
                            display: true,
                            origin: MathOrigin::AltDescription,
                            confidence: 0.0,
                            tag: Some("Formula".to_string()),
                            warnings,
                            description,
                            structure_namespace,
                            structure_attributes,
                            associated_filename: None,
                            associated_relationship: None,
                            associated_mime_type: None,
                        });
                    } else {
                        out.push(MathSpan {
                            page: target_page,
                            bbox,
                            latex: String::new(),
                            display: true,
                            origin: MathOrigin::TaggedMathMl,
                            confidence: 0.0,
                            tag: Some("Formula".to_string()),
                            warnings,
                            description,
                            structure_namespace,
                            structure_attributes,
                            associated_filename: None,
                            associated_relationship: None,
                            associated_mime_type: None,
                        });
                    }
                }
            }
            // A nested Formula is represented by its enclosing formula's
            // MathML; visiting it again would emit the same content twice.
            *formula_index += count_nested_formula_nodes(elem);
            return;
        }
    }

    for child in &elem.children {
        if *visited >= max_nodes {
            break;
        }
        if let pdf_oxide::structure::StructChild::StructElem(child) = child {
            collect_formula_spans(
                child,
                page,
                target_page,
                text_spans,
                visited,
                max_nodes,
                formula_index,
                raw_metadata,
                embedded_files,
                max_formula_attachment_bytes,
                raw_warning,
                attachment_warning,
                out,
            );
        }
    }
}

fn first_marked_page(elem: &pdf_oxide::structure::StructElem) -> Option<u32> {
    for child in &elem.children {
        match child {
            pdf_oxide::structure::StructChild::MarkedContentRef { page, .. } => return Some(*page),
            pdf_oxide::structure::StructChild::StructElem(child) => {
                if let Some(page) = first_marked_page(child) {
                    return Some(page);
                }
            }
            pdf_oxide::structure::StructChild::ObjectRef(_, _) => {}
        }
    }
    None
}

fn count_nested_formula_nodes(elem: &pdf_oxide::structure::StructElem) -> usize {
    elem.children
        .iter()
        .filter_map(|child| match child {
            pdf_oxide::structure::StructChild::StructElem(child) => Some(
                usize::from(child.struct_type == pdf_oxide::structure::StructType::Formula)
                    + count_nested_formula_nodes(child),
            ),
            _ => None,
        })
        .sum()
}

fn extract_associated_mathml(
    metadata: &FormulaMetadata,
    embedded_files: &[EmbeddedFile],
    max_formula_attachment_bytes: usize,
) -> std::result::Result<(String, AssociatedFileMetadata), Vec<String>> {
    let mut warnings = Vec::new();
    for associated in &metadata.associated_files {
        let Some(filename) = associated.filename.as_deref() else {
            warnings.push("formula /AF filespec has no readable filename".to_string());
            continue;
        };
        let matches: Vec<_> = embedded_files
            .iter()
            .filter(|file| file.filename == filename)
            .collect();
        let file = match matches.as_slice() {
            [] => {
                warnings.push(format!(
                    "formula /AF file {filename} is not registered in /Names /EmbeddedFiles"
                ));
                continue;
            }
            [file] => *file,
            _ => {
                warnings.push(format!(
                    "formula /AF filename {filename} is ambiguous in /Names /EmbeddedFiles"
                ));
                continue;
            }
        };
        if file.bytes.len() > max_formula_attachment_bytes {
            warnings.push(format!(
                "formula /AF file {filename} is {} bytes and exceeds max_formula_attachment_bytes ({max_formula_attachment_bytes})",
                file.bytes.len(),
            ));
            continue;
        }
        let xml = match std::str::from_utf8(&file.bytes) {
            Ok(xml) => xml,
            Err(error) => {
                warnings.push(format!(
                    "formula /AF MathML {filename} is not UTF-8: {error}"
                ));
                continue;
            }
        };
        let parsed = match roxmltree::Document::parse(xml) {
            Ok(parsed) => parsed,
            Err(error) => {
                warnings.push(format!(
                    "formula /AF MathML {filename} parse failed: {error}"
                ));
                continue;
            }
        };
        let math_nodes: Vec<_> = parsed
            .descendants()
            .filter(|node| node.is_element() && node.tag_name().name() == "math")
            .collect();
        if math_nodes.len() != 1 {
            warnings.push(format!(
                "formula /AF file {filename} contains {} MathML <math> elements; expected one",
                math_nodes.len()
            ));
            continue;
        }
        match crate::math_source::math_node_to_latex(math_nodes[0]) {
            Ok(latex) => return Ok((latex, associated.clone())),
            Err(error) => warnings.push(format!("formula /AF MathML {filename}: {error}")),
        }
    }
    if warnings.is_empty() {
        warnings.push("formula has no usable /AF MathML filespec".to_string());
    }
    Err(warnings)
}

fn looks_like_mathml(text: &str) -> bool {
    text.trim_start().starts_with('<') && text.contains("math")
}

fn collect_formula_bbox(
    elem: &pdf_oxide::structure::StructElem,
    target_page: usize,
    text_spans: &[pdf_oxide::layout::TextSpan],
    bbox: &mut Option<Rect>,
) {
    for child in &elem.children {
        match child {
            pdf_oxide::structure::StructChild::MarkedContentRef {
                mcid, page, scope, ..
            } if *page as usize == target_page => {
                for text in matching_mcid_spans(*mcid, scope, text_spans) {
                    union_bbox(bbox, &text.bbox);
                }
            }
            pdf_oxide::structure::StructChild::StructElem(child) => {
                collect_formula_bbox(child, target_page, text_spans, bbox)
            }
            _ => {}
        }
    }
}

fn serialize_formula_children(
    elem: &pdf_oxide::structure::StructElem,
    target_page: usize,
    text_spans: &[pdf_oxide::layout::TextSpan],
    bbox: &mut Option<Rect>,
    xml: &mut String,
    visited: &mut usize,
    max_nodes: usize,
) {
    for child in &elem.children {
        if *visited >= max_nodes {
            break;
        }
        match child {
            pdf_oxide::structure::StructChild::MarkedContentRef {
                mcid, page, scope, ..
            } if *page as usize == target_page => {
                for text in matching_mcid_spans(*mcid, scope, text_spans) {
                    union_bbox(bbox, &text.bbox);
                    xml.push_str(&xml_escape(&text.text));
                }
            }
            pdf_oxide::structure::StructChild::StructElem(child) => {
                *visited += 1;
                let tag = mathml_tag(child);
                let actual_text = child.actual_text.as_deref();
                if let Some(tag) = tag {
                    if tag == "math" {
                        if let Some(actual) = actual_text {
                            xml.push_str(&xml_escape(actual));
                        } else {
                            serialize_formula_children(
                                child,
                                target_page,
                                text_spans,
                                bbox,
                                xml,
                                visited,
                                max_nodes,
                            );
                        }
                    } else {
                        xml.push('<');
                        xml.push_str(tag);
                        xml.push('>');
                        if let Some(actual) = actual_text {
                            xml.push_str(&xml_escape(actual));
                        } else {
                            serialize_formula_children(
                                child,
                                target_page,
                                text_spans,
                                bbox,
                                xml,
                                visited,
                                max_nodes,
                            );
                        }
                        xml.push_str("</");
                        xml.push_str(tag);
                        xml.push('>');
                    }
                } else if let Some(actual) = actual_text {
                    xml.push_str(&xml_escape(actual));
                } else {
                    serialize_formula_children(
                        child,
                        target_page,
                        text_spans,
                        bbox,
                        xml,
                        visited,
                        max_nodes,
                    );
                }
            }
            _ => {}
        }
    }
}

fn mathml_tag(elem: &pdf_oxide::structure::StructElem) -> Option<&'static str> {
    let role = elem
        .source_role
        .as_deref()
        .or_else(|| match &elem.struct_type {
            pdf_oxide::structure::StructType::Custom(name) => Some(name.as_str()),
            _ => None,
        })?;
    let local = role.rsplit(':').next().unwrap_or(role);
    Some(match local {
        "math" => "math",
        "mrow" => "mrow",
        "mi" => "mi",
        "mn" => "mn",
        "mo" => "mo",
        "mtext" => "mtext",
        "mspace" => "mspace",
        "mfrac" => "mfrac",
        "msqrt" => "msqrt",
        "mroot" => "mroot",
        "msup" => "msup",
        "msub" => "msub",
        "msubsup" => "msubsup",
        "mover" => "mover",
        "munder" => "munder",
        "munderover" => "munderover",
        "mfenced" => "mfenced",
        "mtable" => "mtable",
        "mtr" => "mtr",
        "mtd" => "mtd",
        "semantics" => "semantics",
        "annotation" => "annotation",
        "mstyle" => "mstyle",
        "menclose" => "menclose",
        "mphantom" => "mphantom",
        "mpadded" => "mpadded",
        "merror" => "merror",
        "mmultiscripts" => "mmultiscripts",
        "mprescripts" => "mprescripts",
        "none" => "none",
        "mglyph" => "mglyph",
        _ => return None,
    })
}

fn matching_mcid_spans<'a>(
    mcid: u32,
    scope: &pdf_oxide::structure::McidScope,
    text_spans: &'a [pdf_oxide::layout::TextSpan],
) -> Vec<&'a pdf_oxide::layout::TextSpan> {
    let same_mcid: Vec<_> = text_spans
        .iter()
        .filter(|span| span.mcid == Some(mcid))
        .collect();
    let mut matching: Vec<_> = same_mcid
        .into_iter()
        .filter(|span| match span.mcid_scope.as_ref() {
            Some(span_scope) => span_scope == scope,
            None => {
                text_spans
                    .iter()
                    .filter(|candidate| candidate.mcid == Some(mcid))
                    .count()
                    == 1
            }
        })
        .collect();
    matching.sort_by_key(|span| span.sequence);
    matching
}

fn union_bbox(bbox: &mut Option<Rect>, other: &Rect) {
    *bbox = Some(match *bbox {
        Some(current) => {
            let x0 = current.x.min(other.x);
            let y0 = current.y.min(other.y);
            let x1 = (current.x + current.width).max(other.x + other.width);
            let y1 = (current.y + current.height).max(other.y + other.height);
            Rect::new(x0, y0, x1 - x0, y1 - y0)
        }
        None => other.clone(),
    });
}

fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\u{9}' | '\u{a}' | '\u{d}' => out.push(ch),
            c if c >= '\u{20}' => out.push(c),
            _ => {}
        }
    }
    out
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
            std::fs::create_dir_all(dir)
                .map_err(|e| BobineError::Other(format!("fake mkdir: {e}")))?;
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

#[cfg(test)]
mod structured_math_tests {
    use super::*;
    use pdf_oxide::{
        document::PdfDocument,
        elements::{ContentElement, FontSpec, StructureElement, TextContent, TextStyle},
        geometry::Rect,
        writer::{PdfWriter, PdfWriterConfig},
    };

    fn math_tag(tag: &str, text: &str, x: f32) -> ContentElement {
        ContentElement::Structure(StructureElement {
            structure_type: tag.to_string(),
            bbox: Rect::new(x, 700.0, 12.0, 18.0),
            children: vec![ContentElement::Text(TextContent::new(
                text,
                Rect::new(x, 700.0, 12.0, 18.0),
                FontSpec::helvetica(16.0),
                TextStyle::default(),
            ))],
            reading_order: None,
            alt_text: None,
            language: None,
        })
    }

    /// Build a compact tagged fixture with pdf_oxide's writer. The Formula
    /// tree contains real MCIDs and MathML-role child elements.
    fn tagged_formula_fixture() -> Vec<u8> {
        let formula = ContentElement::Structure(StructureElement {
            structure_type: "Formula".to_string(),
            bbox: Rect::new(72.0, 700.0, 55.0, 18.0),
            children: vec![
                math_tag("mi", "x", 72.0),
                math_tag("mo", "+", 88.0),
                math_tag("mn", "1", 103.0),
            ],
            reading_order: None,
            alt_text: Some("x plus one".to_string()),
            language: None,
        });
        let mut writer = PdfWriter::with_config(PdfWriterConfig {
            tagged: true,
            ..PdfWriterConfig::default()
        });
        let mut page = writer.add_page(612.0, 792.0);
        page.add_element(&formula);
        page.finish();
        writer.finish().expect("write tagged formula fixture")
    }

    /// A small PDF object fixture for fields the public writer does not
    /// currently emit (`/AF` on a Formula and an EmbeddedFiles name tree).
    /// Parsing and stream decoding still run through pdf_oxide.
    fn formula_af_fixture(
        actual_text: &str,
        with_af: bool,
        alt: Option<&str>,
        formula_k: &str,
    ) -> Vec<u8> {
        use std::io::Write as _;

        let mathml = b"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>x</mi></math>";
        let filename = "formüla.xml";
        let utf16_name = filename
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .map(|byte| format!("{byte:02X}"))
            .collect::<String>();
        let actual_text_hex = actual_text
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .map(|byte| format!("{byte:02X}"))
            .collect::<String>();
        let associated = if with_af { " /AF [7 0 R]" } else { "" };
        let alt = alt.map(|alt| format!(" /Alt ({alt})")).unwrap_or_default();
        let mut stream = format!(
            "<< /Type /EmbeddedFile /Subtype /application#2Fmathml+xml /Length {} >>\nstream\n",
            mathml.len()
        )
        .into_bytes();
        stream.extend_from_slice(mathml);
        stream.extend_from_slice(b"\nendstream");
        let objects = vec![
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 5 0 R /Names 9 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >>".to_vec(),
            b"<< /Length 0 >>\nstream\n\nendstream".to_vec(),
            b"<< /Type /StructTreeRoot /K [6 0 R] >>".to_vec(),
            format!(
                "<< /Type /StructElem /S /Formula /P 5 0 R /Pg 3 0 R /K {formula_k} /ActualText <FEFF{actual_text_hex}>{alt} /NS << /Type /Namespace /NS (MathML) >> /A << /O /Layout /Placement /Block >>{associated} >>"
            )
            .into_bytes(),
            format!(
                "<< /Type /Filespec /F (formula.xml) /UF <FEFF{utf16_name}> /AFRelationship /Supplement /EF << /F 8 0 R >> >>"
            )
            .into_bytes(),
            stream,
            format!("<< /EmbeddedFiles << /Names [<FEFF{utf16_name}> 7 0 R] >> >>")
                .into_bytes(),
        ];

        let mut pdf = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n".to_vec();
        let mut offsets = vec![0usize];
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            write!(&mut pdf, "{} 0 obj\n", index + 1).unwrap();
            pdf.extend_from_slice(object);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref_offset = pdf.len();
        write!(&mut pdf, "xref\n0 {}\n", offsets.len()).unwrap();
        pdf.extend_from_slice(b"0000000000 65535 f \n");
        for offset in offsets.iter().skip(1) {
            write!(&mut pdf, "{offset:010} 00000 n \n").unwrap();
        }
        write!(
            &mut pdf,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            offsets.len()
        )
        .unwrap();
        pdf
    }

    #[test]
    fn tagged_formula_fixture_aligns_and_converts_mathml_roles() {
        let document = PdfDocument::from_bytes(tagged_formula_fixture()).unwrap();
        let tree = document.structure_tree().unwrap().expect("tagged tree");
        let raw = pdf_math_raw::aligned_formula_metadata(&document, &tree, 128)
            .expect("raw Formula metadata aligns with the parsed tree");
        assert_eq!(raw.len(), 1);

        let mut source = PdfDocumentSource::from_document(document);
        let formulas = source.structured_math(0).unwrap();
        assert_eq!(formulas.len(), 1);
        assert_eq!(formulas[0].origin, MathOrigin::TaggedMathMl);
        assert_eq!(formulas[0].latex, "x+1");
        assert_eq!(formulas[0].description.as_deref(), Some("x plus one"));
        assert!(formulas[0].bbox.is_some());
        assert!(
            formulas[0].warnings.is_empty(),
            "{:?}",
            formulas[0].warnings
        );
    }

    #[test]
    fn raw_formula_alignment_fails_closed_when_projection_differs() {
        let document = PdfDocument::from_bytes(tagged_formula_fixture()).unwrap();
        let mut tree = document.structure_tree().unwrap().expect("tagged tree");
        tree.root_elements.clear();
        let error = pdf_math_raw::aligned_formula_metadata(&document, &tree, 128)
            .expect_err("raw metadata must not align to a different projection");
        assert!(error.contains("Formula counts differ"), "{error}");
    }

    #[test]
    fn encrypted_formula_metadata_is_not_read_without_authentication() {
        use pdf_oxide::editor::{DocumentEditor, EncryptionConfig, SaveOptions};

        let mut editor = DocumentEditor::from_bytes(tagged_formula_fixture()).unwrap();
        let encrypted = editor
            .save_to_bytes_with_options(SaveOptions::with_encryption(EncryptionConfig::new(
                "user-password",
                "owner-password",
            )))
            .unwrap();
        let document = PdfDocument::from_bytes(encrypted).unwrap();
        assert!(!document.is_authenticated());

        let error = pdf_math_raw::aligned_formula_metadata(
            &document,
            &pdf_oxide::structure::StructTreeRoot::new(),
            128,
        )
        .expect_err("encrypted metadata must fail closed until authentication");
        assert!(error.contains("encrypted and not authenticated"), "{error}");
    }

    #[test]
    fn embedded_files_use_pdf_oxide_name_tree_extraction() {
        use pdf_oxide::{
            editor::{DocumentEditor, SaveOptions},
            writer::{AFRelationship, EmbeddedFile},
        };

        let mut editor = DocumentEditor::from_bytes(tagged_formula_fixture()).unwrap();
        let payload = b"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>x</mi></math>";
        editor
            .embed_file_with_options(
                EmbeddedFile::new("formula.xml", payload.to_vec())
                    .with_mime_type("application/mathml+xml")
                    .with_af_relationship(AFRelationship::Supplement),
            )
            .unwrap();
        let bytes = editor
            .save_to_bytes_with_options(SaveOptions::full_rewrite())
            .unwrap();
        let mut source = PdfDocumentSource::from_document(PdfDocument::from_bytes(bytes).unwrap());
        let files = source.embedded_files().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filename, "formula.xml");
        assert_eq!(files[0].bytes, payload);
        // The pinned pdf_oxide extraction API returns only names and bytes.
        assert_eq!(files[0].mime_type, None);
        assert_eq!(files[0].relationship, None);
    }

    #[test]
    fn formula_af_uses_matching_embedded_mathml_and_preserves_metadata() {
        let document =
            PdfDocument::from_bytes(formula_af_fixture("<math", true, None, "[]")).unwrap();
        let mut source = PdfDocumentSource::from_document(document);
        let formulas = source.structured_math(0).unwrap();
        assert_eq!(formulas.len(), 1);
        assert_eq!(
            formulas[0].origin,
            MathOrigin::AssociatedFile,
            "{formulas:#?}"
        );
        assert_eq!(formulas[0].latex, "x");
        assert_eq!(
            formulas[0].associated_filename.as_deref(),
            Some("formüla.xml")
        );
        assert_eq!(
            formulas[0].associated_relationship.as_deref(),
            Some("Supplement")
        );
        assert_eq!(
            formulas[0].associated_mime_type.as_deref(),
            Some("application/mathml+xml")
        );
        assert!(formulas[0].structure_namespace.is_some());
        assert!(formulas[0].structure_attributes.is_some());
    }

    #[test]
    fn structure_mathml_takes_precedence_over_formula_attachment() {
        let structure_mathml =
            "<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>y</mi></math>";
        let document =
            PdfDocument::from_bytes(formula_af_fixture(structure_mathml, true, None, "[]"))
                .unwrap();
        let mut source = PdfDocumentSource::from_document(document);
        let formulas = source.structured_math(0).unwrap();
        assert_eq!(formulas.len(), 1);
        assert_eq!(formulas[0].origin, MathOrigin::TaggedMathMl);
        assert_eq!(formulas[0].latex, "y");
        assert_eq!(formulas[0].associated_filename, None);
    }

    #[test]
    fn alt_only_formula_stays_a_description() {
        let document = PdfDocument::from_bytes(formula_af_fixture(
            "<math",
            false,
            Some("a formula description"),
            "[]",
        ))
        .unwrap();
        let mut source = PdfDocumentSource::from_document(document);
        let formulas = source.structured_math(0).unwrap();
        assert_eq!(formulas.len(), 1);
        assert_eq!(formulas[0].origin, MathOrigin::AltDescription);
        assert!(formulas[0].latex.is_empty());
        assert_eq!(
            formulas[0].description.as_deref(),
            Some("a formula description")
        );
    }

    #[test]
    fn cyclic_formula_kid_is_bounded_and_does_not_block_conversion() {
        let structure_mathml =
            "<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>z</mi></math>";
        let document =
            PdfDocument::from_bytes(formula_af_fixture(structure_mathml, false, None, "[6 0 R]"))
                .unwrap();
        let mut source = PdfDocumentSource::from_document(document);
        let start = std::time::Instant::now();
        let formulas = source.structured_math(0).unwrap();
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
        assert_eq!(formulas.len(), 1);
        assert_eq!(formulas[0].latex, "z");
    }

    #[test]
    fn formula_attachment_filename_ambiguity_is_reported() {
        let metadata = FormulaMetadata {
            associated_files: vec![AssociatedFileMetadata {
                filename: Some("formula.xml".to_string()),
                relationship: Some("Supplement".to_string()),
                mime_type: Some("application/mathml+xml".to_string()),
            }],
            ..FormulaMetadata::default()
        };
        let payload = b"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>x</mi></math>";
        let embedded = ["formula.xml", "formula.xml"].map(|filename| EmbeddedFile {
            filename: filename.to_string(),
            bytes: payload.to_vec(),
            mime_type: None,
            relationship: None,
        });

        let error = extract_associated_mathml(&metadata, &embedded, 1024).unwrap_err();
        assert!(error.iter().any(|warning| warning.contains("is ambiguous")));
    }

    #[test]
    fn unique_formula_attachment_is_converted_with_provenance_metadata() {
        let metadata = FormulaMetadata {
            associated_files: vec![AssociatedFileMetadata {
                filename: Some("formula.xml".to_string()),
                relationship: Some("Supplement".to_string()),
                mime_type: Some("application/mathml+xml".to_string()),
            }],
            ..FormulaMetadata::default()
        };
        let embedded = [EmbeddedFile {
            filename: "formula.xml".to_string(),
            bytes: b"<math xmlns=\"http://www.w3.org/1998/Math/MathML\"><mi>x</mi></math>".to_vec(),
            mime_type: None,
            relationship: None,
        }];

        let (latex, file) = extract_associated_mathml(&metadata, &embedded, 1024).unwrap();
        assert_eq!(latex, "x");
        assert_eq!(file.filename.as_deref(), Some("formula.xml"));
        assert_eq!(file.relationship.as_deref(), Some("Supplement"));
        assert_eq!(file.mime_type.as_deref(), Some("application/mathml+xml"));
    }

    #[test]
    #[ignore = "manual synthetic benchmark; run with --ignored --nocapture"]
    fn tagged_math_extraction_benchmark() {
        use std::time::{Duration, Instant};

        const FORMULA_COUNT: usize = 128;
        const RUNS: usize = 7;
        let mut writer = PdfWriter::with_config(PdfWriterConfig {
            tagged: true,
            ..PdfWriterConfig::default()
        });
        let mut page = writer.add_page(612.0, 792.0);
        for index in 0..FORMULA_COUNT {
            let x = 36.0 + (index % 32) as f32 * 16.0;
            let y = 740.0 - (index / 32) as f32 * 24.0;
            let formula = ContentElement::Structure(StructureElement {
                structure_type: "Formula".to_string(),
                bbox: Rect::new(x, y, 45.0, 18.0),
                children: vec![
                    math_tag("mi", "x", x),
                    math_tag("mo", "=", x + 12.0),
                    math_tag("mn", &index.to_string(), x + 24.0),
                ],
                reading_order: Some(index),
                alt_text: None,
                language: None,
            });
            page.add_element(&formula);
        }
        page.finish();
        let bytes = writer.finish().unwrap();
        let mut durations: Vec<Duration> = Vec::with_capacity(RUNS);
        let mut extracted = 0;
        for _ in 0..RUNS {
            let document = PdfDocument::from_bytes(bytes.clone()).unwrap();
            let mut source = PdfDocumentSource::from_document(document);
            let start = Instant::now();
            extracted = source.structured_math(0).unwrap().len();
            durations.push(start.elapsed());
        }
        durations.sort_unstable();
        eprintln!(
            "tagged math synthetic baseline: pdf_bytes={} formulas={} runs={} median_ms={:.3}",
            bytes.len(),
            extracted,
            RUNS,
            durations[RUNS / 2].as_secs_f64() * 1000.0
        );
        assert_eq!(extracted, FORMULA_COUNT);
    }
}
