//! Provenance and source data for structured PDF mathematics.

use pdf_oxide::geometry::Rect;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where a formula's LaTeX came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathOrigin {
    TaggedMathMl,
    AssociatedFile,
    EmbeddedSource,
    OcrTexTeller,
    TextLayer,
    AltDescription,
}

/// One formula and its best available location on a PDF page.
#[derive(Debug, Clone)]
pub struct MathSpan {
    /// Zero-based page index.
    pub page: usize,
    /// PDF points, with the coordinate convention supplied by `pdf_oxide`.
    /// `None` means the source could not be localized reliably.
    pub bbox: Option<Rect>,
    /// Bare LaTeX. Descriptions are carried separately and are never emitted
    /// as math.
    pub latex: String,
    pub display: bool,
    pub origin: MathOrigin,
    pub confidence: f32,
    pub tag: Option<String>,
    pub warnings: Vec<String>,
    /// Accessibility description, when present. This is not LaTeX.
    pub description: Option<String>,
    /// Raw `/NS` value normalized to JSON for downstream inspection.
    pub structure_namespace: Option<Value>,
    /// Raw `/A` value normalized to JSON for downstream inspection.
    pub structure_attributes: Option<Value>,
    /// Filename resolved from the formula's `/AF` filespec, when matched.
    pub associated_filename: Option<String>,
    /// Raw `/AFRelationship` name from the matched filespec.
    pub associated_relationship: Option<String>,
    /// Embedded stream `/Subtype` from the matched filespec.
    pub associated_mime_type: Option<String>,
}

/// Counts and warnings from the most recent PDF conversion.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MathSourceReport {
    pub tagged_spans: usize,
    pub associated_file_spans: usize,
    pub embedded_source_spans: usize,
    pub ocr_spans: usize,
    pub text_layer_spans: usize,
    pub alt_only_spans: usize,
    pub pages_with_structured_math: usize,
    pub suppressed_ocr_spans: usize,
    pub validation_failed: usize,
    pub validation_repaired: usize,
    pub validation_skipped: usize,
    pub warnings: Vec<String>,
}

/// A file embedded in a PDF catalog name tree.
#[derive(Debug, Clone)]
pub struct EmbeddedFile {
    pub filename: String,
    pub bytes: Vec<u8>,
    /// Only populated when exposed by the PDF backend.
    pub mime_type: Option<String>,
    /// PDF `/AFRelationship`, when available.
    pub relationship: Option<String>,
}

/// Convert a MathML XML fragment with Bobine's existing converter.
pub(crate) fn mathml_to_latex(xml: &str) -> std::result::Result<String, String> {
    let parsed = roxmltree::Document::parse(xml).map_err(|e| format!("MathML parse: {e}"))?;
    let math = parsed
        .descendants()
        .find(|node| node.is_element() && node.tag_name().name() == "math")
        .ok_or_else(|| "MathML source has no <math> element".to_string())?;
    math_node_to_latex(math)
}

pub(crate) fn math_node_to_latex(
    math: roxmltree::Node<'_, '_>,
) -> std::result::Result<String, String> {
    let mut stats = crate::mml::MmlStats::default();
    let latex = crate::mml::mml_to_tex(math, &mut stats);
    if latex.trim().is_empty() {
        return Err("MathML conversion produced empty LaTeX".to_string());
    }
    Ok(latex)
}
