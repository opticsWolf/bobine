use serde::{Deserialize, Serialize};

/// Controls how pages are routed between fast (pdf_oxide) and ONNX paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Fast path only. No ONNX models loaded.
    Never,
    /// Heuristics per page → full ONNX pipeline only on flagged pages.
    Auto,
    /// Formula crops via formula OCR; full pipeline only for scans.
    Surgical,
    /// Every page through the full ONNX layout + OCR pipeline.
    Always,
}

/// Which formula recognition backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormulaBackend {
    /// TexTeller ONNX via ort (1.25 GB disk, ~2.5 GB RAM, ~1 s/crop).
    TexTeller,
}

/// Which numeric precision to use for ONNX model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPrecision {
    /// 32-bit float (default, ~2.1 GB model files).
    Fp32,
    /// 16-bit float (~1.0 GB model files, faster GPU, same CPU speed).
    Fp16,
}

/// Weight quantization for the formula recognizer (TexTeller) — the
/// accuracy/memory trade-off knob.
///
/// * `Int8` (default): onnx-community quantized exports
///   (`decoder_model_int8.onnx` etc., 316 MB total — one quarter the
///   download/RAM of fp32 at equal speed).
/// * `Fp32`: full-precision weights + KV-cache decode. Exact output;
///   ~1.25 GB of model files. Also carries
///   occasional *typographic* drift in the emitted LaTeX (lost `\mathbf`
///   bold, `\epsilon` vs `arepsilon`); math content is unaffected in
///   tests. The int8 export has no KV-cache inputs and decodes by
///   full-sequence recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelQuantization {
    /// Full-precision TexTeller (default).
    Fp32,
    /// onnx-community int8 exports (compact memory footprint).
    Int8,
}

/// Tunable knobs for the HybridConverter pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverterConfig {
    /// Extract embedded images from pages.
    pub extract_images: bool,

    /// Append unreferenced images as a gallery.
    pub append_unreferenced_images: bool,

    /// Whether ONNX models are loaded at all.
    pub use_onnx: bool,

    /// Page routing strategy.
    pub routing_mode: RoutingMode,

    /// Formula recognition backend.
    pub formula_backend: FormulaBackend,

    /// Numeric precision for ONNX model weights.
    pub model_precision: ModelPrecision,
    /// Formula-recognizer weight selection (accuracy vs memory). See
    /// [`ModelQuantization`].
    pub model_quantization: ModelQuantization,

    /// ONNX Runtime execution providers e.g. ["CPUExecutionProvider"].
    /// Requesting an accelerator the loaded ORT library lacks degrades
    /// gracefully to CPU.
    pub ort_providers: Vec<String>,

    /// Provider override for the TexTeller *encoder* session only.
    /// `None` (default) = use `ort_providers`. Useful to offload the
    /// compute-bound ViT encoder to a GPU while keeping the latency-bound
    /// autoregressive decoder on CPU, e.g. `["cuda", "cpu"]`.
    pub encoder_ort_providers: Option<Vec<String>>,

    /// Provider override for the TexTeller *decoder* (KV-cache graph)
    /// session only. `None` (default) = use `ort_providers`.
    #[serde(default)]
    pub decoder_ort_providers: Option<Vec<String>>,

    /// Provider override for the RapidLayout session.
    /// `None` (default) = **auto**: CUDA when the loaded ONNX Runtime
    /// library registers it (measured 12.3x faster), else the base
    /// `ort_providers`. Set e.g. `["CPUExecutionProvider"]` to pin.
    #[serde(default)]
    pub layout_ort_providers: Option<Vec<String>>,

    /// Provider override for the RapidOCR det + rec sessions.
    /// `None` (default) = **auto**, same policy as layout (measured
    /// 3.6x faster on CUDA).
    #[serde(default)]
    pub ocr_ort_providers: Option<Vec<String>>,

    /// Provider override for the RapidTable (SLANet-plus) session.
    /// `None` (default) = **always CPU**: SLANet's graph fragments
    /// across devices on CUDA and measures 2-9x slower than CPU
    /// (IMPLEMENTATION_PLAN.md).
    #[serde(default)]
    pub table_ort_providers: Option<Vec<String>>,

    /// DPI for scanned-page renders.
    pub render_dpi: u32,

    /// DPI for formula-crop renders.
    pub formula_dpi: u32,

    /// Detect markdown headings from fonts.
    pub detect_headings: bool,

    /// Convert HTML tables to GFM pipe tables.
    pub convert_html_tables: bool,

    /// Detect monospaced code blocks.
    pub detect_code_blocks: bool,

    /// Minimum math chars to consider a region a formula box.
    pub min_formula_math_chars: usize,

    /// Maximum width (points) for inline formula vs display.
    pub formula_inline_max_width_pts: f64,

    /// Pad formula crop boxes by this many points.
    pub formula_pad_pts: f64,

    /// Fall back to layout model for equation detection.
    pub formula_layout_fallback: bool,

    /// Threshold for page_math_signal → flag page as math-heavy.
    pub math_char_threshold: usize,

    /// Fewer chars than this + has images → scanned page.
    pub scanned_text_threshold: usize,

    /// OCR recognition language hint (e.g. "en", "ch", "ja", "latin").
    ///
    /// Used to select a CTC charset fallback when the rec model carries no
    /// `character` metadata. Models loaded via explicit paths usually embed
    /// their charset, making this a no-op for them.
    pub ocr_lang: String,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            extract_images: true,
            append_unreferenced_images: true,
            use_onnx: true,
            routing_mode: RoutingMode::Auto,
            formula_backend: FormulaBackend::TexTeller,
            model_precision: ModelPrecision::Fp32,
            model_quantization: ModelQuantization::Int8,
            ort_providers: vec!["CPUExecutionProvider".into()],
            encoder_ort_providers: None,
            decoder_ort_providers: None,
            layout_ort_providers: None,
            ocr_ort_providers: None,
            table_ort_providers: None,
            render_dpi: 300,
            formula_dpi: 200,
            detect_headings: true,
            convert_html_tables: true,
            detect_code_blocks: true,
            min_formula_math_chars: 5,
            formula_inline_max_width_pts: 220.0,
            formula_pad_pts: 4.0,
            formula_layout_fallback: false,
            math_char_threshold: 30,
            scanned_text_threshold: 50,
            ocr_lang: "en".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_routing() {
        let c = ConverterConfig::default();
        assert_eq!(c.routing_mode, RoutingMode::Auto);
        assert_eq!(c.formula_backend, FormulaBackend::TexTeller);
        assert_eq!(c.model_precision, ModelPrecision::Fp32);
    }

    #[test]
    fn config_fields_match_python() {
        let c = ConverterConfig::default();
        assert_eq!(c.render_dpi, 300);
        assert_eq!(c.formula_dpi, 200);
        assert_eq!(c.formula_inline_max_width_pts, 220.0);
        assert_eq!(c.formula_pad_pts, 4.0);
        assert_eq!(c.min_formula_math_chars, 5);
        assert_eq!(c.math_char_threshold, 30);
        assert_eq!(c.scanned_text_threshold, 50);
    }

    #[test]
    fn routing_mode_serialization() {
        let json = serde_json::to_string(&RoutingMode::Surgical).unwrap();
        assert_eq!(json, "\"surgical\"");
        let back: RoutingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RoutingMode::Surgical);
    }
}
