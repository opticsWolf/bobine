use serde::{Deserialize, Serialize};

/// Controls how pages are routed between fast (pdf_oxide) and ONNX paths.
///
/// Cost ladder: `Never` (no models) < `Surgical` (TexTeller only) < `Auto`
/// (heavy path on flagged pages) < `Always` (heavy path everywhere).
/// Every mode shares the same tail: heading promotion, code blocks,
/// figure interleaving, unreferenced-image gallery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Fast path only: pdf_oxide text layer + embedded figures.
    /// No ONNX models loaded (works with no runtime at all).
    Never,
    /// Per-page heuristics (`needs_onnx`: scanned / math-heavy / tables)
    /// → full layout + OCR pipeline only on flagged pages, fast path
    /// elsewhere. Loads TexTeller + layout + OCR; table model lazy-loads
    /// on the first scanned table region. The default.
    Auto,
    /// Fast path plus TexTeller formula crops spliced into the text
    /// layer; the full pipeline runs only for scanned pages. Loads
    /// TexTeller only — no layout model, no OCR.
    Surgical,
    /// Every page through the full ONNX layout + OCR pipeline, with
    /// hybrid formula refinement (text-layer math boxes intersect layout
    /// formula regions). Most thorough, most expensive.
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

/// Page-routing knobs: which pages take the ONNX heavy path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingOpts {
    /// Whether ONNX models are loaded at all.
    pub use_onnx: bool,

    /// Page routing strategy.
    pub routing_mode: RoutingMode,
}

/// ONNX Runtime execution-provider selection, with per-model overrides.
///
/// Layout/OCR slots auto-enable CUDA when the loaded library registers it
/// (measured 12.3x / 3.6x); the table slot stays CPU-pinned unless
/// overridden (SLANet measures 2-9x slower on CUDA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOpts {
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
}

/// Render / asset-staging knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderOpts {
    /// Extract embedded images from pages.
    pub extract_images: bool,

    /// Append unreferenced images as a gallery.
    pub append_unreferenced_images: bool,

    /// DPI for scanned-page renders.
    pub render_dpi: u32,

    /// DPI for formula-crop renders.
    pub formula_dpi: u32,

    /// Embedded images whose placement bbox is smaller than this many
    /// square points are treated as decoration (logos, rules, bullets):
    /// excluded from inline figure matching and from the unreferenced-image
    /// gallery.
    pub min_figure_area_pts: f64,

    /// Directory (relative to the work dir) where extracted figure assets
    /// are written, structured as `<dir>/p{page}/img{k}.{ext}`.
    pub image_output_dir: String,
}

/// Model-selection knobs (weights, backends, language).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOpts {
    /// Formula recognition backend.
    pub formula_backend: FormulaBackend,

    /// Numeric precision for ONNX model weights.
    pub model_precision: ModelPrecision,
    /// Formula-recognizer weight selection (accuracy vs memory). See
    /// [`ModelQuantization`].
    pub model_quantization: ModelQuantization,

    /// OCR recognition language hint (e.g. "en", "ch", "ja", "latin").
    ///
    /// Used to select a CTC charset fallback when the rec model carries no
    /// `character` metadata. Models loaded via explicit paths usually embed
    /// their charset, making this a no-op for them.
    pub ocr_lang: String,
}

/// Text-layer / markdown-shaping knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextOpts {
    /// Detect markdown headings from fonts.
    pub detect_headings: bool,

    /// Convert HTML tables to GFM pipe tables.
    pub convert_html_tables: bool,

    /// Probe layout table regions with structured grid extraction
    /// (Tagged-PDF structure tree / ruled-grid detection) before falling
    /// back to a plain text dump. Preserves cell/column structure.
    pub structured_tables: bool,

    /// Detect monospaced code blocks.
    pub detect_code_blocks: bool,

    /// Promote scholarly section headings ("I. INTRODUCTION", "3.1 Encoder
    /// Stacks") that pdf_oxide's font heuristics leave as bold/plain text to
    /// proper markdown headings. Purely pattern-based, guarded against prose
    /// false positives.
    pub promote_headings: bool,

    /// Promote the document title: if the first block of the document is a
    /// short, non-sentence line it becomes `# …`; otherwise the first heading
    /// found near the top of the document is promoted to level 1.
    pub promote_title: bool,

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
}

/// Tunable knobs for the HybridConverter pipeline.
///
/// Grouped into sub-structs by concern; `#[serde(flatten)]` keeps the
/// serialized (JSON/TOML) form flat, so files written by older versions
/// still parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverterConfig {
    /// Page-routing knobs.
    #[serde(flatten)]
    pub routing: RoutingOpts,
    /// Execution-provider selection.
    #[serde(flatten)]
    pub providers: ProviderOpts,
    /// Render / asset-staging knobs.
    #[serde(flatten)]
    pub render: RenderOpts,
    /// Model-selection knobs.
    #[serde(flatten)]
    pub models: ModelOpts,
    /// Text-layer / markdown-shaping knobs.
    #[serde(flatten)]
    pub text: TextOpts,
}

impl Default for RoutingOpts {
    fn default() -> Self {
        Self {
            use_onnx: true,
            routing_mode: RoutingMode::Auto,
        }
    }
}

impl Default for ProviderOpts {
    fn default() -> Self {
        Self {
            ort_providers: vec!["CPUExecutionProvider".into()],
            encoder_ort_providers: None,
            decoder_ort_providers: None,
            layout_ort_providers: None,
            ocr_ort_providers: None,
            table_ort_providers: None,
        }
    }
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            extract_images: true,
            append_unreferenced_images: true,
            render_dpi: 300,
            formula_dpi: 200,
            min_figure_area_pts: 100.0,
            image_output_dir: "assets".to_string(),
        }
    }
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self {
            formula_backend: FormulaBackend::TexTeller,
            model_precision: ModelPrecision::Fp32,
            model_quantization: ModelQuantization::Int8,
            ocr_lang: "en".to_string(),
        }
    }
}

impl Default for TextOpts {
    fn default() -> Self {
        Self {
            detect_headings: true,
            convert_html_tables: true,
            structured_tables: true,
            detect_code_blocks: true,
            promote_headings: true,
            promote_title: true,
            min_formula_math_chars: 5,
            formula_inline_max_width_pts: 220.0,
            formula_pad_pts: 4.0,
            formula_layout_fallback: false,
            math_char_threshold: 30,
            scanned_text_threshold: 50,
        }
    }
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            routing: RoutingOpts::default(),
            providers: ProviderOpts::default(),
            render: RenderOpts::default(),
            models: ModelOpts::default(),
            text: TextOpts::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_routing() {
        let c = ConverterConfig::default();
        assert_eq!(c.routing.routing_mode, RoutingMode::Auto);
        assert_eq!(c.models.formula_backend, FormulaBackend::TexTeller);
        assert_eq!(c.models.model_precision, ModelPrecision::Fp32);
    }

    #[test]
    fn config_fields_match_python() {
        let c = ConverterConfig::default();
        assert_eq!(c.render.render_dpi, 300);
        assert_eq!(c.render.formula_dpi, 200);
        assert_eq!(c.text.formula_inline_max_width_pts, 220.0);
        assert_eq!(c.text.formula_pad_pts, 4.0);
        assert_eq!(c.text.min_formula_math_chars, 5);
        assert_eq!(c.text.math_char_threshold, 30);
        assert_eq!(c.text.scanned_text_threshold, 50);
    }

    #[test]
    fn grouped_config_serde_stays_flat() {
        // The grouped struct must serialize exactly like the old flat one
        // so config files written by older versions still parse.
        let c = ConverterConfig::default();
        let v = serde_json::to_value(&c).unwrap();
        for key in [
            "routing_mode",
            "use_onnx",
            "ort_providers",
            "encoder_ort_providers",
            "render_dpi",
            "formula_dpi",
            "extract_images",
            "model_precision",
            "ocr_lang",
            "detect_headings",
            "scanned_text_threshold",
        ] {
            assert!(v.get(key).is_some(), "missing flat key {key}");
        }
        for group in ["routing", "providers", "render", "models", "text"] {
            assert!(v.get(group).is_none(), "nested group {group} leaked");
        }
        let back: ConverterConfig = serde_json::from_value(v).unwrap();
        assert_eq!(format!("{:?}", back), format!("{:?}", c));
    }

    #[test]
    fn routing_mode_serialization() {
        let json = serde_json::to_string(&RoutingMode::Surgical).unwrap();
        assert_eq!(json, "\"surgical\"");
        let back: RoutingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, RoutingMode::Surgical);
    }
}
