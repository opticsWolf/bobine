use serde::{Deserialize, Serialize};

fn default_image_mode() -> ImageMode {
    ImageMode::Link
}

fn default_max_embed_bytes() -> usize {
    262_144
}

fn default_table_backend() -> TableBackend {
    TableBackend::Slanet
}

fn default_potatr_node_threshold() -> f32 {
    0.8
}

fn default_potatr_edge_threshold() -> f32 {
    0.5
}

fn default_potatr_suppress_iou() -> f32 {
    0.5
}

fn default_potatr_min_text_fill() -> f32 {
    0.2
}

fn default_render_gate_min_layer_words() -> usize {
    30
}

fn default_render_gate_min_jaccard() -> f32 {
    0.35
}

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
    /// TexTeller ONNX via ort (~1.19 GB disk, ~1 s/crop on CPU).
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
/// * `Fp32` (default): full-precision weights + KV-cache decode
///   (~1.19 GB). Reference output on CPU and CUDA.
/// * `Int8`: dynamic-int8 pair from the unified serving repo
///   (~303 MB, a quarter of the download/RAM). Fastest on CPU (≈1.9×
///   the fp32 median) at the cost of occasional decode drift; do not
///   pair it with CUDA (dequant/memcpy overhead makes it ~1.8× slower
///   than fp32 there).
///
/// `ModelPrecision::Fp16` is the CUDA alternative to fp32: half the
/// weights, ≈1.2× faster on long decodes, and 28–29/30 exact parity.
/// On CPU it is ~3× slower than fp32 (cast overhead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelQuantization {
    /// Full-precision TexTeller.
    Fp32,
    /// Dynamic-int8 pair from the unified serving repo (compact footprint).
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
    /// (measured 1.7x at 700x400, 6.7x at 1024x1024; see docs/benchmarks.md).
    #[serde(default)]
    pub table_ort_providers: Option<Vec<String>>,

    /// Provider override for the POTATR page-table slot.
    /// `None` (default) = **always CPU**: same attention-heavy profile
    /// as the table slot, and no CUDA numbers exist yet (P0 gate is
    /// CPU-only; measure before overriding).
    #[serde(default)]
    pub potatr_ort_providers: Option<Vec<String>>,
}

/// How figure images are referenced in the markdown output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageMode {
    /// `![alt](assets/p{n}/img{k}.png)` links (default — diff-friendly).
    Link,
    /// `![alt](data:image/png;base64,…)` URIs for figures at or under
    /// `max_embed_bytes`; larger figures stay links.
    EmbedBase64,
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

    /// How figure images are referenced: links (default) or opt-in
    /// base64 data URIs (capped by `max_embed_bytes`).
    #[serde(default = "default_image_mode")]
    pub image_mode: ImageMode,

    /// Cap for `ImageMode::EmbedBase64`: figures whose file is larger
    /// stay links (data URIs bloat/diff badly). Default 256 KiB.
    #[serde(default = "default_max_embed_bytes")]
    pub max_embed_bytes: usize,

    /// Skip semantic reflow on the fast path: emit text lines in strict
    /// visual (top-to-bottom, left-to-right) order — no heading
    /// promotion, no column merging, no table detection. For documents
    /// where reflow scrambles content (indexes, listings). Off by
    /// default; ONNX paths ignore it (structural by design).
    #[serde(default)]
    pub preserve_layout: bool,
}

/// Which table-structure backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableBackend {
    /// SLANet-plus crop-level path via `table_block` (default in v1;
    /// POTATR ships opt-in — see `docs/proposal_potatr.md` §6 flip gate).
    Slanet,
    /// POTATR page-level path (once P2 routing lands; P0 wires the slot
    /// + preprocess only).
    Potatr,
    /// POTATR on pages with words, SLANet fallback elsewhere (P2).
    Auto,
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

    /// Which table-structure backend to use (opt-in POTATR in v1).
    #[serde(default = "default_table_backend")]
    pub table_backend: TableBackend,

    /// POTATR node-score threshold for final table claims in P2 routing
    /// (reference eval default 0.8). Internal structure refinement keeps
    /// the reference 0.8 fixed for fidelity; this filters emitted tables.
    #[serde(default = "default_potatr_node_threshold")]
    pub potatr_node_threshold: f32,

    /// POTATR edge threshold (reference default 0.5, `inference.py`).
    #[serde(default = "default_potatr_edge_threshold")]
    pub potatr_edge_threshold: f32,

    /// IoU at which a POTATR-claimed box suppresses an overlapping layout
    /// region in P2 routing (option A: POTATR wins its claim set).
    #[serde(default = "default_potatr_suppress_iou")]
    pub potatr_suppress_iou: f32,

    /// Minimum fraction of a claimed table's cells carrying text. Claims
    /// below this (e.g. textless on CID-hole pages) fall back to the region
    /// arms instead of suppressing them.
    #[serde(default = "default_potatr_min_text_fill")]
    pub potatr_min_text_fill: f32,
}

/// Render-health gate knobs (Option C, `docs/cid_render_blocker.md` §2).
///
/// Temporary shim for the pdf_oxide CID-renderer defect (upstream #3098):
/// per page, on born-digital input with a substantial text layer, fire
/// when OCR↔text-layer word-Jaccard falls below threshold — the ONNX
/// output is discarded and the fast path is used instead. Fires only
/// while built against pdf_oxide < 0.3.79 (compile-time scope from
/// `build.rs`; the dep bump deactivates it automatically) **and**
/// `render_gate_enabled`. Math handling is suppression-first: TexTeller
/// spans on fired pages are dropped, with opt-in inline salvage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderHealthOpts {
    /// Master switch for the render-health gate. Default true; the
    /// version scope (`build.rs`) can still deactivate it underneath.
    #[serde(default = "default_true")]
    pub render_gate_enabled: bool,

    /// Minimum text-layer words for a page to count as born-digital
    /// with a substantial layer (gate only judges such pages).
    #[serde(default = "default_render_gate_min_layer_words")]
    pub render_gate_min_layer_words: usize,

    /// Fire when OCR↔layer word-Jaccard is strictly below this.
    #[serde(default = "default_render_gate_min_jaccard")]
    pub render_gate_min_jaccard: f32,

    /// Salvage inline-only formula boxes from the text layer as `$…$`
    /// on fired pages instead of dropping all math. Default false
    /// (suppression-first: confident false equations are worse than
    /// none; salvage is best-effort — corrupt layers salvage garbage).
    #[serde(default)]
    pub render_gate_salvage_inline_math: bool,
}

fn default_true() -> bool {
    true
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

    /// Post-OCR LaTeX repair pass over TexTeller output (command splitting,
    /// fence fixup, same-level script separation, brace-balance hard-fail).
    /// Default true — bug-fix positioning. Disable only to measure raw
    /// recognizer output.
    pub sanitize_ocr: bool,

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
    /// Render-health gate knobs (Option C shim, pdf_oxide < 0.3.79).
    #[serde(flatten)]
    pub health: RenderHealthOpts,
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
            potatr_ort_providers: None,
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
            image_mode: ImageMode::Link,
            max_embed_bytes: 262_144,
            preserve_layout: false,
        }
    }
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self {
            formula_backend: FormulaBackend::TexTeller,
            model_precision: ModelPrecision::Fp32,
            model_quantization: ModelQuantization::Fp32,
            ocr_lang: "en".to_string(),
            table_backend: TableBackend::Slanet,
            potatr_node_threshold: 0.8,
            potatr_edge_threshold: 0.5,
            potatr_suppress_iou: 0.5,
            potatr_min_text_fill: 0.2,
        }
    }
}

impl Default for RenderHealthOpts {
    fn default() -> Self {
        Self {
            render_gate_enabled: true,
            render_gate_min_layer_words: 30,
            render_gate_min_jaccard: 0.35,
            render_gate_salvage_inline_math: false,
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
            sanitize_ocr: true,
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
            health: RenderHealthOpts::default(),
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
        assert_eq!(c.models.model_quantization, ModelQuantization::Fp32);
        assert!(c.health.render_gate_enabled);
        assert_eq!(c.health.render_gate_min_layer_words, 30);
        assert_eq!(c.health.render_gate_min_jaccard, 0.35);
        assert!(!c.health.render_gate_salvage_inline_math);
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
            "render_gate_enabled",
            "render_gate_min_layer_words",
            "render_gate_min_jaccard",
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
        for group in ["routing", "providers", "render", "models", "text", "health"] {
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
