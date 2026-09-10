// ONNX inference engine — lazy model loading with ort.
//
// - pdf_oxide for text-layer extraction
// - TexTeller for formula OCR (auto-downloaded from HuggingFace)
// - RapidLayout for page layout analysis

use std::path::{Path, PathBuf};

use tracing::info;

use crate::config::{ConverterConfig, RoutingMode};
use crate::error::{BobineError, Result};
use crate::rapid_layout::RapidLayout;
use crate::rapid_ocr::RapidOcr;
use crate::tex_teller::TexTeller;

/// Apply configured execution providers to an ort session builder.
/// Unknown names are warned and skipped; CPU is always available implicitly.
pub(crate) fn apply_providers(
    builder: ort::session::builder::SessionBuilder,
    providers: &[String],
) -> crate::error::Result<ort::session::builder::SessionBuilder> {
    use ort::ep::*;
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
    for p in providers {
        let lower = p.to_lowercase();
        let dispatch = match lower.as_str() {
            "cudaexecutionprovider" | "cuda" => Some(CUDA::default().build()),
            "rocmexecutionprovider" | "rocm" => Some(ROCm::default().build()),
            "directmlexecutionprovider" | "directml" => Some(DirectML::default().build()),
            "openvinoexecutionprovider" | "openvino" => Some(OpenVINO::default().build()),
            "coremlexecutionprovider" | "coreml" => Some(CoreML::default().build()),
            "cpuexecutionprovider" | "cpu" => None, // implicit default
            other => {
                tracing::warn!(provider = other, "unknown ORT provider, skipping");
                None
            }
        };
        if let Some(d) = dispatch {
            eps.push(d);
        }
    }
    if eps.is_empty() {
        return Ok(builder);
    }

    // Dynamic provider selection: with load-dynamic, whether an accelerator
    // actually works depends solely on which ONNX Runtime shared library is
    // loaded (ORT_DYLIB_PATH). A CPU-only library cannot register CUDA, so
    // treat registration failure as "fall back to CPU" instead of failing
    // the whole conversion. The clone deep-copies the session options
    // (CloneSessionOptions), so the original stays pristine for fallback.
    let attempt = builder.clone();
    match attempt.with_execution_providers(&eps) {
        Ok(configured) => Ok(configured),
        Err(e) => {
            tracing::warn!(
                error = %e,
                requested = ?providers,
                "accelerator providers unavailable in this ONNX Runtime library; using CPU"
            );
            Ok(builder)
        }
    }
}

/// Whether the loaded ONNX Runtime library can register the CUDA
/// execution provider. Probed once — registration is the step that
/// fails on a CPU-only dylib (see `apply_providers`).
fn cuda_available() -> bool {
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PROBE.get_or_init(|| {
        let Ok(builder) = ort::session::Session::builder() else {
            return false;
        };
        let cuda = ort::ep::CUDA::default().build();
        builder.with_execution_providers(std::slice::from_ref(&cuda)).is_ok()
    })
}

// ------------------------------------------------------------------
// HuggingFace model acquisition (layout / OCR)
// ------------------------------------------------------------------

/// DocLayout-YOLO ONNX conversion of the official DocStructBench weights
/// (the author's repo ships .pt only; this export matches bobine's
/// fallback label order).
const LAYOUT_REPO: (&str, &str) = ("wybxc", "DocLayout-YOLO-DocStructBench-onnx");
const LAYOUT_FILENAME: &str = "doclayout_yolo_docstructbench_imgsz1024.onnx";

/// PP-OCRv4 mobile det + rec exports; the rec model embeds its charset in
/// the ONNX custom metadata key "character".
const OCR_REPO: (&str, &str) = ("SWHL", "RapidOCR");
const OCR_DET_FILENAME: &str = "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx";
const OCR_REC_FILENAME: &str = "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx";

/// Download a file from HuggingFace into `cache_dir/<filename>` unless it
/// already exists (download_file never probes the destination itself).
fn hf_fetch(cache_dir: &Path, repo: (&str, &str), filename: &str) -> Result<PathBuf> {
    let dest = cache_dir.join(filename);
    if dest.exists() {
        return Ok(dest);
    }
    info!(
        repo = repo.0,
        file = filename,
        "Downloading model from HuggingFace..."
    );
    let client =
        hf_hub::HFClientSync::new().map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    client
        .model(repo.0, repo.1)
        .download_file()
        .filename(filename.to_string())
        .local_dir(cache_dir.to_path_buf())
        .send()
        .map_err(|e| BobineError::Ort(format!("download {filename}: {e}")))?;
    Ok(dest)
}

/// Manages the ONNX model lifecycle with lazy loading.
pub struct OnnxEngine {
    config: ConverterConfig,

    /// TexTeller formula recognition (loaded on demand).
    tex_teller: Option<TexTeller>,

    /// RapidLayout page layout analysis (loaded on demand).
    layout: Option<RapidLayout>,

    /// RapidOCR text detection + recognition (loaded on demand).
    ocr: Option<RapidOcr>,

    /// RapidTable table-structure recognition (loaded on demand, lazily —
    /// only when a scanned table region is actually encountered).
    table: Option<crate::rapid_table::RapidTable>,

    /// Cache directory for downloaded models.
    cache_dir: PathBuf,

    /// Path to the RapidLayout ONNX model file.
    layout_model_path: Option<PathBuf>,

    /// Path to the RapidOCR detection model.
    ocr_det_path: Option<PathBuf>,

    /// Path to the RapidOCR recognition model.
    ocr_rec_path: Option<PathBuf>,

    /// Path to a SLANet-plus table-structure model.
    table_model_path: Option<PathBuf>,
}

impl OnnxEngine {
    /// Resolve providers for a GPU-eligible slot (layout / OCR): an
    /// explicit override, else CUDA + CPU fallback when the loaded ORT
    /// library registers CUDA (measured 12.3x / 3.6x speedups), else
    /// the base `ort_providers` list.
    fn resolve_auto_gpu_providers(
        &self,
        override_providers: Option<&Vec<String>>,
        slot: &str,
    ) -> Vec<String> {
        match override_providers {
            Some(p) => p.clone(),
            None if cuda_available() => {
                tracing::info!(
                    slot,
                    "CUDA registers in the loaded ONNX Runtime library: auto-enabling CUDAExecutionProvider"
                );
                vec![
                    "CUDAExecutionProvider".to_string(),
                    "CPUExecutionProvider".to_string(),
                ]
            }
            None => self.config.providers.ort_providers.clone(),
        }
    }

    /// Resolve providers for the table slot: explicit override, else
    /// always CPU — SLANet measures 2-9x slower on CUDA (the graph
    /// fragments across devices).
    fn resolve_table_providers(&self) -> Vec<String> {
        match self.config.providers.table_ort_providers.clone() {
            Some(p) => p,
            None => {
                if cuda_available() {
                    tracing::info!(
                        "table slot pinned to CPUExecutionProvider (SLANet measures 2-9x slower on CUDA; set table_ort_providers to override)"
                    );
                }
                vec!["CPUExecutionProvider".to_string()]
            }
        }
    }

    pub fn new(config: &ConverterConfig, cache_dir: &Path) -> Self {
        Self {
            config: config.clone(),
            tex_teller: None,
            layout: None,
            ocr: None,
            table: None,
            cache_dir: cache_dir.to_path_buf(),
            layout_model_path: None,
            ocr_det_path: None,
            ocr_rec_path: None,
            table_model_path: None,
        }
    }

    /// Set the path to the RapidLayout ONNX model file.
    pub fn set_layout_model(&mut self, path: &Path) {
        self.layout_model_path = Some(path.to_path_buf());
    }

    /// Set paths for RapidOCR detection + recognition ONNX models.
    pub fn set_ocr_models(&mut self, det: &Path, rec: &Path) {
        self.ocr_det_path = Some(det.to_path_buf());
        self.ocr_rec_path = Some(rec.to_path_buf());
    }

    /// Set an explicit SLANet-plus table-structure ONNX model path.
    /// When unset, the model auto-downloads from HuggingFace on first use.
    pub fn set_table_model(&mut self, path: &Path) {
        self.table_model_path = Some(path.to_path_buf());
    }

    /// Load models required by the current routing mode.
    pub fn ensure_models(&mut self) -> Result<()> {
        if !self.config.routing.use_onnx {
            return Ok(());
        }
        match self.config.routing.routing_mode {
            RoutingMode::Never => {}
            RoutingMode::Surgical => {
                self.ensure_tex_teller()?;
            }
            _ => {
                self.ensure_tex_teller()?;
                self.ensure_layout()?;
                self.ensure_ocr()?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // TexTeller
    // ------------------------------------------------------------------

    fn ensure_tex_teller(&mut self) -> Result<()> {
        if self.tex_teller.is_none() {
            info!(
                "Loading TexTeller ({:?}) from OleehyO/TexTeller...",
                self.config.models.model_precision
            );
            // Quantized ops have no CUDA kernels - requesting CUDA with
            // Int8 weights makes ORT split the graph across devices
            // (171-267 Memcpy nodes) and runs ~2x SLOWER than plain CPU.
            if self.config.models.model_quantization == crate::config::ModelQuantization::Int8
                && self
                    .config
                    .providers
                    .ort_providers
                    .iter()
                    .chain(self.config.providers.encoder_ort_providers.iter().flatten())
                    .chain(self.config.providers.decoder_ort_providers.iter().flatten())
                    .any(|p| p.to_lowercase().contains("cuda"))
            {
                tracing::warn!(
                    "model_quantization=Int8 combined with CUDA providers: quantized ops                      fall back across devices (Memcpy-node overhead) and measure ~2x slower                      than CPU. Prefer model_quantization=Fp32 when running on a GPU."
                );
            }
            let enc_providers = self.config.providers.encoder_ort_providers.as_deref();
            let dec_providers: Vec<String> = self
                .config
                .providers
                .decoder_ort_providers
                .clone()
                .unwrap_or_else(|| self.config.providers.ort_providers.clone());
            let tt = match self.config.models.model_quantization {
                crate::config::ModelQuantization::Fp32 => TexTeller::from_pretrained_split(
                    "OleehyO/TexTeller",
                    &self.cache_dir,
                    self.config.models.model_precision,
                    enc_providers,
                    &dec_providers,
                )?,
                crate::config::ModelQuantization::Int8 => TexTeller::from_pretrained_int8_split(
                    &self.cache_dir,
                    enc_providers,
                    &dec_providers,
                )?,
            };
            self.tex_teller = Some(tt);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // RapidTable (lazy: downloads on first table region)
    // ------------------------------------------------------------------

    fn ensure_table(&mut self) -> Result<()> {
        if self.table.is_none() {
            let path = match self.table_model_path.clone() {
                Some(p) => p,
                None => crate::rapid_table::download_slanet_plus(&self.cache_dir)?,
            };
            let providers = self.resolve_table_providers();
            self.table = Some(crate::rapid_table::RapidTable::load(&path, &providers)?);
        }
        Ok(())
    }

    /// Recognize a table crop using OCR lines; returns full HTML or None.
    pub fn recognize_table(
        &mut self,
        img: &image::DynamicImage,
        ocr_lines: &[crate::rapid_ocr::OcrLine],
    ) -> Result<Option<String>> {
        self.ensure_table()?;
        self.table.as_mut().unwrap().recognize(img, ocr_lines)
    }

    pub fn recognize_formula(&mut self, image_path: &Path) -> Result<Option<String>> {
        self.recognize_formula_capped(image_path, 1024)
    }

    /// Like [`recognize_formula`] but with an explicit decode-step budget.
    /// Scale it with the crop's area - small crops cannot contain large
    /// equations, and the cap turns pathological inputs into fast failures.
    pub fn recognize_formula_capped(
        &mut self,
        image_path: &Path,
        max_tokens: usize,
    ) -> Result<Option<String>> {
        self.ensure_tex_teller()?;
        let tt = self.tex_teller.as_mut().unwrap();
        let prev = std::mem::replace(&mut tt.max_tokens, max_tokens.clamp(16, 1024));
        let r = tt.recognize(image_path);
        // Restore even on error so a failed crop cannot clamp later pages.
        self.tex_teller.as_mut().unwrap().max_tokens = prev;
        Ok(Some(r?))
    }

    // ------------------------------------------------------------------
    // RapidLayout
    // ------------------------------------------------------------------

    fn ensure_layout(&mut self) -> Result<()> {
        if self.layout.is_none() {
            let path = match self.layout_model_path.clone() {
                Some(p) => p,
                None => hf_fetch(&self.cache_dir, LAYOUT_REPO, LAYOUT_FILENAME)?,
            };
            info!("Loading RapidLayout from {}...", path.display());
            let providers = self.resolve_auto_gpu_providers(
                self.config.providers.layout_ort_providers.as_ref(),
                "layout",
            );
            let layout = RapidLayout::load(&path, &providers)?;
            self.layout = Some(layout);
        }
        Ok(())
    }

    pub fn layout_regions(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<crate::rapid_layout::LayoutRegion>> {
        self.ensure_layout()?;
        self.layout.as_mut().unwrap().detect(img)
    }

    // ------------------------------------------------------------------
    // RapidOCR
    // ------------------------------------------------------------------

    fn ensure_ocr(&mut self) -> Result<()> {
        if self.ocr.is_none() {
            let det = match self.ocr_det_path.clone() {
                Some(p) => p,
                None => hf_fetch(&self.cache_dir, OCR_REPO, OCR_DET_FILENAME)?,
            };
            let rec = match self.ocr_rec_path.clone() {
                Some(p) => p,
                None => hf_fetch(&self.cache_dir, OCR_REPO, OCR_REC_FILENAME)?,
            };
            info!(
                "Loading RapidOCR from {} and {}...",
                det.display(),
                rec.display()
            );
            let providers = self.resolve_auto_gpu_providers(
                self.config.providers.ocr_ort_providers.as_ref(),
                "ocr",
            );
            let ocr = RapidOcr::load(&det, &rec, &self.config.models.ocr_lang, &providers)?;
            self.ocr = Some(ocr);
        }
        Ok(())
    }

    pub fn ocr_lines(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<crate::rapid_ocr::OcrLine>> {
        self.ensure_ocr()?;
        self.ocr.as_mut().unwrap().detect_and_recognize(img)
    }

    // ------------------------------------------------------------------
    // PDF conversion (fast path via pdf_oxide)
    // ------------------------------------------------------------------

    pub fn convert_pdf(&mut self, path: &Path, _work_dir: &Path) -> Result<String> {
        self.ensure_models()?;

        let pdf_bytes = std::fs::read(path)?;
        let doc = pdf_oxide::PdfDocument::from_bytes(pdf_bytes)
            .map_err(|e| crate::error::BobineError::PdfOxide(format!("{:?}", e)))?;

        let n_pages = doc
            .page_count()
            .map_err(|e| crate::error::BobineError::PdfOxide(format!("{:?}", e)))?;

        let mut md_pages: Vec<String> = Vec::new();

        for i in 0..n_pages {
            // NOTE: routing (Never/Auto/Surgical/Always) lives in
            // HybridConverter::route_page_inner; this fast-path helper
            // intentionally ignores it.
            let md = doc
                .to_markdown(i as usize, &Default::default())
                .unwrap_or_else(|_| doc.extract_text(i as usize).unwrap_or_default());
            md_pages.push(md);
        }

        Ok(md_pages.join("\n\n---\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderOpts;

    /// Requesting CUDA on a machine/library without it must degrade to CPU
    /// gracefully (Ok), never fail the session build.
    #[test]
    fn cuda_request_degrades_gracefully_without_gpu() {
        // load-dynamic requires a resolvable ONNX Runtime library
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skipped: ORT_DYLIB_PATH not set");
            return;
        }
        let builder = ort::session::Session::builder().unwrap();
        let result = apply_providers(
            builder,
            &["CUDAExecutionProvider".to_string(), "cpu".to_string()],
        );
        assert!(
            result.is_ok(),
            "cuda request must fall back to cpu: {}",
            result
                .as_ref()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        );
    }

    #[test]
    fn unknown_providers_are_skipped_and_cpu_still_works() {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skipped: ORT_DYLIB_PATH not set");
            return;
        }
        let builder = ort::session::Session::builder().unwrap();
        let result = apply_providers(
            builder,
            &["warp-drive".to_string(), "CPUExecutionProvider".to_string()],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn table_slot_defaults_to_cpu_even_with_cuda_base() {
        let cfg = ConverterConfig {
            providers: ProviderOpts {
                ort_providers: vec![
                    "CUDAExecutionProvider".into(),
                    "CPUExecutionProvider".into(),
                ],
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = OnnxEngine::new(&cfg, Path::new("/tmp/bobine_test"));
        assert_eq!(
            engine.resolve_table_providers(),
            vec!["CPUExecutionProvider".to_string()]
        );

        // An explicit override is honored, even against the default policy.
        let cfg = ConverterConfig {
            providers: ProviderOpts {
                table_ort_providers: Some(vec!["CUDAExecutionProvider".into()]),
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = OnnxEngine::new(&cfg, Path::new("/tmp/bobine_test"));
        assert_eq!(
            engine.resolve_table_providers(),
            vec!["CUDAExecutionProvider".to_string()]
        );
    }

    #[test]
    fn auto_gpu_slot_respects_pins_and_cuda_probe() {
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skipped: ORT_DYLIB_PATH not set");
            return;
        }
        let cfg = ConverterConfig::default();
        let engine = OnnxEngine::new(&cfg, Path::new("/tmp/bobine_test"));
        let resolved = engine.resolve_auto_gpu_providers(None, "layout");
        if cuda_available() {
            // GPU dylib: CUDA first, CPU fallback behind it.
            assert_eq!(resolved[0], "CUDAExecutionProvider");
            assert!(resolved.contains(&"CPUExecutionProvider".to_string()));
        } else {
            // CPU-only dylib: untouched base list.
            assert_eq!(resolved, vec!["CPUExecutionProvider".to_string()]);
        }
        // An explicit pin always wins.
        let pinned = engine.resolve_auto_gpu_providers(
            Some(&vec!["CPUExecutionProvider".to_string()]),
            "ocr",
        );
        assert_eq!(pinned, vec!["CPUExecutionProvider".to_string()]);
    }
}
