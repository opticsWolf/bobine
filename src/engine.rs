// ONNX inference engine — lazy model loading with ort.
//
// - pdf_oxide for text-layer extraction
// - TexTeller for formula OCR (auto-downloaded from HuggingFace)
// - RapidLayout for page layout analysis

use std::path::{Path, PathBuf};

use tracing::info;

use crate::config::{ConverterConfig, RoutingMode};
use crate::error::Result;
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
            "directmlexecutionprovider" | "directml" => {
                Some(DirectML::default().build())
            }
            "openvinoexecutionprovider" | "openvino" => {
                Some(OpenVINO::default().build())
            }
            "coremlexecutionprovider" | "coreml" => {
                Some(CoreML::default().build())
            }
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
    builder.with_execution_providers(&eps).map_err(|e| {
        tracing::warn!(error = %e, "provider registration failed; CPU-only session");
        crate::error::BobineError::Ort(e.to_string())
    })
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
        if !self.config.use_onnx {
            return Ok(());
        }
        match self.config.routing_mode {
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
                self.config.model_precision
            );
            let tt = match self.config.model_quantization {
                crate::config::ModelQuantization::Fp32 => TexTeller::from_pretrained(
                    "OleehyO/TexTeller",
                    &self.cache_dir,
                    self.config.model_precision,
                    &self.config.ort_providers,
                )?,
                crate::config::ModelQuantization::Int8 => {
                    TexTeller::from_pretrained_int8(&self.cache_dir, &self.config.ort_providers)?
                }
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
            self.table = Some(crate::rapid_table::RapidTable::load(
                &path,
                &self.config.ort_providers,
            )?);
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
        self.ensure_tex_teller()?;
        let tt = self.tex_teller.as_mut().unwrap();
        let latex = tt.recognize(image_path)?;
        Ok(Some(latex))
    }

    // ------------------------------------------------------------------
    // RapidLayout
    // ------------------------------------------------------------------

    fn ensure_layout(&mut self) -> Result<()> {
        if self.layout.is_none() {
            let path = self
                .layout_model_path
                .as_deref()
                .unwrap_or_else(|| Path::new("layout.onnx"));
            info!("Loading RapidLayout from {}...", path.display());
            let layout = RapidLayout::load(path, &self.config.ort_providers)?;
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
            let det = self.ocr_det_path.as_deref().unwrap_or_else(|| Path::new("det.onnx"));
            let rec = self.ocr_rec_path.as_deref().unwrap_or_else(|| Path::new("rec.onnx"));
            info!("Loading RapidOCR from {} and {}...", det.display(), rec.display());
            let ocr = RapidOcr::load(det, rec, &self.config.ocr_lang, &self.config.ort_providers)?;
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
            let md = match self.config.routing_mode {
                RoutingMode::Never | RoutingMode::Auto => doc
                    .to_markdown(i as usize, &Default::default())
                    .unwrap_or_else(|_| doc.extract_text(i as usize).unwrap_or_default()),
                RoutingMode::Surgical | RoutingMode::Always => doc
                    .to_markdown(i as usize, &Default::default())
                    .unwrap_or_else(|_| doc.extract_text(i as usize).unwrap_or_default()),
            };
            md_pages.push(md);
        }

        Ok(md_pages.join("\n\n---\n\n"))
    }
}
