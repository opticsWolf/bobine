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

/// Manages the ONNX model lifecycle with lazy loading.
pub struct OnnxEngine {
    config: ConverterConfig,

    /// TexTeller formula recognition (loaded on demand).
    tex_teller: Option<TexTeller>,

    /// RapidLayout page layout analysis (loaded on demand).
    layout: Option<RapidLayout>,

    /// RapidOCR text detection + recognition (loaded on demand).
    ocr: Option<RapidOcr>,

    /// Cache directory for downloaded models.
    cache_dir: PathBuf,

    /// Path to the RapidLayout ONNX model file.
    layout_model_path: Option<PathBuf>,

    /// Path to the RapidOCR detection model.
    ocr_det_path: Option<PathBuf>,

    /// Path to the RapidOCR recognition model.
    ocr_rec_path: Option<PathBuf>,
}

impl OnnxEngine {
    pub fn new(config: &ConverterConfig, cache_dir: &Path) -> Self {
        Self {
            config: config.clone(),
            tex_teller: None,
            layout: None,
            ocr: None,
            cache_dir: cache_dir.to_path_buf(),
            layout_model_path: None,
            ocr_det_path: None,
            ocr_rec_path: None,
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
            let tt = TexTeller::from_pretrained(
                "OleehyO/TexTeller",
                &self.cache_dir,
                self.config.model_precision,
            )?;
            self.tex_teller = Some(tt);
        }
        Ok(())
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
            let layout = RapidLayout::load(path)?;
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
            let ocr = RapidOcr::load(det, rec)?;
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
