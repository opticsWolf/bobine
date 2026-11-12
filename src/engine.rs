// ONNX inference engine — lazy model loading with ort.
//
// - pdf_oxide for text-layer extraction
// - TexTeller for formula OCR (auto-downloaded from HuggingFace)
// - RapidLayout for page layout analysis

use std::path::{Path, PathBuf};

use embroider::{SessionPolicy, cuda_available};
use tracing::info;

use crate::config::{ConverterConfig, RoutingMode};
use crate::error::{BobineError, Result};
use crate::rapid_layout::RapidLayout;
use crate::rapid_ocr::RapidOcr;
use crate::tex_teller::TexTeller;

/// Session-builder entrypoint shared by every slot: the crate's session
/// policy — ORT defaults; this engine never tunes threads/optimization —
/// applied explicitly via `SessionPolicy::ort_defaults()`, then the shared
/// clone-and-fallback provider plumbing from `embroider` (which also carries
/// the corrected CUDA probe: EP availability, not the lax registration probe
/// this crate used to duplicate here).
/// Arena stays ON (plain `apply_providers`): the arena-off path exists for
/// text embedding only (8x RSS win there) and must never leak into vision
/// without its own benchmark — pinned by `vision_slots_never_use_text_policy`.
/// Unknown provider names are warned and skipped; CPU is always available
/// implicitly. The returned builder is ready for `commit_from_file`.
pub(crate) fn session_builder(
    providers: &[String],
) -> Result<ort::session::builder::SessionBuilder> {
    let builder =
        ort::session::Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?;
    let tuned = vision_policy()
        .apply(builder)
        .map_err(|e| BobineError::Ort(e.to_string()))?;
    Ok(embroider::apply_providers(tuned, providers))
}

/// The one session policy every vision slot gets (via `session_builder`).
/// Its own fn so the test below pins what is actually wired, not just
/// embroider's constructors.
pub(crate) fn vision_policy() -> SessionPolicy {
    SessionPolicy::ort_defaults()
}

// ------------------------------------------------------------------
// HuggingFace model acquisition (layout / OCR)
// ------------------------------------------------------------------

/// DocLayout-YOLO ONNX conversion of the official DocStructBench weights
/// (the author's repo ships .pt only; this export matches bobine's
/// fallback label order).
const LAYOUT_REPO: (&str, &str) = ("opticsWolf", "pp-doclayout-plus-l-onnx");
/// RT-DETR-L, 20 classes; the class names travel with the weights
/// (the ONNX export carries no metadata).
const LAYOUT_FILENAME: &str = "pp_doclayout_plus_l.onnx";
const LAYOUT_LABELS_FILENAME: &str = "labels.txt";
/// Pinned hub revision for the PP-DocLayout_plus-L weights + labels
/// (Apache-2.0, byte-identical to the official PaddlePaddle export).
const LAYOUT_REV: &str = "cee9ee321d3460a25b497d899aa4719b4a5800ba";

/// PP-OCRv5 mobile det + rec exports (Apache-2.0, byte-identical to the
/// official PaddlePaddle ONNX). The rec export carries no charset
/// metadata — the dict below (18,383 lines → 18,385 classes via
/// blank+space alignment) travels with the weights.
const OCR_REPO: (&str, &str) = ("opticsWolf", "rapidocr-ppocrv5-onnx");
const OCR_DET_FILENAME: &str = "ppocrv5_det_infer.onnx";
const OCR_REC_FILENAME: &str = "ppocrv5_rec_infer.onnx";
const OCR_DICT_FILENAME: &str = "ppocrv5_dict.txt";
/// Pinned hub revision for the PP-OCRv5 det + rec + dict.
const OCR_REV: &str = "d19f569c677b3b883276a901a2cefe1aaf0e36a6";

/// Download a file from HuggingFace through the shared hub cache (see
/// [`crate::hub_cache`]): `cache_dir` is the hub-cache root, the file
/// resolves to its snapshot path, and an already-cached file costs zero
/// network ("using cached").
fn hf_fetch(
    cache_dir: &Path,
    repo: (&str, &str),
    filename: &str,
    what: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    crate::hub_cache::hub_fetch(cache_dir, repo.0, repo.1, filename, what, revision)
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

    /// POTATR page-level table extraction (loaded on demand, lazily —
    /// only when `table_backend` selects it and a page has words).
    /// Opt-in in v1 (`TableBackend::Slanet` default); see proposal_potatr.
    potatr: Option<crate::rapid_potatr::Potatr>,

    /// Cache directory for downloaded models: the hub-cache root (an
    /// override of the standard HuggingFace cache, not a flat model dir).
    cache_dir: PathBuf,

    /// Path to the RapidLayout ONNX model file.
    layout_model_path: Option<PathBuf>,

    /// Path to the RapidOCR detection model.
    ocr_det_path: Option<PathBuf>,

    /// Path to the RapidOCR recognition model.
    ocr_rec_path: Option<PathBuf>,

    /// Path to a SLANet-plus table-structure model.
    table_model_path: Option<PathBuf>,

    /// Path to a POTATR page-level table model.
    potatr_model_path: Option<PathBuf>,
}

impl OnnxEngine {
    /// The resolved plan for this engine, or the policy-only view when no
    /// ONNX pass has opened yet (probing CUDA would load the runtime just
    /// to answer a question about a future load).
    fn plan(&self) -> crate::config::RuntimePlan {
        let cuda = if self.any_loaded() { Some(embroider::cuda_available()) } else { None };
        self.config.runtime_plan(cuda)
    }

    /// Drop every open session (0.9, plan B4: the caller owns the
    /// lifetime). Idempotent; a later pass re-opens lazily, so this is only
    /// worth calling when the caller means it.
    pub fn close(&mut self) {
        self.tex_teller = None;
        self.layout = None;
        self.ocr = None;
        self.table = None;
        self.potatr = None;
    }

    /// Whether any ONNX session is already open.
    pub fn any_loaded(&self) -> bool {
        self.tex_teller.is_some()
            || self.layout.is_some()
            || self.ocr.is_some()
            || self.table.is_some()
            || self.potatr.is_some()
    }

    /// Which passes hold a session (for `describe()`).
    pub fn loaded(&self) -> crate::converter::LoadedPasses {
        crate::converter::LoadedPasses {
            formula: self.tex_teller.is_some(),
            layout: self.layout.is_some(),
            ocr: self.ocr.is_some(),
            table: self.table.is_some(),
            potatr: self.potatr.is_some(),
        }
    }

    /// Providers for a slot, from the plan. The plan is concrete here: a
    /// session is about to open, so the CUDA question is answerable.
    fn providers_for(&self, slot: &str) -> Vec<String> {
        let plan = self.config.runtime_plan(Some(embroider::cuda_available()));
        let pass = match slot {
            "encoder" => &plan.encoder,
            "decoder" => &plan.decoder,
            "layout" => &plan.layout,
            "ocr" => &plan.ocr,
            "table" => &plan.table,
            "potatr" => &plan.potatr,
            other => unreachable!("unknown ONNX slot {other}"),
        };
        if let Some(list) = &pass.providers {
            return list.clone();
        }
        // `None` only happens for an auto-CUDA slot with the probe
        // withheld; this path always probes, so fall back to the base.
        self.config.providers.ort_providers.clone()
    }

    pub fn new(config: &ConverterConfig, cache_dir: &Path) -> Self {
        // `cache_dir` is the hub-cache root (0.6.0): converter models
        // resolve into its standard `models--*` layout, shared with
        // every other caller on the machine.
        Self {
            config: config.clone(),
            tex_teller: None,
            layout: None,
            ocr: None,
            table: None,
            potatr: None,
            cache_dir: cache_dir.to_path_buf(),
            layout_model_path: None,
            ocr_det_path: None,
            ocr_rec_path: None,
            table_model_path: None,
            potatr_model_path: None,
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

    /// Set an explicit POTATR page-level table ONNX model path.
    /// When unset, the model auto-downloads from HuggingFace on first use.
    pub fn set_potatr_model(&mut self, path: &Path) {
        self.potatr_model_path = Some(path.to_path_buf());
    }

    /// Load models required by the current routing mode.
    pub fn ensure_models(&mut self) -> Result<()> {
        if !self.config.routing.use_onnx {
            return Ok(());
        }
        // Shared-runtime report (plan-onnx-only Phase 3): which ORT binary
        // serves every slot, and whether its CUDA EP is usable.
        let ort = embroider::report();
        info!(
            dylib = ort.dylib_path.as_deref().unwrap_or("(loader search)"),
            cuda_usable = ort.cuda_usable,
            "ONNX Runtime: shared binary via embroider plumbing"
        );
        // Cache preflight (hub-cache alignment): one offline lookup per
        // family, logged before anything loads — a second ingest reuses
        // every blob and only re-HEADs.
        match self.model_status() {
            Ok(reps) => {
                for r in &reps {
                    info!(
                        repo = %r.repo,
                        cached = r.cached,
                        bytes = r.disk_usage_bytes,
                        "converter model cache status"
                    );
                }
            }
            Err(e) => {
                info!("converter model cache preflight failed: {e}");
            }
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

    /// Offline cache status for every converter-model family (hub-cache
    /// alignment): one [`embroider::CacheReport`] per family — TexTeller
    /// (the configured precision/quantization variant), layout, OCR
    /// (det + rec), SLANet-plus — through embroider's shared
    /// `cache_info_files` surface, the same report contract okfgraph's
    /// `model_info` emits. Missing files read `cached: false` and raise
    /// nothing; the only errors are impossible repo ids (all constants).
    pub fn model_status(&self) -> Result<Vec<embroider::CacheReport>> {
        // Variant selection mirrors ensure_tex_teller so the status
        // filenames are exactly the fetch filenames (pinned by test).
        // (Owned names hoisted: the probe borrows them.)
        let (variant_precision, variant_quantization) = self.config.formula_variant();
        let tt_owned: Option<[String; 3]> = match variant_quantization {
            crate::config::ModelQuantization::Fp32 => {
                Some(crate::tex_teller::texteller_filenames(variant_precision))
            }
            crate::config::ModelQuantization::Int8 => None,
        };
        // Pinned revision follows the variant (mirrors the fetch sites).
        let tt_rev = match &tt_owned {
            Some(_) => crate::tex_teller::TEX_TELLER_REV,
            None => crate::tex_teller::TEX_TELLER_INT8_REV,
        };
        let (tt_repo, tt_files): (&str, Vec<(&str, bool)>) = match &tt_owned {
            Some(names) => (
                crate::tex_teller::TEX_TELLER_REPO,
                names.iter().map(|s| (s.as_str(), true)).collect(),
            ),
            None => (
                crate::tex_teller::TEX_TELLER_INT8_REPO,
                crate::tex_teller::TEX_TELLER_INT8_FILES
                    .iter()
                    .map(|f| (*f, true))
                    .collect(),
            ),
        };
        let dir = Some(self.cache_dir.clone());
        let probe = |repo: &str, files: &[(&str, bool)], rev: Option<&str>| {
            embroider::cache_info_files(repo, files, rev, dir.clone())
                .map_err(|e| BobineError::Ort(e.to_string()))
        };
        let mut out = Vec::with_capacity(4);
        out.push(probe(tt_repo, &tt_files, Some(tt_rev))?);
        out.push(probe(
            &format!("{}/{}", LAYOUT_REPO.0, LAYOUT_REPO.1),
            &[(LAYOUT_FILENAME, true), (LAYOUT_LABELS_FILENAME, false)],
            Some(LAYOUT_REV),
        )?);
        out.push(probe(
            &format!("{}/{}", OCR_REPO.0, OCR_REPO.1),
            &[(OCR_DET_FILENAME, true), (OCR_REC_FILENAME, true), (OCR_DICT_FILENAME, false)],
            Some(OCR_REV),
        )?);
        out.push(probe(
            &format!(
                "{}/{}",
                crate::rapid_table::SLANET_PLUS_REPO.0,
                crate::rapid_table::SLANET_PLUS_REPO.1
            ),
            &[(crate::rapid_table::SLANET_PLUS_FILENAME, true)],
            Some(crate::rapid_table::SLANET_PLUS_REV),
        )?);
        Ok(out)
    }

    // ------------------------------------------------------------------
    // TexTeller
    // ------------------------------------------------------------------

    fn ensure_tex_teller(&mut self) -> Result<()> {
        if self.tex_teller.is_none() {
            let (variant_precision, variant_quantization) = self.config.formula_variant();
            info!(
                "Loading TexTeller ({:?}, {:?})...",
                variant_precision, variant_quantization
            );
            // The provider question is the plan's (device vocabulary and
            // the per-pass overrides already resolved).
            let enc_providers: Option<Vec<String>> = self
                .config
                .providers
                .encoder_ort_providers
                .clone()
                .or_else(|| Some(self.providers_for("encoder")));
            let dec_providers: Vec<String> = self
                .config
                .providers
                .decoder_ort_providers
                .clone()
                .unwrap_or_else(|| self.providers_for("decoder"));
            let has_cuda = enc_providers
                .iter()
                .flatten()
                .chain(dec_providers.iter())
                .any(|p| p.to_lowercase().contains("cuda"));
            // Quantized ops have no CUDA kernels - requesting CUDA with
            // Int8 weights makes ORT split the graph across devices
            // (171-267 Memcpy nodes) and runs ~1.8x SLOWER than fp32.
            if variant_quantization == crate::config::ModelQuantization::Int8 && has_cuda {
                tracing::warn!(
                    "int8 weights combined with CUDA providers: quantized ops fall back across devices (Memcpy-node overhead) and measure ~1.8x slower than fp32. Prefer precision=fp32 (or fp16) on a GPU."
                );
            }
            // fp16 has no fast CPU path: the f16 graph casts around every
            // op and measures ~3x slower than fp32 on CPU. Only relevant
            // when fp32 quantization is selected (int8 ignores precision).
            if variant_quantization == crate::config::ModelQuantization::Fp32
                && variant_precision == crate::config::ModelPrecision::Fp16
                && !has_cuda
            {
                tracing::warn!(
                    "fp16 weights without a CUDA provider: the f16 graph measures ~3x slower than fp32 on CPU (cast overhead). Prefer precision=fp32 on CPU, or precision=int8 for speed."
                );
            }
            let tt = match variant_quantization {
                crate::config::ModelQuantization::Fp32 => TexTeller::from_pretrained_split(
                    crate::tex_teller::TEX_TELLER_REPO,
                    &self.cache_dir,
                    variant_precision,
                    enc_providers.as_deref(),
                    &dec_providers,
                )?,
                crate::config::ModelQuantization::Int8 => TexTeller::from_pretrained_int8_split(
                    &self.cache_dir,
                    enc_providers.as_deref(),
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
            let providers = self.providers_for("table");
            self.table = Some(crate::rapid_table::RapidTable::load(&path, &providers)?);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // POTATR (lazy: downloads on first qualifying page)
    // ------------------------------------------------------------------

    fn ensure_potatr(&mut self) -> Result<()> {
        if self.potatr.is_none() {
            let path = match self.potatr_model_path.clone() {
                Some(p) => p,
                None => crate::rapid_potatr::download_potatr(&self.cache_dir)?,
            };
            let providers = self.providers_for("potatr");
            self.potatr = Some(crate::rapid_potatr::Potatr::load(&path, &providers)?);
        }
        Ok(())
    }

    /// Run POTATR on a full page render; returns raw network outputs (P1
    /// post-processes them into tables). Public so P2 routing can call it;
    /// unused until then (P0 wires the slot + preprocess only).
    #[allow(dead_code)]
    pub fn recognize_page_tables(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<crate::rapid_potatr::PotatrOutputs> {
        self.ensure_potatr()?;
        self.potatr
            .as_mut()
            .expect("ensured")
            .forward(img)
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
                None => hf_fetch(
                    &self.cache_dir,
                    LAYOUT_REPO,
                    LAYOUT_FILENAME,
                    "RapidLayout",
                    Some(LAYOUT_REV),
                )?,
            };
            let labels = match self.layout_model_path.clone() {
                // Local model override: labels from its own metadata.
                Some(_) => None,
                None => Some(hf_fetch(
                    &self.cache_dir,
                    LAYOUT_REPO,
                    LAYOUT_LABELS_FILENAME,
                    "RapidLayout labels",
                    Some(LAYOUT_REV),
                )?),
            };
            info!("Loading RapidLayout from {}...", path.display());
            let providers = self.providers_for("layout");
            let layout = RapidLayout::load(&path, labels.as_deref(), &providers)?;
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
                None => hf_fetch(
                    &self.cache_dir,
                    OCR_REPO,
                    OCR_DET_FILENAME,
                    "RapidOCR det",
                    Some(OCR_REV),
                )?,
            };
            let rec = match self.ocr_rec_path.clone() {
                Some(p) => p,
                None => hf_fetch(
                    &self.cache_dir,
                    OCR_REPO,
                    OCR_REC_FILENAME,
                    "RapidOCR rec",
                    Some(OCR_REV),
                )?,
            };
            let dict = match self.ocr_rec_path.clone() {
                // Local rec override: charset from its own metadata.
                Some(_) => None,
                None => Some(hf_fetch(
                    &self.cache_dir,
                    OCR_REPO,
                    OCR_DICT_FILENAME,
                    "RapidOCR dict",
                    Some(OCR_REV),
                )?),
            };
            let providers = self.providers_for("ocr");
            info!(
                "Loading RapidOCR from {} and {}...",
                det.display(),
                rec.display()
            );
            let ocr = RapidOcr::load(&det, &rec, dict.as_deref(), &self.config.models.ocr_lang, &providers)?;
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

    /// Vision slots run untuned sessions, always. If anyone wires the
    /// measured text policy (`Level3`, thread counts) into `session_builder`,
    /// this fails loudly instead of silently retuning every vision model.
    /// (plan-onnx-only Phase 4; needs no dylib — pure policy data.)
    #[test]
    fn vision_slots_never_use_text_policy() {
        let vision = vision_policy();
        assert!(vision.opt_level.is_none());
        assert!(vision.intra_threads.is_none());
        assert!(vision.inter_threads.is_none());
        // The text policy IS tuned — the pin above is what keeps it out.
        let text = embroider::SessionPolicy::text_embed();
        assert!(text.opt_level.is_some());
    }

    /// Requesting CUDA on a machine/library without it must degrade to CPU
    /// gracefully (Ok), never fail the session build.
    #[test]
    fn cuda_request_degrades_gracefully_without_gpu() {
        // load-dynamic requires a resolvable ONNX Runtime library
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            eprintln!("skipped: ORT_DYLIB_PATH not set");
            return;
        }
        let result = session_builder(&[
            "CUDAExecutionProvider".to_string(),
            "cpu".to_string(),
        ]);
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
        let result = session_builder(&[
            "warp-drive".to_string(),
            "CPUExecutionProvider".to_string(),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn table_slot_defaults_to_cpu_even_with_cuda_base() {
        // The measured pin survives a CUDA-capable base list.
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
        let plan = cfg.runtime_plan(Some(true));
        assert_eq!(plan.table.providers.as_deref().unwrap(), vec!["CPUExecutionProvider"]);
        assert_eq!(plan.table.policy, "cpu-pinned");
        assert_eq!(plan.potatr.providers.as_deref().unwrap(), vec!["CPUExecutionProvider"]);

        // An explicit override is honored, even against the default policy.
        let cfg = ConverterConfig {
            providers: ProviderOpts {
                table_ort_providers: Some(vec!["CUDAExecutionProvider".into()]),
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = cfg.runtime_plan(Some(true));
        assert_eq!(plan.table.providers.as_deref().unwrap(), vec!["CUDAExecutionProvider"]);
        assert_eq!(plan.table.policy, "explicit");
    }

    #[test]
    fn auto_gpu_slot_respects_pins_and_cuda_probe() {
        let cfg = ConverterConfig::default();
        // With CUDA registered, the auto slots take CUDA + CPU fallback.
        let plan = cfg.runtime_plan(Some(true));
        assert_eq!(
            plan.layout.providers.as_deref().unwrap(),
            vec!["CUDAExecutionProvider".to_string(), "CPUExecutionProvider".to_string()]
        );
        assert_eq!(plan.layout.policy, "auto-cuda");
        // Without it, the base list is untouched.
        let plan = cfg.runtime_plan(Some(false));
        assert_eq!(plan.layout.providers.as_deref().unwrap(), vec!["CPUExecutionProvider"]);
        assert_eq!(plan.layout.policy, "auto-cpu");
        // Before any session the answer is the policy, not a list.
        let plan = cfg.runtime_plan(None);
        assert!(plan.layout.providers.is_none());
        assert_eq!(plan.layout.policy, "auto-cuda");

        // An explicit pin always wins.
        let cfg = ConverterConfig {
            providers: ProviderOpts {
                ocr_ort_providers: Some(vec!["CPUExecutionProvider".into()]),
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = cfg.runtime_plan(Some(true));
        assert_eq!(plan.ocr.providers.as_deref().unwrap(), vec!["CPUExecutionProvider"]);
        assert_eq!(plan.ocr.policy, "explicit");
    }

    // ---- model_status: shared-surface preflight -------------------------

    #[test]
    fn model_status_empty_cache_raises_nothing() {
        // Offline-only (local_files_only underneath); the unroutable
        // endpoint would fail any accidental request loudly anyway.
        let saved = std::env::var("HF_ENDPOINT").ok();
        // SAFETY: test-only; no other thread reads HF_ENDPOINT here.
        unsafe { std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1"); }

        let dir = std::env::temp_dir().join(format!("bobine-status-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Explicit fp32 (also the default): int8 is covered below.
        let cfg = ConverterConfig {
            models: crate::config::ModelOpts {
                model_quantization: crate::config::ModelQuantization::Fp32,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = OnnxEngine::new(&cfg, &dir);
        let reps = engine.model_status().unwrap();
        assert_eq!(reps.len(), 4);
        for r in &reps {
            assert!(!r.cached);
            assert!(r.files.values().all(|p| p.is_none()));
            assert!(r.snapshot_path.is_none());
            assert_eq!(r.disk_usage_bytes, 0);
        }
        // One report per family, repos in engine order.
        let repos: Vec<&str> = reps.iter().map(|r| r.repo.as_str()).collect();
        assert_eq!(
            repos,
            [
                crate::tex_teller::TEX_TELLER_REPO,
                "opticsWolf/pp-doclayout-plus-l-onnx",
                "opticsWolf/rapidocr-ppocrv5-onnx",
                "opticsWolf/slanet-plus-onnx",
            ]
        );

        // SAFETY: test-only; restores the pre-test value.
        unsafe {
            match saved {
                Some(v) => std::env::set_var("HF_ENDPOINT", v),
                None => std::env::remove_var("HF_ENDPOINT"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_status_filenames_match_fetch_filenames() {
        // Structural pin: status names are exactly fetch names — both
        // sides read the same constants, so a rename breaks one side
        // loudly instead of silently reporting the wrong files.
        let dir = std::env::temp_dir().join(format!("bobine-status-names-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = ConverterConfig {
            models: crate::config::ModelOpts {
                model_quantization: crate::config::ModelQuantization::Fp32,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = OnnxEngine::new(&cfg, &dir);
        let reps = engine.model_status().unwrap();

        let tt = &reps[0];
        let expected: Vec<String> =
            crate::tex_teller::texteller_filenames(crate::config::ModelPrecision::Fp32)
                .into_iter()
                .collect();
        // BTreeMap iterates sorted — compare sorted for clarity.
        let mut got: Vec<&str> = tt.files.keys().map(String::as_str).collect();
        got.sort_unstable();
        let mut want: Vec<&str> = expected.iter().map(String::as_str).collect();
        want.sort_unstable();
        assert_eq!(got, want);

        let layout = &reps[1];
        assert!(layout.files.contains_key(LAYOUT_FILENAME));
        let ocr = &reps[2];
        assert!(ocr.files.contains_key(OCR_DET_FILENAME));
        assert!(ocr.files.contains_key(OCR_REC_FILENAME));
        let table = &reps[3];
        assert!(table
            .files
            .contains_key(crate::rapid_table::SLANET_PLUS_FILENAME));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_status_int8_config_selects_int8_repo() {
        let dir = std::env::temp_dir().join(format!("bobine-status-int8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = ConverterConfig {
            models: crate::config::ModelOpts {
                model_quantization: crate::config::ModelQuantization::Int8,
                ..Default::default()
            },
            ..Default::default()
        };
        let engine = OnnxEngine::new(&cfg, &dir);
        let reps = engine.model_status().unwrap();
        assert_eq!(reps[0].repo, crate::tex_teller::TEX_TELLER_INT8_REPO);
        for f in crate::tex_teller::TEX_TELLER_INT8_FILES {
            assert!(reps[0].files.contains_key(f), "missing {f}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
