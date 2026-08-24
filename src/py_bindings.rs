// PyO3 bindings — exposes bobine to Python as `bobine._native`.

use pyo3::prelude::*;

use crate::config::{ConverterConfig, FormulaBackend, ModelPrecision, ModelQuantization, RoutingMode};
use crate::converter::{HybridConverter, ProgressHooks};

#[pyclass(eq, eq_int, name = "RoutingMode")]
#[derive(Clone, PartialEq)]
pub enum PyRoutingMode { Never, Auto, Surgical, Always }

impl From<PyRoutingMode> for RoutingMode {
    fn from(py: PyRoutingMode) -> Self {
        match py {
            PyRoutingMode::Never => RoutingMode::Never,
            PyRoutingMode::Auto => RoutingMode::Auto,
            PyRoutingMode::Surgical => RoutingMode::Surgical,
            PyRoutingMode::Always => RoutingMode::Always,
        }
    }
}

impl From<RoutingMode> for PyRoutingMode {
    fn from(r: RoutingMode) -> Self {
        match r {
            RoutingMode::Never => PyRoutingMode::Never,
            RoutingMode::Auto => PyRoutingMode::Auto,
            RoutingMode::Surgical => PyRoutingMode::Surgical,
            RoutingMode::Always => PyRoutingMode::Always,
        }
    }
}

#[pyclass(eq, eq_int, name = "FormulaBackend")]
#[derive(Clone, PartialEq)]
pub enum PyFormulaBackend { TexTeller }

#[pyclass(eq, eq_int, name = "ModelPrecision")]
#[derive(Clone, PartialEq)]
pub enum PyModelPrecision { Fp32, Fp16 }

/// Formula-recognizer weight selection: `Fp32` (accurate, default) or
/// `Int8` (compact memory footprint, minor typographic drift possible).
#[pyclass(eq, eq_int, name = "ModelQuantization")]
#[derive(Clone, PartialEq)]
pub enum PyModelQuantization { Fp32, Int8 }

#[pyclass(name = "ConverterConfig")]
#[derive(Clone)]
pub struct PyConverterConfig { inner: ConverterConfig }

#[pymethods]
impl PyConverterConfig {
    #[new]
    #[pyo3(signature = (
        extract_images = true,
        append_unreferenced_images = true,
        use_onnx = true,
        routing_mode = PyRoutingMode::Auto,
        _formula_backend = PyFormulaBackend::TexTeller,
        model_precision = PyModelPrecision::Fp32,
        model_quantization = PyModelQuantization::Fp32,
        render_dpi = 300u32,
        formula_dpi = 200u32,
        detect_headings = true,
        convert_html_tables = true,
        detect_code_blocks = true,
        min_formula_math_chars = 5usize,
        formula_inline_max_width_pts = 220.0f64,
        formula_pad_pts = 4.0f64,
        formula_layout_fallback = false,
        math_char_threshold = 30usize,
        scanned_text_threshold = 50usize,
        ocr_lang = "en".to_string(),
    ))]
    fn new(
        extract_images: bool,
        append_unreferenced_images: bool,
        use_onnx: bool,
        routing_mode: PyRoutingMode,
        _formula_backend: PyFormulaBackend,
        model_precision: PyModelPrecision,
        model_quantization: PyModelQuantization,
        render_dpi: u32,
        formula_dpi: u32,
        detect_headings: bool,
        convert_html_tables: bool,
        detect_code_blocks: bool,
        min_formula_math_chars: usize,
        formula_inline_max_width_pts: f64,
        formula_pad_pts: f64,
        formula_layout_fallback: bool,
        math_char_threshold: usize,
        scanned_text_threshold: usize,
        ocr_lang: String,
    ) -> Self {
        PyConverterConfig {
            inner: ConverterConfig {
                extract_images, append_unreferenced_images, use_onnx,
                routing_mode: routing_mode.into(),
                formula_backend: FormulaBackend::TexTeller,
                model_precision: match model_precision {
                    PyModelPrecision::Fp32 => ModelPrecision::Fp32,
                    PyModelPrecision::Fp16 => ModelPrecision::Fp16,
                },
                model_quantization: match model_quantization {
                    PyModelQuantization::Fp32 => ModelQuantization::Fp32,
                    PyModelQuantization::Int8 => ModelQuantization::Int8,
                },
                ort_providers: vec!["CPUExecutionProvider".into()],
                render_dpi, formula_dpi, detect_headings, convert_html_tables,
                detect_code_blocks, min_formula_math_chars,
                formula_inline_max_width_pts, formula_pad_pts,
                formula_layout_fallback, math_char_threshold, scanned_text_threshold,
                ocr_lang,
            },
        }
    }

    #[getter] fn extract_images(&self) -> bool { self.inner.extract_images }
    #[getter] fn routing_mode(&self) -> PyRoutingMode { self.inner.routing_mode.into() }
    #[getter] fn ocr_lang(&self) -> String { self.inner.ocr_lang.clone() }
    #[getter] fn model_precision(&self) -> PyModelPrecision {
        match self.inner.model_precision {
            ModelPrecision::Fp32 => PyModelPrecision::Fp32,
            ModelPrecision::Fp16 => PyModelPrecision::Fp16,
        }
    }
    #[getter] fn model_quantization(&self) -> PyModelQuantization {
        match self.inner.model_quantization {
            ModelQuantization::Fp32 => PyModelQuantization::Fp32,
            ModelQuantization::Int8 => PyModelQuantization::Int8,
        }
    }
    fn __repr__(&self) -> String {
        format!("ConverterConfig(routing={:?}, precision={:?})", self.inner.routing_mode, self.inner.model_precision)
    }
}

#[pyclass(name = "HybridConverter")]
pub struct PyHybridConverter { inner: HybridConverter }

#[pymethods]
impl PyHybridConverter {
    #[new]
    fn new(config: &PyConverterConfig, cache_dir: &str) -> PyResult<Self> {
        let inner = HybridConverter::new(config.inner.clone(), std::path::Path::new(cache_dir));
        Ok(PyHybridConverter { inner })
    }

    fn convert(&mut self, input: &str, work_dir: &str) -> PyResult<String> {
        self.inner.convert(std::path::Path::new(input), std::path::Path::new(work_dir))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn convert_pdf(&mut self, path: &str, work_dir: &str) -> PyResult<String> {
        self.inner.convert_pdf(std::path::Path::new(path), std::path::Path::new(work_dir))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn recognize_formula(&mut self, image_path: &str) -> PyResult<Option<String>> {
        self.inner.engine.recognize_formula(std::path::Path::new(image_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String { "HybridConverter(...)".into() }
}

#[pyfunction]
fn convert_to_markdown(path: &str, config: &PyConverterConfig, work_dir: &str, cache_dir: &str) -> PyResult<String> {
    let mut converter = HybridConverter::new(config.inner.clone(), std::path::Path::new(cache_dir));
    converter.convert(std::path::Path::new(path), std::path::Path::new(work_dir))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

// -- pipeline layer ---------------------------------------------------------

use crate::pipeline::{self, ConvertedDocument, LintOutcome};

#[pyclass(name = "ConvertedDocument")]
pub struct PyConvertedDocument { inner: ConvertedDocument }

#[pymethods]
impl PyConvertedDocument {
    #[getter] fn md_path(&self) -> String { self.inner.md_path.display().to_string() }
    #[getter] fn md_text(&self) -> String { self.inner.md_text.clone() }
    #[getter] fn image_dir(&self) -> String { self.inner.image_dir.display().to_string() }
    #[getter] fn image_count(&self) -> usize { self.inner.image_count }
    #[getter] fn page_count(&self) -> usize { self.inner.page_count }

    fn __repr__(&self) -> String {
        format!(
            "ConvertedDocument(md_path={}, pages={}, images={})",
            self.inner.md_path.display(), self.inner.page_count, self.inner.image_count
        )
    }
}

fn to_py_err(e: crate::error::BobineError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Wrap optional Python callables into pipeline hooks.
struct PyCallbacks<'py> {
    lint: Option<Bound<'py, PyAny>>,
    should_continue: Option<Bound<'py, PyAny>>,
    on_page: Option<Bound<'py, PyAny>>,
}

#[pyfunction]
#[pyo3(signature = (path, output_dir, config=None, lint_callback=None, should_continue=None, on_page=None))]
fn ingest_document<'py>(
    py: Python<'py>,
    path: &str,
    output_dir: &str,
    config: Option<&PyConverterConfig>,
    lint_callback: Option<Bound<'py, PyAny>>,
    should_continue: Option<Bound<'py, PyAny>>,
    on_page: Option<Bound<'py, PyAny>>,
) -> PyResult<PyConvertedDocument> {
    let cbs = PyCallbacks { lint: lint_callback, should_continue, on_page };

    let lint_fn = cbs.lint.as_ref().map(|f| {
        let f = f.clone();
        move |md: &str| -> crate::error::Result<LintOutcome> {
            let res = f.call1((md,)).map_err(|e| crate::error::BobineError::Other(e.to_string()))?;
            if res.is_none() {
                return Ok(LintOutcome::default());
            }
            let tuple: (bool, String) = res
                .extract()
                .map_err(|e| crate::error::BobineError::Other(format!("lint callback must return None or (fixed, content): {e}")))?;
            Ok(LintOutcome { fixed: tuple.0, content: tuple.1 })
        }
    });

    let hooks = ProgressHooks {
        should_continue: match &cbs.should_continue {
            Some(f) => {
                let f = f.clone();
                Box::new(move || f.call0().and_then(|b| b.extract::<bool>()).unwrap_or(true))
            }
            None => Box::new(|| true),
        },
        on_page: match &cbs.on_page {
            Some(f) => {
                let f = f.clone();
                Box::new(move |i: usize, n: usize| {
                    let _ = f.call1((i, n));
                })
            }
            None => Box::new(|_, _| {}),
        },
    };

    // LintFn is a reference type; keep the boxed closure alive in this scope.
    let owned_lint = lint_fn;
    let doc = pipeline::ingest_document(
        std::path::Path::new(path),
        std::path::Path::new(output_dir),
        config.map(|c| &c.inner),
        owned_lint.as_ref().map(|f| f as &crate::pipeline::LintFn),
        &hooks,
    )
    .map_err(to_py_err)?;
    Ok(PyConvertedDocument { inner: doc })
}

#[pyfunction]
#[pyo3(signature = (source_dir, output_dir, config=None, lint_callback=None))]
fn convert_directory(
    py: Python<'_>,
    source_dir: &str,
    output_dir: &str,
    config: Option<&PyConverterConfig>,
    lint_callback: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<PyConvertedDocument>> {
    let lint_fn = lint_callback.map(|f| {
        move |md: &str| -> crate::error::Result<LintOutcome> {
            let res = f.call1((md,)).map_err(|e| crate::error::BobineError::Other(e.to_string()))?;
            if res.is_none() {
                return Ok(LintOutcome::default());
            }
            let tuple: (bool, String) = res
                .extract()
                .map_err(|e| crate::error::BobineError::Other(format!("lint callback must return None or (fixed, content): {e}")))?;
            Ok(LintOutcome { fixed: tuple.0, content: tuple.1 })
        }
    });

    let docs = pipeline::convert_directory(
            std::path::Path::new(source_dir),
            std::path::Path::new(output_dir),
            config.map(|c| &c.inner),
            lint_fn.as_ref().map(|f| f as &crate::pipeline::LintFn),
        )
    .map_err(to_py_err)?;
    Ok(docs.into_iter().map(|inner| PyConvertedDocument { inner }).collect())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRoutingMode>()?;
    m.add_class::<PyFormulaBackend>()?;
    m.add_class::<PyModelPrecision>()?;
    m.add_class::<PyConverterConfig>()?;
    m.add_class::<PyHybridConverter>()?;
    m.add_function(wrap_pyfunction!(convert_to_markdown, m)?)?;
    m.add_class::<PyConvertedDocument>()?;
    m.add_function(wrap_pyfunction!(ingest_document, m)?)?;
    m.add_function(wrap_pyfunction!(convert_directory, m)?)?;
    Ok(())
}
