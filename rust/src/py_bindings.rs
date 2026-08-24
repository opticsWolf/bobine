// PyO3 bindings — exposes bobine_rs to Python as `bobine_rs`.

use pyo3::prelude::*;

use crate::config::{ConverterConfig, FormulaBackend, ModelPrecision, RoutingMode};
use crate::converter::HybridConverter;

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
    ))]
    fn new(
        extract_images: bool,
        append_unreferenced_images: bool,
        use_onnx: bool,
        routing_mode: PyRoutingMode,
        _formula_backend: PyFormulaBackend,
        model_precision: PyModelPrecision,
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
                ort_providers: vec!["CPUExecutionProvider".into()],
                render_dpi, formula_dpi, detect_headings, convert_html_tables,
                detect_code_blocks, min_formula_math_chars,
                formula_inline_max_width_pts, formula_pad_pts,
                formula_layout_fallback, math_char_threshold, scanned_text_threshold,
            },
        }
    }

    #[getter] fn extract_images(&self) -> bool { self.inner.extract_images }
    #[getter] fn routing_mode(&self) -> PyRoutingMode { self.inner.routing_mode.into() }
    #[getter] fn model_precision(&self) -> PyModelPrecision {
        match self.inner.model_precision {
            ModelPrecision::Fp32 => PyModelPrecision::Fp32,
            ModelPrecision::Fp16 => PyModelPrecision::Fp16,
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRoutingMode>()?;
    m.add_class::<PyFormulaBackend>()?;
    m.add_class::<PyModelPrecision>()?;
    m.add_class::<PyConverterConfig>()?;
    m.add_class::<PyHybridConverter>()?;
    m.add_function(wrap_pyfunction!(convert_to_markdown, m)?)?;
    Ok(())
}
