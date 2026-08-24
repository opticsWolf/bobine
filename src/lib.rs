// bobine_rs — PDF / Office / text → Markdown ingestion engine (Rust core).
//
// Python bindings via PyO3 (`import bobine_rs`).

pub mod config;
pub mod converter;
pub mod engine;
pub mod error;
pub mod rapid_layout;
pub mod rapid_ocr;
pub mod tables;
pub mod tex_teller;

mod py_bindings;

pub use config::{ConverterConfig, FormulaBackend, ModelPrecision, RoutingMode};
pub use converter::HybridConverter;
pub use engine::OnnxEngine;
pub use error::BobineError;
pub use tables::html_tables_to_gfm;
pub use tex_teller::TexTeller;
