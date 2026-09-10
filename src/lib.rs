// bobine_rs — PDF / Office / text → Markdown ingestion engine (Rust core).
//
// Python bindings via PyO3 (`import bobine_rs`).

pub mod assets;
pub mod config;
pub mod converter;
pub mod documents;
pub mod engine;
pub mod error;
pub mod excel;
pub mod office_images;
pub mod pdf_source;
pub mod pipeline;
pub mod rapid_layout;
pub mod rapid_ocr;
pub mod rapid_table;
pub mod tables;
pub mod tex_teller;

#[cfg(feature = "extension-module")]
mod py_bindings;

pub use config::{
    ConverterConfig, FormulaBackend, ModelOpts, ModelPrecision, ModelQuantization, ProviderOpts,
    RenderOpts, RoutingMode, RoutingOpts, TextOpts,
};
pub use converter::{HybridConverter, ProgressHooks};
pub use engine::OnnxEngine;
pub use error::BobineError;
pub use excel::{CellData, CellJson, ExcelDocument, SheetData, convert_excel, excel_to_json, sheets_to_csv, sheets_to_markdown};
pub use office_images::{OfficeImage, collect_office_images, collect_package_images, splice_office_images};
pub use pipeline::{ConvertedDocument, LintOutcome};
pub use rapid_table::RapidTable;
pub use tables::html_tables_to_gfm;
pub use tex_teller::TexTeller;
