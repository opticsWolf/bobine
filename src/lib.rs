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
pub mod hub_cache;
pub mod input_limits;
pub mod jats;
pub mod math_render;
pub mod math_source;
pub mod mml;
pub mod office_images;
mod pdf_math_raw;
pub mod pdf_source;
pub mod pipeline;
pub mod rapid_layout;
pub mod rapid_ocr;
pub mod rapid_potatr;
pub mod rapid_table;
pub mod render_gate;
pub mod runtime;
pub mod sanitize;
pub mod tables;
pub mod tex_teller;

#[cfg(feature = "extension-module")]
mod py_bindings;

pub use config::{
    ConverterConfig, FormulaBackend, ImageMode, MathOpts, ModelOpts, ModelPrecision,
    ModelQuantization, ProviderOpts, RenderHealthOpts, RenderOpts, RoutingMode, RoutingOpts,
    TableBackend, TextOpts,
};
pub use converter::{HybridConverter, ProgressHooks};
pub use engine::OnnxEngine;
pub use error::BobineError;
pub use excel::{
    CellData, CellJson, ExcelDocument, SheetData, convert_excel, excel_to_json, sheets_to_csv,
    sheets_to_markdown,
};
pub use input_limits::MAX_INPUT_FILE_BYTES;
pub use jats::{JatsDocument, JatsOutput, convert_jats_str};
#[cfg(feature = "math-export")]
pub use math_render::render_math_markup;
pub use math_source::{EmbeddedFile, MathOrigin, MathSourceReport, MathSpan};
pub use mml::{MmlStats, mathify, merge_math, mml_to_tex, split_cmds as mml_split_cmds};
pub use office_images::{
    OfficeImage, collect_office_images, collect_package_images, splice_office_images,
};
pub use pdf_source::PdfDocumentSource;
pub use pipeline::{ConvertedDocument, LintOutcome};
pub use rapid_table::RapidTable;
pub use render_gate::{GateAction, RENDER_GATE_CUTOFF, RenderGateStats};
pub use sanitize::{SanitizeOutcome, sanitize_latex};
pub use tables::html_tables_to_gfm;
pub use tex_teller::TexTeller;
