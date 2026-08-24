use thiserror::Error;

#[derive(Error, Debug)]
pub enum BobineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("pdf_oxide error: {0}")]
    PdfOxide(String),

    #[error("office_oxide error: {0}")]
    OfficeOxide(String),

    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model not available: {0}")]
    ModelNotAvailable(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BobineError>;
