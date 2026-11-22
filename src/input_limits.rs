//! Shared input-size guard for path-based Bobine document conversion.

use std::path::Path;

use crate::error::{BobineError, Result};

/// Maximum path-input file length accepted by Bobine: 256 MiB.
///
/// This measures source bytes, not decoded PDF streams or expanded archives.
pub const MAX_INPUT_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Reject an input whose byte length exceeds Bobine's global document limit.
pub(crate) fn ensure_input_file_size(path: &Path) -> Result<()> {
    let size_bytes = std::fs::metadata(path)?.len();
    ensure_input_size(path.display().to_string(), size_bytes)
}

/// Apply the same limit to in-memory document inputs such as JATS XML.
pub(crate) fn ensure_input_bytes_size(input: impl Into<String>, size_bytes: u64) -> Result<()> {
    ensure_input_size(input.into(), size_bytes)
}

fn ensure_input_size(input: String, size_bytes: u64) -> Result<()> {
    if size_bytes > MAX_INPUT_FILE_BYTES {
        return Err(BobineError::InputFileTooLarge {
            input,
            size_bytes,
            limit_bytes: MAX_INPUT_FILE_BYTES,
        });
    }
    Ok(())
}
