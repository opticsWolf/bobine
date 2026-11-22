//! Optional LaTeX validation and markup rendering.
//!
//! This small adapter follows Mordant 0.10.1's math-rendering design
//! (shared `KatexContext`, strict-but-nonfatal settings, and a memoized
//! `(display, format, latex)` result) while depending only on `katex-rs`.
//! The optional feature therefore avoids Mordant's Markdown parser and
//! renderer dependencies.

#[cfg(any(feature = "math-validate", feature = "math-export"))]
use std::collections::HashMap;
#[cfg(any(feature = "math-validate", feature = "math-export"))]
use std::sync::{Arc, LazyLock, RwLock};

#[cfg(any(feature = "math-validate", feature = "math-export"))]
use katex::{KatexContext, OutputFormat, Settings, StrictMode, StrictSetting, render_to_string};

#[cfg(any(feature = "math-validate", feature = "math-export"))]
static KATEX: LazyLock<KatexContext> = LazyLock::new(KatexContext::default);

#[cfg(any(feature = "math-validate", feature = "math-export"))]
static CACHE: LazyLock<RwLock<HashMap<(bool, u8, String), Arc<str>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[cfg(any(feature = "math-validate", feature = "math-export"))]
fn format_key(format: OutputFormat) -> u8 {
    match format {
        OutputFormat::HtmlAndMathml => 0,
        OutputFormat::Html => 1,
        OutputFormat::Mathml => 2,
    }
}

#[cfg(any(feature = "math-validate", feature = "math-export"))]
fn render_cached(
    latex: &str,
    display: bool,
    output: OutputFormat,
) -> std::result::Result<Arc<str>, String> {
    let key = (display, format_key(output), latex.to_owned());
    if let Some(hit) = CACHE
        .read()
        .map_err(|_| "KaTeX render cache lock poisoned".to_string())?
        .get(&key)
    {
        return Ok(hit.clone());
    }

    let settings = Settings::builder()
        .display_mode(display)
        .output(output)
        .strict(StrictSetting::Mode(StrictMode::Warn))
        // KaTeX's default trust=false blocks unsafe commands such as href.
        .build();
    let markup = render_to_string(&KATEX, latex, &settings).map_err(|error| error.to_string())?;
    let markup: Arc<str> = Arc::from(markup);

    let mut cache = CACHE
        .write()
        .map_err(|_| "KaTeX render cache lock poisoned".to_string())?;
    // Bound the process-global cache for long-lived Bobine servers. Clearing
    // at capacity is simple and keeps cache memory independent of input volume.
    if cache.len() >= 2048 {
        cache.clear();
    }
    cache.insert(key, markup.clone());
    Ok(markup)
}

/// Validate LaTeX with the optional KaTeX engine.
#[cfg(feature = "math-validate")]
pub(crate) fn validate_latex(latex: &str, display: bool) -> std::result::Result<(), String> {
    render_cached(latex, display, OutputFormat::Mathml).map(|_| ())
}

/// Render LaTeX to HTML, MathML, or both with the optional KaTeX engine.
///
/// Available with Bobine's `math-export` feature. `format` accepts `html`,
/// `mathml`, or `both`. HTML output requires KaTeX's stylesheet and fonts in
/// the consuming document.
#[cfg(feature = "math-export")]
pub fn render_math_markup(
    latex: &str,
    display: bool,
    format: &str,
) -> std::result::Result<String, String> {
    let output = match format.to_ascii_lowercase().as_str() {
        "both" | "htmlandmathml" | "html_and_mathml" => OutputFormat::HtmlAndMathml,
        "html" => OutputFormat::Html,
        "mathml" => OutputFormat::Mathml,
        _ => return Err("format must be 'html', 'mathml', or 'both'".to_string()),
    };
    render_cached(latex, display, output).map(|markup| markup.to_string())
}
