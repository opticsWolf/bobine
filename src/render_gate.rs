//! Render-health gate (Option C, `docs/cid_render_blocker.md` §2).
//!
//! Temporary, version-scoped shim for the pdf_oxide CID-renderer defect
//! (upstream [#3098](https://github.com/yfedoseev/pdf_oxide/issues/3098)):
//! on born-digital pages with a substantial text layer, OCR↔text-layer
//! agreement (word-Jaccard over alphanumeric tokens) below threshold
//! means the render is corrupt — discard the ONNX output for the page,
//! emit fast-path markdown instead, and stay loud about it.
//!
//! Scope: fires only while built against pdf_oxide < 0.3.79 (compile-time
//! `pdf_oxide_pre_0_3_79` cfg from `build.rs`, automatic expiry on the dep
//! bump) **and** `RenderHealthOpts.render_gate_enabled`. Suppression-first
//! math handling: TexTeller spans on fired pages are dropped (confident
//! false equations are worse than none); `render_gate_salvage_inline_math`
//! optionally salvages inline-only spans from the text layer as `$…$`.

use std::collections::HashSet;

/// pdf_oxide version at/above which the gate auto-expires.
pub const RENDER_GATE_CUTOFF: &str = "0.3.79";

/// Compile-time version scope: true only while built against
/// pdf_oxide < 0.3.79 (see `build.rs`).
#[allow(clippy::needless_bool)]
pub fn version_in_scope() -> bool {
    #[cfg(pdf_oxide_pre_0_3_79)]
    {
        true
    }
    #[cfg(not(pdf_oxide_pre_0_3_79))]
    {
        false
    }
}

/// Per-conversion gate counters. Read via
/// [`HybridConverter::gate_stats`](crate::converter::HybridConverter::gate_stats)
/// after [`convert_pdf`](crate::converter::HybridConverter::convert_pdf);
/// the P-D bench harness appends them to `report.md`.
#[derive(Debug, Clone, Default)]
pub struct RenderGateStats {
    /// Pages where the layer was substantial enough to judge.
    pub pages_checked: u64,
    /// Pages where the gate fired (ONNX output discarded).
    pub pages_fired: u64,
    /// Math boxes dropped by suppression-first handling.
    pub math_spans_suppressed: u64,
    /// Inline spans salvaged as `$…$` (only when salvage enabled).
    pub math_spans_salvaged: u64,
}

impl RenderGateStats {
    /// Add another page's counters (bench harness accumulates per paper).
    pub fn accumulate(&mut self, other: &RenderGateStats) {
        self.pages_checked += other.pages_checked;
        self.pages_fired += other.pages_fired;
        self.math_spans_suppressed += other.math_spans_suppressed;
        self.math_spans_salvaged += other.math_spans_salvaged;
    }

    /// One loud line for bench outputs and reports; empty when idle.
    pub fn report_line(&self) -> Option<String> {
        if self.pages_checked == 0 && self.pages_fired == 0 {
            return None;
        }
        Some(format!(
            "render-health gate (pdf_oxide < {}): checked={} fired={} math_suppressed={} math_salvaged={}",
            RENDER_GATE_CUTOFF,
            self.pages_checked,
            self.pages_fired,
            self.math_spans_suppressed,
            self.math_spans_salvaged,
        ))
    }
}

/// Gate probe outcome for one page.
#[derive(Debug)]
pub enum GateAction {
    /// Normal path (gate disabled, out of version scope, thin layer,
    /// OCR failure, or agreement above threshold).
    Pass,
    /// Corrupt render: discard ONNX output for the page. Carries
    /// opt-in salvaged inline spans (empty under suppression-first).
    Fire {
        /// Inline `$…$` spans salvaged from the text layer.
        salvaged: Vec<String>,
    },
}
/// Lowercase alphanumeric token set (order-insensitive comparison basis).
pub fn gate_tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Word-Jaccard over token sets; `None` when both are empty (no decision).
pub fn word_jaccard(a: &HashSet<String>, b: &HashSet<String>) -> Option<f32> {
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 {
        return None;
    }
    Some(inter / union)
}

/// Pure gate decision (unit-testable without models):
/// fire when the layer is substantial and agreement is below threshold.
pub fn render_gate_decision(
    layer_word_count: usize,
    jaccard: Option<f32>,
    min_layer_words: usize,
    min_jaccard: f32,
) -> bool {
    if layer_word_count < min_layer_words {
        return false;
    }
    match jaccard {
        Some(j) => j < min_jaccard,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_lowercase_alphanumeric_runs() {
        let t = gate_tokens("Hello, WORLD! x^2 123 (Eq. 1)");
        assert!(t.contains("hello"));
        assert!(t.contains("world"));
        assert!(t.contains("x"));
        assert!(t.contains("2"));
        assert!(t.contains("123"));
        assert!(t.contains("eq"));
        assert!(t.contains("1"));
        assert!(!t.contains(" "));
        assert!(!t.contains(","));
    }

    #[test]
    fn jaccard_identities() {
        let a = gate_tokens("the quick brown fox");
        assert_eq!(word_jaccard(&a, &a), Some(1.0));
        let b = gate_tokens("alpha beta gamma delta");
        let c = gate_tokens("epsilon zeta eta theta");
        assert_eq!(word_jaccard(&b, &c), Some(0.0));
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(word_jaccard(&empty, &empty), None);
        // Partial overlap: {a,b,c} vs {b,c,d} = 2/4.
        let d: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let e: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(word_jaccard(&d, &e), Some(0.5));
    }

    #[test]
    fn decision_thresholds() {
        // Thin layer: never fires, even on total disagreement.
        assert!(!render_gate_decision(5, Some(0.0), 30, 0.35));
        // Substantial layer, low agreement: fires.
        assert!(render_gate_decision(120, Some(0.05), 30, 0.35));
        // Substantial layer, healthy agreement: passes.
        assert!(!render_gate_decision(120, Some(0.8), 30, 0.35));
        // Boundary: threshold itself passes (strictly-below fires).
        assert!(!render_gate_decision(120, Some(0.35), 30, 0.35));
        assert!(render_gate_decision(120, Some(0.349), 30, 0.35));
        // No OCR tokens at all: fail-open (do not fire blind).
        assert!(!render_gate_decision(120, None, 30, 0.35));
    }

    #[test]
    fn version_scope_matches_lock() {
        // Cargo.lock pins 0.3.78 < cutoff: gate must be active in this tree.
        // (If the dep is bumped past the cutoff this test fails loudly —
        // that is the automatic expiry working; delete the gate then.)
        assert!(
            version_in_scope(),
            "pdf_oxide >= cutoff but gate still wired"
        );
    }

    #[test]
    fn stats_report_line() {
        assert!(RenderGateStats::default().report_line().is_none());
        let mut s = RenderGateStats::default();
        s.pages_checked = 10;
        s.pages_fired = 2;
        s.math_spans_suppressed = 7;
        let line = s.report_line().unwrap();
        assert!(line.contains("checked=10"));
        assert!(line.contains("fired=2"));
        assert!(line.contains("math_suppressed=7"));
    }
}
