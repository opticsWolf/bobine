//! Post-OCR LaTeX sanitizer (proposal P-C, v0.6.1).
//!
//! TexTeller's decoded LaTeX exhibits the same structural failure classes
//! as raw MathML conversion: shredded/merged commands (`\inL`), unclosed
//! stretchy fences (`\left` without `\right`), and same-level double
//! scripts (`x^a^b` — a fatal TeX group error). This module repairs the
//! first three and hard-fails the span on brace imbalance (OCR output
//! that cannot compile is worse than the text layer's loose glyphs).
//!
//! Wired into the three OCR call sites in `converter.rs` (`math_blocks`
//! hybrid + fallback, `surgical_page_markdown`), gated by
//! `TextOpts::sanitize_ocr` (default true).
//!
//! Relationship to `mml.rs` (proposal P-A): the future MathML port reuses
//! [`split_cmds`] directly — its command inventory derives from the same
//! GREEK/OPS/FUNC_NAMES tables, so the sets are identical by construction
//! (pinned by `known_set_matches_mml_reference` in mml.rs tests). The
//! fence counter here is deliberately *strict* (command-boundary aware):
//! the Python reference counts `\left` as a substring, so `\leftarrow`
//! alone gains a spurious `\right.` — faithful for the differential test,
//! wrong for OCR output. `mml.rs` carries that quirk locally with a
//! comment; this module does not.

/// Outcome of [`sanitize_latex`]: repaired LaTeX plus repair counters
/// (for debug logging and the benchmark harness).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizeOutcome {
    /// Repaired LaTeX (bare, no `$` wrapping — the caller owns math mode).
    pub latex: String,
    /// `\left`/`\right` imbalances closed with `\right.`.
    pub fences_closed: usize,
    /// Same-level double scripts separated with `{}`.
    pub scripts_split: usize,
}

/// Sanitize TexTeller OCR output.
///
/// Returns `None` when braces are imbalanced — the caller falls back to
/// the raw text layer (surgical/hybrid) or skips the span (scan fallback).
/// Otherwise returns the repaired LaTeX, which may be unchanged.
pub fn sanitize_latex(raw: &str) -> Option<SanitizeOutcome> {
    let cmds_fixed = split_cmds(raw);
    let (fenced, fences_closed) = fix_unclosed_fences(&cmds_fixed);
    match separate_consecutive_scripts(&fenced) {
        Some((latex, scripts_split)) => Some(SanitizeOutcome {
            latex,
            fences_closed,
            scripts_split,
        }),
        None => None,
    }
}

/// LaTeX command inventory for [`split_cmds`].
///
/// Derived from the `mml_reference.py` tables (GREEK + OPS + FUNC_NAMES
/// values matching `\\[a-zA-Z]+`): LaTeX base + amsmath + amssymb.
/// Sorted for binary search; `mml.rs` asserts its derived set equals this.
static KNOWN_CMDS: &[&str] = &[
    "Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Psi", "Re", "Rightarrow", "Sigma",
    "Theta", "Vert", "Xi", "aleph", "alpha", "angle", "approx", "arg", "ast", "beta",
    "bigcap", "bigcup", "bigvee", "bigwedge", "cap", "cdot", "cdots", "chi", "circ", "cong",
    "cos", "cup", "ddots", "deg", "delta", "det", "dim", "div", "ell", "emptyset", "epsilon",
    "equiv", "eta", "exists", "exp", "forall", "gamma", "geq", "hbar", "in", "inf", "infty",
    "int", "iota", "kappa", "ker", "lambda", "langle", "leftarrow", "leq", "lim", "ln",
    "log", "max", "min", "mu", "nabla", "neg", "neq", "ni", "notin", "nu", "omega", "oplus",
    "otimes", "partial", "perp", "phi", "pi", "pm", "prod", "propto", "psi", "quad",
    "rangle", "rho", "sigma", "sim", "simeq", "sin", "subset", "subseteq", "sum", "sup",
    "supset", "supseteq", "tan", "tau", "theta", "times", "to", "upsilon", "varepsilon",
    "varphi", "varpi", "varsigma", "vartheta", "vdots", "vee", "wedge", "wp", "xi",
];

fn is_known_cmd(name: &str) -> bool {
    KNOWN_CMDS.binary_search(&name).is_ok()
}

/// Split `\inL` -> `\in L`, leave `\infty}` alone.
///
/// TeX reads the longest alpha run after `\`, so `\infty}` is one command
/// (never split) while publisher/OCR output `\inL` is undefined — split it
/// after the longest KNOWN prefix. A regex alternation cannot do this: when
/// the longest fails it backtracks and shreds (`\infty` -> `\inf ty`).
/// Port of `mml_reference._split_cmds`.
pub fn split_cmds(tex: &str) -> String {
    let bytes = tex.as_bytes();
    let n = bytes.len();
    let mut out = String::with_capacity(tex.len() + 8);
    let mut i = 0;
    while i < n {
        let is_cmd_start = bytes[i] == b'\\'
            && i + 1 < n
            && (bytes[i + 1] as char).is_ascii_alphabetic();
        if !is_cmd_start {
            // Copy one UTF-8 char (backslash escapes like `\{` stay intact).
            let ch = tex[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        let mut j = i + 1;
        while j < n && (bytes[j] as char).is_ascii_alphabetic() {
            j += 1;
        }
        let run = &tex[i + 1..j];
        if is_known_cmd(run) {
            out.push_str(&tex[i..j]);
            i = j;
            continue;
        }
        let mut k = run.len();
        // `run` is ASCII alpha by construction, so byte steps are safe.
        while k > 0 && !is_known_cmd(&run[..k]) {
            k -= 1;
        }
        if k > 0 {
            out.push_str(&tex[i..i + 1 + k]);
            out.push(' ');
            i += 1 + k;
        } else {
            out.push_str(&tex[i..j]);
            i = j;
        }
    }
    out
}

/// Count `\left`-style opens minus `\right`-style closes, appending
/// `\right.` per stray open. Strict: only counts the fence commands at a
/// command boundary (`\left(`, `\left\{`, `\left.`), never the `\left`
/// prefix inside `\leftarrow` (unlike the Python reference's substring
/// count — see module docs).
pub fn fix_unclosed_fences(tex: &str) -> (String, usize) {
    let mut opens = 0usize;
    let mut closes = 0usize;
    let bytes = tex.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if bytes[i] == b'\\' {
            if tex[i..].starts_with("\\left")
                && (i + 5 >= n || !(bytes[i + 5] as char).is_ascii_alphabetic())
            {
                opens += 1;
            } else if tex[i..].starts_with("\\right")
                && (i + 6 >= n || !(bytes[i + 6] as char).is_ascii_alphabetic())
            {
                closes += 1;
            }
        }
        let ch = tex[i..].chars().next().unwrap();
        i += ch.len_utf8();
    }
    if opens > closes {
        let missing = opens - closes;
        let mut fixed = String::with_capacity(tex.len() + missing * 8);
        fixed.push_str(tex);
        for _ in 0..missing {
            fixed.push_str(" \\right.");
        }
        (fixed, missing)
    } else {
        (tex.to_string(), 0)
    }
}

/// Characters that start a new TeX nucleus at the same brace depth: a
/// script operator seen after one of these belongs to fresh math, not to
/// the previous scripts. `(`/`)`/`[`/`]` count because `(x^a)^b` attaches
/// the outer sup to the parenthesized group — legal, no separator wanted.
fn is_nucleus_boundary(ch: char) -> bool {
    matches!(
        ch,
        '+' | '-' | '=' | '<' | '>' | '(' | ')' | '[' | ']' | ',' | ';' | '|' | '/' | '&'
            | ' ' | '\t' | '\n' | '\r' | '$'
    )
}

/// Script attachments at the same brace depth that TeX forbids doubling:
/// `x^a^b` -> `x^a{}^b`, `x_a_b` -> `x_a{}_b`. Mixed levels (`x^a_b`)
/// pass through. Braces scope the tracking (`\frac{a^1}{b^2}` untouched),
/// escaped braces (`\{`) never touch the depth count.
///
/// Returns `None` on brace imbalance (caller hard-fails the span).
pub fn separate_consecutive_scripts(tex: &str) -> Option<(String, usize)> {
    // Per-brace-depth memory: has this nucleus already taken `^` / `_`?
    let mut seen: Vec<(bool, bool)> = vec![(false, false)];
    let mut out = String::with_capacity(tex.len() + 8);
    let mut splits = 0usize;
    let n = tex.len();
    let mut i = 0;
    while i < n {
        let ch = tex[i..].chars().next().unwrap();
        if ch == '\\' {
            // Copy command/escape verbatim; letters are atom content,
            // never boundaries or depth changes.
            out.push(ch);
            i += 1;
            if i < n {
                let next = tex[i..].chars().next().unwrap();
                out.push(next);
                i += next.len_utf8();
                if next.is_ascii_alphabetic() {
                    while i < n {
                        let c = tex[i..].chars().next().unwrap();
                        if !c.is_ascii_alphabetic() {
                            break;
                        }
                        out.push(c);
                        i += c.len_utf8();
                    }
                }
            }
            continue;
        }
        match ch {
            '{' => {
                seen.push((false, false));
                out.push(ch);
                i += 1;
            }
            '}' => {
                if seen.len() <= 1 {
                    return None; // stray close
                }
                seen.pop();
                out.push(ch);
                i += 1;
            }
            '^' | '_' => {
                let frame = seen.last_mut().unwrap();
                let is_caret = ch == '^';
                let already = if is_caret { frame.0 } else { frame.1 };
                if already {
                    out.push_str("{}");
                    splits += 1;
                }
                if is_caret {
                    frame.0 = true;
                } else {
                    frame.1 = true;
                }
                out.push(ch);
                i += 1;
            }
            _ => {
                if is_nucleus_boundary(ch) {
                    let frame = seen.last_mut().unwrap();
                    *frame = (false, false);
                }
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    if seen.len() != 1 {
        return None; // unclosed open
    }
    Some((out, splits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_table_is_sorted_for_binary_search() {
        let mut sorted = KNOWN_CMDS.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, KNOWN_CMDS);
        assert_eq!(KNOWN_CMDS.len(), 112);
    }

    #[test]
    fn split_cmds_splits_unknown_prefix_keeps_known() {
        assert_eq!(split_cmds(r"\inL"), r"\in L");
        assert_eq!(split_cmds(r"\infty}"), r"\infty}");
        assert_eq!(split_cmds(r"\alpha\beta"), r"\alpha\beta");
        assert_eq!(split_cmds(r"x + y"), "x + y");
        // Unknown with no known prefix passes through.
        assert_eq!(split_cmds(r"\zxq"), r"\zxq");
    }

    #[test]
    fn fence_fixup_closes_strays_ignores_arrows() {
        let (fixed, n) = fix_unclosed_fences(r"\left( x");
        assert_eq!(n, 1);
        assert!(fixed.ends_with(r"\right."));
        let (same, n) = fix_unclosed_fences(r"\left( x \right)");
        assert_eq!(n, 0);
        assert_eq!(same, r"\left( x \right)");
        // Strict: \leftarrow / \rightarrow are not fences.
        let (arrow, n) = fix_unclosed_fences(r"\leftarrow x");
        assert_eq!(n, 0);
        assert_eq!(arrow, r"\leftarrow x");
    }

    #[test]
    fn script_separation_splits_same_level_only() {
        assert_eq!(separate_consecutive_scripts("x^a^b"), Some(("x^a{}^b".into(), 1)));
        assert_eq!(separate_consecutive_scripts("x_a_b"), Some(("x_a{}_b".into(), 1)));
        assert_eq!(separate_consecutive_scripts("x^a_b"), Some(("x^a_b".into(), 0)));
        assert_eq!(
            separate_consecutive_scripts("x^{a}^{b}"),
            Some(("x^{a}{}^{b}".into(), 1))
        );
        assert_eq!(
            separate_consecutive_scripts(r"\frac{a^1}{b^2}"),
            Some((r"\frac{a^1}{b^2}".into(), 0))
        );
        assert_eq!(separate_consecutive_scripts("(x^a)^b"), Some(("(x^a)^b".into(), 0)));
        assert_eq!(separate_consecutive_scripts("x^a_b^c"), Some(("x^a_b{}^c".into(), 1)));
    }

    #[test]
    fn brace_imbalance_fails_closed() {
        assert_eq!(separate_consecutive_scripts("x^{a"), None);
        assert_eq!(separate_consecutive_scripts("x^a}"), None);
        assert!(sanitize_latex(r"\frac{a}{b}").is_some());
        assert!(sanitize_latex(r"\left( x^{a}").is_some());
        assert!(sanitize_latex("x^{a").is_none());
    }

    #[test]
    fn full_sanitize_repairs_triple_failure() {
        let out = sanitize_latex(r"\left( \inL + x^a^b").unwrap();
        assert_eq!(out.fences_closed, 1);
        assert_eq!(out.scripts_split, 1);
        assert!(out.latex.contains(r"\in L"));
        assert!(out.latex.contains("x^a{}^b"));
        assert!(out.latex.ends_with(r"\right."));
    }
}
