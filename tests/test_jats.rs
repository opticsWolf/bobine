//! JATS adapter acceptance: corpus-wide invariants + frozen golden.
//!
//! The MathML fidelity contract lives in `tests/test_mml.rs` (692/692
//! byte-identical formulas); here we pin the document-level shape:
//! every corpus paper converts, formula counts match the MathML census,
//! dollars stay balanced, and one paper is frozen byte-for-byte.

use std::path::Path;

/// (paper id, expected formula count, expected figure count).
const CORPUS: &[(&str, usize, usize)] = &[
    ("bio-medsea-2010", 0, 14),
    ("chem-olive-2024", 2, 7),
    ("cs-bodvw-2024", 9, 6),
    ("earth-ph-2011", 0, 4),
    ("econ-tax-2019", 28, 0),
    ("eng-disk-2024", 29, 4),
    ("math-hilfer-2024", 511, 4),
    ("med-ioannidis-2005", 0, 2),
    ("physics-pauli-2024", 113, 8),
    ("psych-maia-2012", 0, 1),
];

fn convert(id: &str) -> bobine::JatsOutput {
    let xml = std::fs::read_to_string(format!("ikat-corpus/jats/{id}.xml")).unwrap();
    bobine::convert_jats_str(&xml).unwrap()
}

/// Unescaped dollars must pair up (prose dollars are `\$`-escaped).
fn dollars_balanced(md: &str) -> bool {
    let mut n = 0u32;
    let mut prev = ' ';
    for c in md.chars() {
        if c == '$' && prev != '\\' {
            n += 1;
        }
        prev = c;
    }
    n % 2 == 0
}

#[test]
fn corpus_converts_with_census_counts() {
    let mut total = 0;
    for (id, formulas, figs) in CORPUS {
        let out = convert(id);
        assert_eq!(out.n_formulas, *formulas, "{id} formula count");
        assert_eq!(out.n_figures, *figs, "{id} figure count");
        assert!(!out.title.is_empty(), "{id} has a title");
        assert!(out.markdown.starts_with("# "), "{id} starts with H1");
        assert!(dollars_balanced(&out.markdown), "{id} dollars balanced");
        total += out.n_formulas;
    }
    assert_eq!(total, 692, "corpus formula census");
}

#[test]
fn figures_link_out() {
    let out = convert("bio-medsea-2010");
    let links: Vec<&str> = out
        .markdown
        .lines()
        .filter(|l| l.starts_with("!["))
        .collect();
    assert_eq!(links.len(), 14);
    assert!(links[0].contains("info:doi/10.1371/journal.pone.0011842.g001"));
}

#[test]
fn display_equations_carry_tags() {
    let out = convert("math-hilfer-2024");
    assert!(out.markdown.contains("\\tag{1}"));
    let disp: Vec<&str> = out
        .markdown
        .lines()
        .filter(|l| l.starts_with("$$") && l.ends_with("$$"))
        .collect();
    assert!(!disp.is_empty());
}

#[test]
fn chem_olive_golden_stable() {
    let out = convert("chem-olive-2024");
    let golden =
        std::fs::read_to_string(Path::new("tests/golden/jats-chem-olive-2024.md")).unwrap();
    assert_eq!(out.markdown, golden, "golden drift — inspect before blasting");
}
