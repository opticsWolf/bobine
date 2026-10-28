//! P-A.3 differential test: `mml_to_tex` byte-parity with the Python
//! reference (`ikat-corpus/mml_reference.py`, SPECIAL shimmed out) over
//! all 692 `<math>` elements extracted from the 10-paper JATS set.
//!
//! Fixtures (`tests/fixtures/mml/fNNNN.mml` + `.tex` + `manifest.json`)
//! ship in git but are excluded from the crates.io package (see
//! `Cargo.toml exclude`). Regenerate with
//! `scripts/gen_mml_fixtures.py` when `ikat-corpus/` changes.
//! No models needed — pure XML/Latex logic.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mml")
}

#[test]
fn mml_matches_python_reference_on_all_fixtures() {
    let dir = fixture_dir();
    let manifest =
        std::fs::read_to_string(dir.join("manifest.json")).expect("mml manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("manifest parses");
    let count = manifest["count"].as_u64().expect("count") as usize;
    assert!(count > 0, "no fixtures");
    assert_eq!(
        manifest["special_shimmed"],
        serde_json::Value::Bool(true),
        "fixtures must be generated with SPECIAL neutralised"
    );

    let mut stats = bobine::MmlStats::default();
    let mut compared = 0;
    for i in 0..count {
        let name = format!("f{i:04}");
        let xml = std::fs::read_to_string(dir.join(format!("{name}.mml")))
            .unwrap_or_else(|_| panic!("missing {name}.mml"));
        let expected = std::fs::read_to_string(dir.join(format!("{name}.tex")))
            .unwrap_or_else(|_| panic!("missing {name}.tex"));
        let doc =
            roxmltree::Document::parse(&xml).unwrap_or_else(|_| panic!("{name}.mml parses"));
        let got = bobine::mml_to_tex(doc.root_element(), &mut stats);
        assert_eq!(got, expected, "byte-parity on {name}");
        compared += 1;
    }
    assert_eq!(compared, count);

    // Aggregate diagnostics must match the Python reference run
    // (SPECIAL shimmed): guards against silent behavioral drift that
    // byte-parity alone could mask if both sides degraded identically.
    let mut stat_map = BTreeMap::new();
    stat_map.insert("mml_matrix".to_string(), stats.mml_matrix);
    stat_map.insert("mml_unclosed_fence".to_string(), stats.mml_unclosed_fence);
    stat_map.insert("mml_short_frac".to_string(), stats.mml_short_frac);
    assert_eq!(stat_map["mml_matrix"], 125, "mtable count drift");
    assert_eq!(
        stat_map["mml_unclosed_fence"], 18,
        "fence-fixup count drift"
    );
    assert_eq!(
        stats.mml_lost,
        vec!["U+01BB".to_string(), "U+0277".to_string(), "U+049F".to_string()],
        "shimmed glyphs must land in mml_lost"
    );
    let mut mathify_lost = stats.mathify_lost.clone();
    mathify_lost.sort();
    assert_eq!(
        mathify_lost,
        vec!["U+2010".to_string(), "U+2254".to_string(), "U+25C7".to_string()],
        "mathify losses drift"
    );
}
