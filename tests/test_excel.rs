//! Excel multi-format export over the real `types.xlsx` fixture.
//!
//! Covers [`bobine::convert_excel`] (xlsx path), [`bobine::sheets_to_csv`],
//! [`bobine::excel_to_json`], and [`bobine::sheets_to_markdown`].

use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("office")
        .join("types.xlsx")
}

#[test]
fn excel_sheets_and_headers() {
    let doc = bobine::convert_excel(&fixture()).expect("convert types.xlsx");
    let names: Vec<&str> = doc.sheets.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["Data", "Pic", "Summary"]);
    let data = &doc.sheets[0];
    assert_eq!(data.headers, vec!["Inventory — merged header", "", "", ""]);
    // Header + 2 item rows + interior empty row + Total + date/percent.
    // The empty row is preserved (row alignment matters for formulas).
    assert_eq!(data.rows.len(), 7);
    assert!(data.rows[4].iter().all(|c| c.text.is_empty()));
}

#[test]
fn excel_typed_values_and_formula() {
    let doc = bobine::convert_excel(&fixture()).expect("convert types.xlsx");
    let v = bobine::excel_to_json(&doc);
    let rows = v["sheets"][0]["rows"].as_array().unwrap();
    // Apples row: int stays int, float stays float, bool stays bool.
    assert_eq!(rows[2][0]["value"], "Apples 🍎");
    assert_eq!(rows[2][1]["value"], 12);
    assert_eq!(rows[2][2]["value"], 1.5);
    assert_eq!(rows[2][3]["value"], true);
    // Formula cell: cached text empty (writer stored none), formula kept.
    assert_eq!(rows[5][0]["text"], "Total");
    assert_eq!(rows[5][1]["formula"], "=SUM(B3:B4)");
    // Date + percent render as display text.
    assert_eq!(rows[6][0]["text"], "2024-01-01");
    assert_eq!(rows[6][1]["text"], "38%");
}

#[test]
fn excel_csv_parses_and_skips_empty_pic_sheet() {
    let doc = bobine::convert_excel(&fixture()).expect("convert types.xlsx");
    let csvs = bobine::sheets_to_csv(&doc);
    let names: Vec<&str> = csvs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["Data", "Summary"]);
    // Round-trips through a real CSV parser.
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(csvs[0].1.as_bytes());
    let records: Vec<Vec<String>> = rdr
        .records()
        .collect::<Result<Vec<csv::StringRecord>, csv::Error>>()
        .expect("csv parses")
        .iter()
        .map(|r| r.iter().map(str::to_string).collect())
        .collect();
    assert_eq!(records[1], vec!["Item", "Count", "Price", "In stock"]);
    assert_eq!(records[2][0], "Apples 🍎");
}

#[test]
fn excel_markdown_has_sheet_sections() {
    let doc = bobine::convert_excel(&fixture()).expect("convert types.xlsx");
    assert!(doc.markdown.contains("## Data"), "{}", doc.markdown);
    assert!(doc.markdown.contains("## Summary"), "{}", doc.markdown);
    assert!(doc.markdown.contains("| Item | Count | Price | In stock |"), "{}", doc.markdown);
    // Image-only sheet has no rows: noted, not a phantom table.
    assert!(doc.markdown.contains("## Pic"), "{}", doc.markdown);
    assert!(doc.markdown.contains("_(empty sheet)_"), "{}", doc.markdown);
}

#[test]
fn ingest_writes_excel_sibling_files() {
    let out = std::env::temp_dir().join("bobine_test").join("excel_ingest");
    let _ = std::fs::remove_dir_all(&out);
    let doc = bobine::pipeline::ingest_document(
        &fixture(),
        &out,
        None,
        None,
        &bobine::ProgressHooks::default(),
    )
    .expect("ingest xlsx");
    assert_eq!(doc.data_files.len(), 3, "{:?}", doc.data_files);
    for p in &doc.data_files {
        assert!(p.is_file(), "sibling exists: {}", p.display());
    }
    assert!(out.join("types.Data.csv").is_file());
    assert!(out.join("types.Summary.csv").is_file());
    assert!(out.join("types.json").is_file());
    assert!(out.join("types.md").is_file());
}

#[test]
fn convert_excel_rejects_non_workbooks() {
    let pdf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("office")
        .join("basic.docx");
    assert!(bobine::convert_excel(&pdf).is_err());
}
