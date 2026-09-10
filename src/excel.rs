//! Excel workbook export: markdown, per-sheet CSV, typed JSON.
//!
//! `convert_office` returns one opaque markdown string. Spreadsheets deserve
//! better: [`convert_excel`] opens `.xlsx`/`.xls` workbooks into an
//! [`ExcelDocument`] — one [`SheetData`] per worksheet with display text,
//! typed values, and formulas — plus uniform renderers ([`sheets_to_csv`],
//! [`excel_to_json`], [`sheets_to_markdown`]).
//!
//! Design notes (see IMPLEMENTATION_PLAN_office.md Phase 2):
//! - Display text reuses office_oxide formatting (`format_cell_value` for
//!   xlsx, `as_text` for xls): dates, percents, and currency render as seen.
//! - Numbers that are exact integers (and fit in i64) map to JSON integers,
//!   so `12` stays `12`, not `12.0`. Non-finite floats become `null`.
//! - Formulas are never evaluated: `text` is the cached display value
//!   (possibly empty when the writer stored none), `formula` carries the
//!   expression with a leading `=`.
//! - Errors (`#DIV/0!`, …) keep their display text; the JSON value is `null`.
//! - Rows are rectangularized to the sheet's max width with empty cells;
//!   fully-empty trailing rows are trimmed. Merged cells surface as the
//!   top-left value, rest empty (writer/reader convention).

use crate::error::{BobineError, Result};

/// Typed cell value for JSON export.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellJson {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    Text(String),
}

/// One cell: display text, typed value, optional formula.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CellData {
    /// Display text (number/date formatting applied).
    pub text: String,
    /// Typed value (`Null` for empty and error cells).
    pub value: CellJson,
    /// Formula expression with leading `=`, when the cell has one.
    pub formula: Option<String>,
}

impl CellData {
    fn empty() -> Self {
        Self {
            text: String::new(),
            value: CellJson::Null,
            formula: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty() && self.formula.is_none() && self.value == CellJson::Null
    }
}

/// One worksheet: rectangular grid plus the header row.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SheetData {
    /// Worksheet display name.
    pub name: String,
    /// Display texts of the first non-empty row (may be empty).
    pub headers: Vec<String>,
    /// Data rows (header row included as `rows[0]` when present).
    pub rows: Vec<Vec<CellData>>,
}

impl SheetData {
    fn new(name: String, mut rows: Vec<Vec<CellData>>) -> Self {
        trim_empty_trailing_rows(&mut rows);
        rectangularize(&mut rows);
        let headers = rows
            .iter()
            .find(|r| r.iter().any(|c| !c.is_empty()))
            .map(|r| r.iter().map(|c| c.text.clone()).collect())
            .unwrap_or_default();
        Self { name, headers, rows }
    }

    fn is_empty(&self) -> bool {
        self.rows.iter().all(|r| r.iter().all(CellData::is_empty))
    }
}

/// A workbook: per-sheet grids plus the assembled markdown.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ExcelDocument {
    pub sheets: Vec<SheetData>,
    /// `## {sheet}` sections with one GFM table each (empty sheets noted).
    pub markdown: String,
}

/// Open an `.xlsx`/`.xls` workbook into an [`ExcelDocument`].
pub fn convert_excel(path: &std::path::Path) -> Result<ExcelDocument> {
    use office_oxide::Document;
    let doc = Document::open(path).map_err(|e| BobineError::OfficeOxide(format!("open: {e}")))?;
    if let Some(xlsx) = doc.as_xlsx() {
        Ok(from_xlsx(xlsx))
    } else if let Some(xls) = doc.as_xls() {
        Ok(from_xls(xls))
    } else {
        Err(BobineError::OfficeOxide(format!(
            "convert_excel: not a workbook: {}",
            path.display()
        )))
    }
}

fn from_xlsx(doc: &office_oxide::xlsx::XlsxDocument) -> ExcelDocument {
    let sheets: Vec<SheetData> = doc
        .workbook
        .sheets
        .iter()
        .enumerate()
        .map(|(i, info)| {
            let grid = doc
                .worksheets
                .get(i)
                .map(|ws| {
                    // Sparse cells keyed by reference; place by (row, col).
                    let mut placed: Vec<Vec<CellData>> = Vec::new();
                    for row in &ws.rows {
                        for cell in &row.cells {
                            let (r, c) = (cell.reference.row as usize, cell.reference.col as usize);
                            while placed.len() <= r {
                                placed.push(Vec::new());
                            }
                            while placed[r].len() <= c {
                                placed[r].push(CellData::empty());
                            }
                            let text = doc.format_cell_value(cell);
                            placed[r][c] = CellData {
                                text,
                                value: xlsx_value(&cell.value, &doc.shared_strings),
                                formula: cell.formula.as_ref().map(|f| format!("={f}")),
                            };
                        }
                    }
                    placed
                })
                .unwrap_or_default();
            SheetData::new(info.name.clone(), grid)
        })
        .collect();
    let markdown = sheets_to_markdown(&sheets);
    ExcelDocument { sheets, markdown }
}

fn xlsx_value(
    v: &office_oxide::xlsx::CellValue,
    sst: &office_oxide::xlsx::SharedStringTable,
) -> CellJson {
    use office_oxide::xlsx::CellValue;
    match v {
        CellValue::Empty => CellJson::Null,
        CellValue::Number(n) => number_json(*n),
        CellValue::String(s) => CellJson::Text(s.clone()),
        CellValue::SharedString(i) => CellJson::Text(sst.get(*i).unwrap_or("").to_string()),
        CellValue::Boolean(b) => CellJson::Bool(*b),
        CellValue::Error(_) => CellJson::Null,
        CellValue::Date(dt) => CellJson::Text(dt.to_iso_string()),
    }
}

fn from_xls(doc: &office_oxide::xls::XlsDocument) -> ExcelDocument {
    use office_oxide::xls::CellValue;
    let sheets: Vec<SheetData> = doc
        .sheets
        .iter()
        .map(|sheet| {
            let grid = sheet
                .rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| CellData {
                            text: v.as_text(),
                            value: match v {
                                CellValue::Empty => CellJson::Null,
                                CellValue::Number(n) => number_json(*n),
                                CellValue::String(s) => CellJson::Text(s.clone()),
                                CellValue::Bool(b) => CellJson::Bool(*b),
                                CellValue::Error(_) => CellJson::Null,
                            },
                            formula: None, // CFB reader exposes cached values only
                        })
                        .collect()
                })
                .collect();
            SheetData::new(sheet.name.clone(), grid)
        })
        .collect();
    let markdown = sheets_to_markdown(&sheets);
    ExcelDocument { sheets, markdown }
}

fn number_json(n: f64) -> CellJson {
    if !n.is_finite() {
        CellJson::Null
    } else if n.fract() == 0.0 && n.abs() < 9.0e15 {
        // `as i64` is exact in this range.
        CellJson::Integer(n as i64)
    } else {
        CellJson::Float(n)
    }
}

fn trim_empty_trailing_rows(rows: &mut Vec<Vec<CellData>>) {
    while rows.last().is_some_and(|r| r.iter().all(CellData::is_empty)) {
        rows.pop();
    }
}

fn rectangularize(rows: &mut Vec<Vec<CellData>>) {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    for row in rows.iter_mut() {
        while row.len() < width {
            row.push(CellData::empty());
        }
    }
}

/// Render every non-empty sheet as `## {name}` + GFM table.
pub fn sheets_to_markdown(sheets: &[SheetData]) -> String {
    let mut out = String::new();
    for sheet in sheets {
        out.push_str("## ");
        out.push_str(&sheet.name);
        out.push_str("\n\n");
        let rows: Vec<&Vec<CellData>> =
            sheets_nonempty_rows(sheet).collect();
        if rows.is_empty() {
            out.push_str("_(empty sheet)_\n\n");
            continue;
        }
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
        out.push_str(&md_row(&rows[0].iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), width));
        out.push_str(&md_sep(width));
        for row in &rows[1..] {
            out.push_str(&md_row(&row.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), width));
        }
        out.push('\n');
    }
    out
}

/// Rows with at least one non-empty cell (fully-empty rows carry no signal).
fn sheets_nonempty_rows(sheet: &SheetData) -> impl Iterator<Item = &Vec<CellData>> {
    sheet.rows.iter().filter(|r| r.iter().any(|c| !c.is_empty()))
}

fn md_row(cells: &[&str], width: usize) -> String {
    let mut s = String::from("|");
    for i in 0..width {
        s.push(' ');
        s.push_str(cells.get(i).copied().unwrap_or(""));
        s.push_str(" |");
    }
    s.push('\n');
    s
}

fn md_sep(width: usize) -> String {
    "|".to_string() + &" --- |".repeat(width) + "\n"
}

/// Render every non-empty sheet to CSV: one `(sheet_name, csv_text)` pair.
/// Empty sheets are skipped (no rows → no CSV).
pub fn sheets_to_csv(doc: &ExcelDocument) -> Vec<(String, String)> {
    doc.sheets
        .iter()
        .filter(|s| sheets_nonempty_rows(s).next().is_some())
        .map(|sheet| {
            let mut buf = Vec::new();
            {
                let mut w = csv::Writer::from_writer(&mut buf);
                for row in sheets_nonempty_rows(sheet) {
                    let record: Vec<&str> =
                        row.iter().map(|c| c.text.as_str()).collect();
                    w.write_record(&record).expect("csv write to vec");
                }
                w.flush().expect("csv flush to vec");
            }
            (sheet.name.clone(), String::from_utf8(buf).expect("csv is utf-8"))
        })
        .collect()
}

/// Render the workbook as typed JSON:
/// `{"sheets": [{"name", "headers", "rows": [[{"text","value","formula"}]]}]}`.
/// Empty cells encode as `{"text": "", "value": null}`; `formula` is omitted
/// when absent.
pub fn excel_to_json(doc: &ExcelDocument) -> serde_json::Value {
    serde_json::json!({
        "sheets": doc.sheets.iter().map(|s| {
            serde_json::json!({
                "name": s.name,
                "headers": s.headers,
                "rows": s.rows.iter().map(|r| {
                    r.iter().map(|c| {
                        let mut obj = serde_json::Map::new();
                        obj.insert("text".into(), serde_json::Value::String(c.text.clone()));
                        obj.insert("value".into(), match &c.value {
                            CellJson::Null => serde_json::Value::Null,
                            CellJson::Bool(b) => serde_json::Value::Bool(*b),
                            CellJson::Integer(i) => serde_json::Value::from(*i),
                            CellJson::Float(f) => serde_json::Number::from_f64(*f)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null),
                            CellJson::Text(t) => serde_json::Value::String(t.clone()),
                        });
                        if let Some(f) = &c.formula {
                            obj.insert("formula".into(), serde_json::Value::String(f.clone()));
                        }
                        serde_json::Value::Object(obj)
                    }).collect::<Vec<_>>()
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(text: &str, value: CellJson) -> CellData {
        CellData { text: text.into(), value, formula: None }
    }

    #[test]
    fn number_json_keeps_integers_exact() {
        assert_eq!(number_json(12.0), CellJson::Integer(12));
        assert_eq!(number_json(-7.0), CellJson::Integer(-7));
        assert_eq!(number_json(1.5), CellJson::Float(1.5));
        assert_eq!(number_json(f64::NAN), CellJson::Null);
        assert_eq!(number_json(f64::INFINITY), CellJson::Null);
        // Beyond i64-exact range stays float.
        assert_eq!(number_json(1e16), CellJson::Float(1e16));
    }

    #[test]
    fn sheet_new_trims_and_rectangularizes() {
        let rows = vec![
            vec![cell("a", CellJson::Text("a".into()))],
            vec![],
            vec![CellData::empty(), CellData::empty()],
        ];
        let s = SheetData::new("S".into(), rows);
        assert_eq!(s.rows.len(), 1);
        assert_eq!(s.headers, vec!["a"]);
    }

    #[test]
    fn markdown_renders_gfm_with_sheet_headers() {
        let doc = ExcelDocument {
            sheets: vec![SheetData::new(
                "Data".into(),
                vec![
                    vec![cell("Item", CellJson::Text("Item".into())), cell("N", CellJson::Text("N".into()))],
                    vec![cell("Apples", CellJson::Text("Apples".into())), cell("12", CellJson::Integer(12))],
                ],
            )],
            markdown: String::new(),
        };
        let md = sheets_to_markdown(&doc.sheets);
        assert!(md.contains("## Data"), "{md}");
        assert!(md.contains("| Item | N |"), "{md}");
        assert!(md.contains("| Apples | 12 |"), "{md}");
    }

    #[test]
    fn csv_quotes_commas_and_skips_empty_sheets() {
        let doc = ExcelDocument {
            sheets: vec![
                SheetData::new("E".into(), vec![]),
                SheetData::new(
                    "D".into(),
                    vec![vec![
                        cell("a,b", CellJson::Text("a,b".into())),
                        cell("q\"q", CellJson::Text("q\"q".into())),
                    ]],
                ),
            ],
            markdown: String::new(),
        };
        let csvs = sheets_to_csv(&doc);
        assert_eq!(csvs.len(), 1);
        assert_eq!(csvs[0].0, "D");
        assert_eq!(csvs[0].1, "\"a,b\",\"q\"\"q\"\n");
    }

    #[test]
    fn json_schema_has_text_value_formula() {
        let mut c = cell("9", CellJson::Integer(9));
        c.formula = Some("=SUM(A1:A2)".into());
        let doc = ExcelDocument {
            sheets: vec![SheetData {
                name: "S".into(),
                headers: vec!["h".into()],
                rows: vec![vec![c, CellData::empty()]],
            }],
            markdown: String::new(),
        };
        let v = excel_to_json(&doc);
        assert_eq!(v["sheets"][0]["name"], "S");
        assert_eq!(v["sheets"][0]["rows"][0][0]["value"], 9);
        assert_eq!(v["sheets"][0]["rows"][0][0]["formula"], "=SUM(A1:A2)");
        assert!(v["sheets"][0]["rows"][0][0].get("formula").is_some());
        assert!(v["sheets"][0]["rows"][0][1].get("formula").is_none());
        assert_eq!(v["sheets"][0]["rows"][0][1]["value"], serde_json::Value::Null);
    }
}
