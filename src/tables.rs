// HTML <table> → GFM pipe-table converter.
//
// Parses simple tables (no rowspan/colspan) into GFM pipe tables.
// Complex tables with merged cells stay as raw HTML.

use regex::Regex;

/// Replace simple HTML tables in a markdown string with GFM pipe tables.
pub fn html_tables_to_gfm(md: &str) -> String {
    let re = Regex::new(r"(?is)<table\b.*?</table>").unwrap();
    re.replace_all(md, |caps: &regex::Captures| {
        let html = &caps[0];
        match one_table_to_gfm(html) {
            Some(gfm) => format!("\n\n{gfm}\n\n"),
            None => html.to_string(),
        }
    })
    .to_string()
}

fn one_table_to_gfm(html: &str) -> Option<String> {
    let rows = parse_simple_table(html)?;
    if rows.is_empty() {
        return None;
    }
    let ncols = rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
    if ncols == 0 {
        return None;
    }

    // First row with any TH cells is the header
    let header_idx = rows.iter().position(|r| r.has_th).unwrap_or(0);
    let header = &rows[header_idx];

    let mut out = String::new();

    // Header row
    let padded: Vec<&str> = header
        .cells
        .iter()
        .map(|c| c.as_str())
        .chain(std::iter::repeat(""))
        .take(ncols)
        .collect();
    out.push_str(&format!("| {} |\n", padded.join(" | ")));

    // Separator
    out.push_str(&format!("| {} |\n", vec!["---"; ncols].join(" | ")));

    // Body rows (all except header)
    for (i, row) in rows.iter().enumerate() {
        if i == header_idx {
            continue;
        }
        let padded: Vec<&str> = row
            .cells
            .iter()
            .map(|c| c.as_str())
            .chain(std::iter::repeat(""))
            .take(ncols)
            .collect();
        out.push_str(&format!("| {} |\n", padded.join(" | ")));
    }

    Some(out)
}

struct TableRow {
    cells: Vec<String>,
    has_th: bool,
}

/// Parse a single <table> into rows. Returns None if rowspan/colspan found.
fn parse_simple_table(html: &str) -> Option<Vec<TableRow>> {
    let mut rows: Vec<TableRow> = Vec::new();
    let mut in_row = false;
    let mut in_cell = false;
    let mut row_has_th = false;
    let mut current_cells: Vec<String> = Vec::new();
    let mut current_text = String::new();

    let mut chars = html.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '<' {
            // Read tag name
            let mut tag_name = String::new();
            let mut in_tag = true;
            let mut attrs = String::new();
            while in_tag {
                match chars.next() {
                    None => break,
                    Some('/') => {
                        if tag_name.is_empty() {
                            // closing tag
                            tag_name.push('/');
                        } else {
                            attrs.push('/');
                        }
                    }
                    Some('>') => in_tag = false,
                    Some(c) if c.is_ascii_whitespace() => {
                        // attributes follow
                        attrs.push(c);
                        while let Some(ac) = chars.next() {
                            if ac == '>' {
                                in_tag = false;
                                break;
                            }
                            attrs.push(ac);
                        }
                    }
                    Some(c) => tag_name.push(c),
                }
            }

            let tag_lower = tag_name.to_lowercase();

            match tag_lower.as_str() {
                "tr" => {
                    if in_row {
                        // Malformed: close previous
                        rows.push(TableRow {
                            cells: current_cells.clone(),
                            has_th: row_has_th,
                        });
                    }
                    in_row = true;
                    row_has_th = false;
                    current_cells = Vec::new();
                }
                "/tr" => {
                    if in_row {
                        rows.push(TableRow {
                            cells: current_cells.clone(),
                            has_th: row_has_th,
                        });
                        in_row = false;
                        current_cells = Vec::new();
                    }
                }
                "td" | "/td" => {
                    // Check for colspan/rowspan
                    if tag_lower == "td" {
                        let al = attrs.to_lowercase();
                        if al.contains("colspan") || al.contains("rowspan") {
                            let has_multi = al.contains("colspan=\"")
                                && !al.contains("colspan=\"1\"")
                                || al.contains("rowspan=\"") && !al.contains("rowspan=\"1\"");
                            if has_multi {
                                return None; // Complex table
                            }
                        }
                    }
                    if tag_lower.starts_with('/') {
                        if in_cell {
                            let text = current_text
                                .split_whitespace()
                                .collect::<Vec<_>>()
                                .join(" ")
                                .replace('|', r"\|");
                            current_cells.push(text);
                            current_text = String::new();
                            in_cell = false;
                        }
                    } else {
                        in_cell = true;
                        current_text = String::new();
                    }
                }
                "th" | "/th" => {
                    if tag_lower == "th" {
                        row_has_th = true;
                        in_cell = true;
                        current_text = String::new();
                    } else {
                        if in_cell {
                            let text = current_text
                                .split_whitespace()
                                .collect::<Vec<_>>()
                                .join(" ")
                                .replace('|', r"\|");
                            current_cells.push(text);
                            current_text = String::new();
                            in_cell = false;
                        }
                    }
                }
                "br" | "br/" => {
                    if in_cell {
                        current_text.push(' ');
                    }
                }
                _ => {}
            }
        } else if in_cell {
            current_text.push(ch);
        }
    }

    // Close any unclosed row
    if in_row && !current_cells.is_empty() {
        rows.push(TableRow {
            cells: current_cells,
            has_th: row_has_th,
        });
    }

    if rows.is_empty() { None } else { Some(rows) }
}

// =====================================================================
// Structured tables (pdf_oxide extract_tables_in_rect) → markdown
// =====================================================================

use crate::pdf_source::{SourceTable, SourceTableRow};

/// Heuristic acceptance gate for text-layer grid extractions.
///
/// The spatial detector currently degrades on real-world tables — booktabs
///-style rules yield fragmented grids with missing header rows and
/// half-empty rows (measured on arXiv 1706.03762), and synthetic grids can
/// collapse body rows into one concatenated cell. Both symptoms show up as
/// sparsely-filled rows, so we require every row to be at least 60% filled.
/// A rejected table lets the caller fall through to the plain text dump,
/// which is never worse than a garbled "structured" table.
pub fn is_plausible_table(table: &SourceTable) -> bool {
    if table.rows.len() < 2 || table.col_count < 2 {
        return false;
    }
    const MIN_FILL_NUM: usize = 3; // 3/5 = 60%
    const MIN_FILL_DEN: usize = 5;
    table.rows.iter().all(|r| {
        let filled = r.cells.iter().filter(|c| !c.text.trim().is_empty()).count();
        filled * MIN_FILL_DEN >= r.cells.len() * MIN_FILL_NUM
    })
}

/// Render a structured table (from the text-layer grid detector / structure
/// tree) as markdown.
///
/// Plain grids become GFM pipe tables. Tables containing merged cells
/// (colspan/rowspan > 1) have no lossless pipe representation and are
/// rendered back as minimal HTML instead — the same pass-through policy
/// [`html_tables_to_gfm`] applies to complex HTML input.
///
/// Returns `None` for tables with no non-empty cell content, letting the
/// caller fall through to the next extraction stage.
pub fn source_table_markdown(table: &SourceTable) -> Option<String> {
    let has_content = table
        .rows
        .iter()
        .any(|r| r.cells.iter().any(|c| !c.text.trim().is_empty()));
    if !has_content {
        return None;
    }
    let has_spans = table
        .rows
        .iter()
        .any(|r| r.cells.iter().any(|c| c.colspan > 1 || c.rowspan > 1));
    if has_spans {
        Some(source_table_to_html(table))
    } else {
        Some(source_table_to_gfm(table))
    }
}

fn gfm_escape(text: &str) -> String {
    text.replace('|', "\\|")
}

fn source_table_to_gfm(table: &SourceTable) -> String {
    let ncols = table
        .rows
        .iter()
        .map(|r| r.cells.len())
        .max()
        .unwrap_or(0)
        .max(table.col_count);
    let cell = |c: Option<&crate::pdf_source::SourceTableCell>| {
        c.map(|c| gfm_escape(c.text.trim()))
            .unwrap_or_default()
    };

    // Header row: first explicitly-flagged header row, else the first row.
    let header_idx = table.rows.iter().position(|r| r.is_header).unwrap_or(0);
    let mut out = String::new();

    let header: Vec<String> = (0..ncols)
        .map(|i| cell(table.rows[header_idx].cells.get(i)))
        .collect();
    out.push_str(&format!("| {} |\n", header.join(" | ")));
    out.push_str(&format!("| {} |\n", vec!["---"; ncols.max(1)].join(" | ")));

    for (ri, row) in table.rows.iter().enumerate() {
        if ri == header_idx {
            continue;
        }
        let body: Vec<String> = (0..ncols).map(|i| cell(row.cells.get(i))).collect();
        out.push_str(&format!("| {} |\n", body.join(" | ")));
    }
    out
}

fn source_table_to_html(table: &SourceTable) -> String {
    let mut out = String::from("<table>\n");
    let header_idx = table.rows.iter().position(|r| r.is_header);
    if table.has_header || header_idx.is_some() {
        let hi = header_idx.unwrap_or(0);
        out.push_str("<thead>\n<tr>\n");
        for c in &table.rows[hi].cells {
            push_html_cell(&mut out, c, "th");
        }
        out.push_str("</tr>\n</thead>\n");
    }
    out.push_str("<tbody>\n");
    for (ri, row) in table.rows.iter().enumerate() {
        if Some(ri) == header_idx && (table.has_header || header_idx.is_some()) {
            continue;
        }
        out.push_str("<tr>\n");
        for c in &row.cells {
            push_html_cell(&mut out, c, "td");
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>");
    out
}

fn push_html_cell(out: &mut String, c: &crate::pdf_source::SourceTableCell, tag: &str) {
    let esc = c
        .text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let colspan = if c.colspan > 1 {
        format!(" colspan=\"{}\"", c.colspan)
    } else {
        String::new()
    };
    let rowspan = if c.rowspan > 1 {
        format!(" rowspan=\"{}\"", c.rowspan)
    } else {
        String::new()
    };
    out.push_str(&format!(
        "<{tag}{colspan}{rowspan}>{esc}</{tag}>\n",
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_table_to_gfm() {
        let html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>";
        let md = html_tables_to_gfm(html);
        assert!(md.contains("| A | B |"));
        assert!(md.contains("---"));
        assert!(md.contains("| 1 | 2 |"));
    }

    #[test]
    fn colspan_table_stays_html() {
        let html = "<table><tr><td colspan=\"2\">merged</td></tr></table>";
        let md = html_tables_to_gfm(html);
        assert!(md.contains("<table"));
    }

    #[test]
    fn pipe_escaped_in_cell() {
        let html = "<table><tr><td>x|y</td></tr></table>";
        let md = html_tables_to_gfm(html);
        assert!(md.contains(r"x\|y"));
    }

    #[test]
    fn parse_basic_table() {
        let rows = parse_simple_table("<table><tr><td>a</td><td>b</td></tr></table>").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].cells, vec!["a", "b"]);
    }

    #[test]
    fn parse_with_th() {
        let rows = parse_simple_table(
            "<table><tr><th>H1</th><th>H2</th></tr><tr><td>d1</td><td>d2</td></tr></table>",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].has_th);
        assert!(!rows[1].has_th);
    }

    #[test]
    fn parse_rejects_colspan() {
        assert!(
            parse_simple_table("<table><tr><td colspan=\"3\">wide</td></tr></table>").is_none()
        );
    }

    #[test]
    fn parse_allows_colspan_one() {
        assert!(parse_simple_table("<table><tr><td colspan=\"1\">ok</td></tr></table>").is_some());
    }

    #[test]
    fn parse_whitespace_normalized() {
        let rows = parse_simple_table("<table><tr><td>hello   world</td></tr></table>").unwrap();
        assert_eq!(rows[0].cells[0], "hello world");
    }

    #[test]
    fn parse_br_adds_space() {
        let rows = parse_simple_table("<table><tr><td>line1<br>line2</td></tr></table>").unwrap();
        assert_eq!(rows[0].cells[0], "line1 line2");
    }

    #[test]
    fn no_table_unchanged() {
        assert_eq!(html_tables_to_gfm("plain text"), "plain text");
    }

    // ==================================================================
    // source_table_markdown (structured tables from the text layer)
    // ==================================================================

    use crate::pdf_source::{SourceTableCell, SourceTable};

    fn cell(text: &str) -> SourceTableCell {
        SourceTableCell {
            text: text.into(),
            colspan: 1,
            rowspan: 1,
        }
    }

    fn simple_table() -> SourceTable {
        SourceTable {
            rows: vec![
                crate::pdf_source::SourceTableRow {
                    cells: vec![cell("A"), cell("B")],
                    is_header: true,
                },
                crate::pdf_source::SourceTableRow {
                    cells: vec![cell("1"), cell("2")],
                    is_header: false,
                },
            ],
            has_header: true,
            col_count: 2,
            bbox: None,
        }
    }

    #[test]
    fn structured_table_to_gfm() {
        let md = source_table_markdown(&simple_table()).unwrap();
        assert!(md.contains("| A | B |"));
        assert!(md.contains("| --- | --- |"));
        assert!(md.contains("| 1 | 2 |"));
    }

    #[test]
    fn structured_table_no_flagged_header_uses_first_row() {
        let mut t = simple_table();
        t.rows[0].is_header = false;
        t.has_header = false;
        let md = source_table_markdown(&t).unwrap();
        assert!(md.starts_with("| A | B |\n| --- | --- |\n| 1 | 2 |\n"));
    }

    #[test]
    fn structured_table_pads_ragged_rows() {
        let mut t = simple_table();
        t.rows[1].cells = vec![cell("only")];
        let md = source_table_markdown(&t).unwrap();
        assert!(md.contains("| only |  |"));
    }

    #[test]
    fn structured_table_escapes_pipes() {
        let mut t = simple_table();
        t.rows[0].cells[0] = cell("x|y");
        let md = source_table_markdown(&t).unwrap();
        assert!(md.contains(r"x\|y"));
    }

    #[test]
    fn structured_table_with_spans_stays_html() {
        let mut t = simple_table();
        t.rows[0].cells[0].colspan = 2;
        let md = source_table_markdown(&t).unwrap();
        assert!(md.contains("<table>"));
        assert!(md.contains("colspan=\"2\""));
        assert!(!md.contains("| --- |"));
    }

    #[test]
    fn structured_table_rowspan_stays_html() {
        let mut t = simple_table();
        t.rows[0].cells[0].rowspan = 3;
        let md = source_table_markdown(&t).unwrap();
        assert!(md.contains("rowspan=\"3\""));
    }

    #[test]
    fn structured_empty_table_is_none() {
        let mut t = simple_table();
        for r in &mut t.rows {
            for c in &mut r.cells {
                c.text = "  ".into();
            }
        }
        assert!(source_table_markdown(&t).is_none());
    }

    #[test]
    fn html_cells_escape_markup() {
        let mut t = simple_table();
        t.rows[0].cells[0].colspan = 2;
        t.rows[0].cells[0].text = "a<b>&c".into();
        let md = source_table_markdown(&t).unwrap();
        assert!(md.contains("a&lt;b&gt;&amp;c"));
    }

    // ==================================================================
    // is_plausible_table (acceptance gate for detector output)
    // ==================================================================

    #[test]
    fn plausible_dense_grid_accepted() {
        assert!(is_plausible_table(&simple_table()));
    }

    #[test]
    fn sparse_row_rejected() {
        // Symptom: body rows collapsed into one concatenated cell.
        let mut t = simple_table();
        t.rows.push(crate::pdf_source::SourceTableRow {
            cells: vec![
                cell("A-10112.43.5188.2"),
                cell(""),
                cell(""),
                cell(""),
            ],
            is_header: false,
        });
        assert!(!is_plausible_table(&t));
    }

    #[test]
    fn fragment_rows_rejected() {
        // Symptom seen on arXiv 1706.03762 p.8: fragmented booktabs grid.
        // A 2-of-4 filled row (50%) must not pass the 60% bar.
        let mut t = simple_table();
        t.rows.push(crate::pdf_source::SourceTableRow {
            cells: vec![cell("ByteNet"), cell("[18]"), cell(""), cell("")],
            is_header: false,
        });
        assert!(!is_plausible_table(&t));
    }

    #[test]
    fn single_row_rejected() {
        let mut t = simple_table();
        t.rows.truncate(1);
        assert!(!is_plausible_table(&t));
    }

    #[test]
    fn single_column_rejected() {
        let mut t = simple_table();
        for r in &mut t.rows {
            r.cells.truncate(1);
        }
        t.col_count = 1;
        assert!(!is_plausible_table(&t));
    }
}
