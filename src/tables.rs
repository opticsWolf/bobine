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
}
