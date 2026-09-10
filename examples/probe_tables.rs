//! Probe: run pdf_oxide's structured table extraction over every page.
//!
//! Usage: probe_tables <pdf>

use pdf_oxide::api::Pdf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: probe_tables <pdf>");
    let mut pdf = Pdf::open(&path)?;
    let n = pdf.page_count()?;
    println!("{path}: {n} pages");
    for i in 0..n {
        match pdf.extract_tables(i) {
            Ok(tables) if !tables.is_empty() => {
                for t in &tables {
                    println!(
                        "page {}: {} rows x {} cols (header={})",
                        i + 1,
                        t.rows.len(),
                        t.col_count,
                        t.has_header
                    );
                    if let Some(r) = t.bbox {
                        println!("  bbox = {r:?}");
                    }
                    for r in t.rows.iter().take(3) {
                        let cells: Vec<String> =
                            r.cells.iter().map(|c| c.text.clone()).collect();
                        println!("  row(hdr={}): {cells:?}", r.is_header);
                    }
                }
            }
            Ok(_) => {}
            Err(e) => println!("page {}: ERR {e}", i + 1),
        }
    }
    Ok(())
}
