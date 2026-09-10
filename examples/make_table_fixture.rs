//! Generate `tests/fixtures/ruled_table.pdf` — a synthetic born-digital page
//! with a fully ruled data grid surrounded by prose, for validating the
//! structured-table cascade (Phase 1 of proposal_media_tables.md).
//!
//! Also prints what pdf_oxide's structured extractor sees under a couple of
//! detector configs, so the fixture doubles as a tuning probe.
//!
//! Usage: cargo run --release --example make_table_fixture

use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::{CellAlign, ColumnWidth, DocumentBuilder, Table, TableCell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = DocumentBuilder::new();
    doc.letter_page()
        .font("Helvetica", 11.0)
        .text("Sample results")
        .paragraph(
            "Table 1 summarizes the measurements collected during the trial. \
             Each row corresponds to one specimen; values are averaged over \
             three independent runs.",
        )
        .table(
            Table::new(vec![
                vec![
                    TableCell::text("Specimen"),
                    TableCell::text("Length (mm)"),
                    TableCell::text("Mass (g)").align(CellAlign::Right),
                    TableCell::text("Yield (%)").align(CellAlign::Right),
                ],
                vec![
                    TableCell::text("A-101"),
                    TableCell::text("12.4"),
                    TableCell::text("3.51").align(CellAlign::Right),
                    TableCell::text("88.2").align(CellAlign::Right),
                ],
                vec![
                    TableCell::text("A-102"),
                    TableCell::text("13.0"),
                    TableCell::text("3.77").align(CellAlign::Right),
                    TableCell::text("91.5").align(CellAlign::Right),
                ],
                vec![
                    TableCell::text("B-201"),
                    TableCell::text("9.8"),
                    TableCell::text("2.44").align(CellAlign::Right),
                    TableCell::text("76.0").align(CellAlign::Right),
                ],
                vec![
                    TableCell::text("B-202"),
                    TableCell::text("10.2"),
                    TableCell::text("2.60").align(CellAlign::Right),
                    TableCell::text("79.8").align(CellAlign::Right),
                ],
            ])
            .with_header_row()
            .with_style({
                let mut st = pdf_oxide::writer::TableStyle::default();
                st.header_background = None; // gray fill rects confuse the grid detector
                st
            })
            .with_column_widths(vec![
                ColumnWidth::Fixed(120.0),
                ColumnWidth::Fixed(110.0),
                ColumnWidth::Fixed(100.0),
                ColumnWidth::Fixed(100.0),
            ]),
        )
        .at(72.0, 660.0)
        .paragraph(
            "The yield percentages increase monotonically with specimen length \
             within each batch. Batch B shows systematically lower yields than \
             batch A across all measured lengths.",
        )
        .done();

    let out = "tests/fixtures/ruled_table.pdf";
    doc.save(out)?;
    println!("wrote {out}");

    let mut pdf = pdf_oxide::api::Pdf::open(out)?;
    let page_rect = Rect::new(0.0, 0.0, 612.0, 792.0);
    for i in 0..pdf.page_count()? {
        for (name, tables) in [
            ("default", pdf.extract_tables_in_rect(i, page_rect)),
            (
                "strict",
                pdf.extract_tables_with_config(
                    i,
                    pdf_oxide::structure::spatial_table_detector::TableDetectionConfig::strict(),
                )
                .map(|ts| {
                    ts.into_iter()
                        .filter(|t| t.bbox.map_or(false, |b| b.intersects(&page_rect)))
                        .collect()
                }),
            ),
        ] {
            match tables {
                Ok(ts) => {
                    println!("page {} [{name}]: {} table(s)", i + 1, ts.len());
                    for t in &ts {
                        println!(
                            "  {} rows x {} cols, header={}",
                            t.rows.len(),
                            t.col_count,
                            t.has_header
                        );
                        if let Some(b) = t.bbox {
                            println!(
                                "  bbox = ({:.0},{:.0} {:.0}x{:.0})",
                                b.x,
                                b.y,
                                b.width,
                                b.height
                            );
                        }
                        for r in &t.rows {
                            let cells: Vec<&str> =
                                r.cells.iter().map(|c| c.text.as_str()).collect();
                            println!("    hdr={} {cells:?}", r.is_header);
                        }
                    }
                }
                Err(e) => println!("page {} [{name}]: ERR {e}", i + 1),
            }
        }
    }
    Ok(())
}
