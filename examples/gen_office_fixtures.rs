//! Generate Office fixtures for `tests/test_office.rs`.
//!
//! Run: `cargo run --example gen_office_fixtures`
//! Writes: `tests/fixtures/office/{basic.docx,types.xlsx,deck.pptx}`
//!
//! Fixtures are built with office_oxide's own `create` API (IR → OOXML),
//! so they exercise exactly what the library can round-trip. Legacy
//! doc/xls/ppt have no creation API — see IMPLEMENTATION_PLAN_office.md.

use office_oxide::DocumentFormat;
use office_oxide::ir::{
    DocumentIR, Element, FootnoteRef, Heading, Image, ImageFormat, InlineContent, List,
    ListItem, Metadata, Note, Paragraph, Section, Table, TableCell, TableRow, TextSpan,
};
use office_oxide::create::create_from_ir;
use office_oxide::xlsx::write::{CellData, CellStyle, NumberFormat, XlsxWriter};

fn span(text: &str) -> InlineContent {
    InlineContent::Text(TextSpan {
        text: text.to_string(),
        ..Default::default()
    })
}

fn para(text: &str) -> Element {
    Element::Paragraph(Paragraph {
        content: vec![span(text)],
        ..Default::default()
    })
}

fn heading(level: u8, text: &str) -> Element {
    Element::Heading(Heading {
        level,
        content: vec![span(text)],
        ..Default::default()
    })
}

/// 16×16 PNG with a red/green checkerboard — distinct pixels so a
/// byte-identity round-trip check is meaningful.
fn fixture_png() -> Vec<u8> {
    let img = image::ImageBuffer::from_fn(16, 16, |x, y| {
        if (x + y) % 2 == 0 {
            image::Rgb([200u8, 30u8, 30u8])
        } else {
            image::Rgb([30u8, 160u8, 30u8])
        }
    });
    let mut buf = Vec::new();
    let dynimg = image::DynamicImage::ImageRgb8(img);
    dynimg
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

fn cell_text(text: &str) -> TableCell {
    TableCell {
        content: vec![para(text)],
        col_span: 1,
        row_span: 1,
        ..Default::default()
    }
}

fn gen_docx(dir: &std::path::Path, png: &[u8]) {
    let styled = Element::Paragraph(Paragraph {
        content: vec![
            span("Plain lead-in with "),
            InlineContent::Text(TextSpan {
                text: "bold".to_string(),
                bold: true,
                ..Default::default()
            }),
            span(" and "),
            InlineContent::Text(TextSpan {
                text: "italic".to_string(),
                italic: true,
                ..Default::default()
            }),
            span(" runs, plus a "),
            InlineContent::Text(TextSpan {
                text: "hyperlink".to_string(),
                hyperlink: Some("https://example.com/bobine".to_string()),
                ..Default::default()
            }),
            span("."),
        ],
        ..Default::default()
    });
    let bullets = Element::List(List {
        ordered: false,
        items: ["alpha item", "beta item", "gamma item"]
            .iter()
            .map(|t| ListItem {
                content: vec![para(t)],
                nested: None,
            })
            .collect(),
        ..Default::default()
    });
    let numbered = Element::List(List {
        ordered: true,
        items: ["first step", "second step"]
            .iter()
            .map(|t| ListItem {
                content: vec![para(t)],
                nested: None,
            })
            .collect(),
        ..Default::default()
    });
    let table = Element::Table(Table {
        rows: vec![
            TableRow {
                cells: vec![cell_text("Name"), cell_text("Qty"), cell_text("Price")],
                is_header: true,
                height_twips: None,
                allow_break: true,
                repeat_as_header: false,
            },
            TableRow {
                cells: vec![cell_text("Apples"), cell_text("12"), cell_text("1.50")],
                is_header: false,
                height_twips: None,
                allow_break: true,
                repeat_as_header: false,
            },
            TableRow {
                cells: vec![
                    TableCell {
                        content: vec![para("Merged across two columns")],
                        col_span: 2,
                        row_span: 1,
                        ..Default::default()
                    },
                    cell_text("9.99"),
                ],
                is_header: false,
                height_twips: None,
                allow_break: true,
                repeat_as_header: false,
            },
        ],
        caption: Some("Fixture inventory".to_string()),
        ..Default::default()
    });
    let footnote_mark = Element::Paragraph(Paragraph {
        content: vec![
            span("A claim needing citation"),
            InlineContent::FootnoteRef(FootnoteRef { note_id: 1, marker: None }),
            span("."),
        ],
        ..Default::default()
    });
    let footnote = Element::Footnote(Note {
        id: 1,
        content: vec![para("The cited source for the claim.")],
        marker: None,
    });
    let image = Element::Image(Image {
        alt_text: Some("red-green checkerboard fixture".to_string()),
        data: Some(png.to_vec()),
        format: Some(ImageFormat::Png),
        ..Default::default()
    });
    let ir = DocumentIR {
        metadata: Metadata {
            format: DocumentFormat::Docx,
            title: Some("Bobine office fixture".to_string()),
            ..Default::default()
        },
        sections: vec![Section {
            title: Some("Main".to_string()),
            elements: vec![
                heading(1, "Bobine Office Fixture"),
                heading(2, "Styled paragraph"),
                styled,
                heading(2, "Lists"),
                bullets,
                numbered,
                heading(2, "Table"),
                table,
                heading(2, "Footnote"),
                footnote_mark,
                footnote,
                heading(2, "Picture"),
                image,
            ],
            ..Default::default()
        }],
    };
    create_from_ir(&ir, DocumentFormat::Docx, dir.join("basic.docx")).unwrap();
}

fn gen_xlsx(dir: &std::path::Path, png: &[u8]) {
    let mut wb = XlsxWriter::new();
    let data = wb.add_sheet_get_index("Data");
    wb.sheet_set_cell(data, 0, 0, CellData::String("Inventory — merged header".to_string()));
    wb.sheet_merge_cells(data, 0, 0, 1, 4);
    for (c, h) in ["Item", "Count", "Price", "In stock"].iter().enumerate() {
        wb.sheet_set_cell(data, 1, c, CellData::String(h.to_string()));
    }
    wb.sheet_set_cell(data, 2, 0, CellData::String("Apples 🍎".to_string()));
    wb.sheet_set_cell(data, 2, 1, CellData::Number(12.0));
    wb.sheet_set_cell(data, 2, 2, CellData::Number(1.5));
    wb.sheet_set_cell(data, 2, 3, CellData::Boolean(true));
    wb.sheet_set_cell(data, 3, 0, CellData::String("Oranges".to_string()));
    wb.sheet_set_cell(data, 3, 1, CellData::Number(7.0));
    wb.sheet_set_cell(data, 3, 2, CellData::Number(2.25));
    wb.sheet_set_cell(data, 3, 3, CellData::Boolean(false));
    // row 4 left fully empty (sparse-row handling)
    wb.sheet_set_cell(
        data,
        5,
        0,
        CellData::String("Total".to_string()),
    );
    wb.sheet_set_cell(data, 5, 1, CellData::Formula("SUM(B3:B4)".to_string()));
    wb.sheet_set_cell_styled(
        data,
        6,
        0,
        CellData::Number(45292.0), // 2024-01-01 as Excel serial
        CellStyle::new().number_format(NumberFormat::Date),
    );
    wb.sheet_set_cell_styled(
        data,
        6,
        1,
        CellData::Number(0.375),
        CellStyle::new().number_format(NumberFormat::Percent),
    );
    // picture anchored on the Data sheet (Phase 3 fixture readiness)
    {
        let mut sheet = wb.add_sheet("Pic");
        sheet.add_image(png.to_vec(), "png", 0, 0, 3_000_000, 3_000_000);
    }
    let summary = wb.add_sheet_get_index("Summary");
    wb.sheet_set_cell(summary, 0, 0, CellData::String("Sheets".to_string()));
    wb.sheet_set_cell(summary, 0, 1, CellData::Number(2.0));
    wb.save(dir.join("types.xlsx")).unwrap();
}

fn gen_pptx(dir: &std::path::Path, png: &[u8]) {
    let slide_image = Element::Image(Image {
        alt_text: Some("deck checkerboard".to_string()),
        data: Some(png.to_vec()),
        format: Some(ImageFormat::Png),
        ..Default::default()
    });
    let ir = DocumentIR {
        metadata: Metadata {
            format: DocumentFormat::Pptx,
            title: Some("Bobine deck fixture".to_string()),
            ..Default::default()
        },
        sections: vec![
            Section {
                title: Some("Title slide".to_string()),
                elements: vec![
                    heading(1, "Bobine Deck Fixture"),
                    para("Generated by gen_office_fixtures."),
                ],
                ..Default::default()
            },
            Section {
                title: Some("Bullets and picture".to_string()),
                elements: vec![
                    heading(2, "Agenda"),
                    Element::List(List {
                        ordered: false,
                        items: ["Convert", "Verify", "Ship"]
                            .iter()
                            .map(|t| ListItem {
                                content: vec![para(t)],
                                nested: None,
                            })
                            .collect(),
                        ..Default::default()
                    }),
                    slide_image,
                ],
                ..Default::default()
            },
            Section {
                title: Some("Numbers".to_string()),
                elements: vec![
                    heading(2, "Quarterly"),
                    Element::Table(Table {
                        rows: vec![
                            TableRow {
                                cells: vec![cell_text("Q"), cell_text("Revenue")],
                                is_header: true,
                                height_twips: None,
                                allow_break: true,
                                repeat_as_header: false,
                            },
                            TableRow {
                                cells: vec![cell_text("Q1"), cell_text("42")],
                                is_header: false,
                                height_twips: None,
                                allow_break: true,
                                repeat_as_header: false,
                            },
                        ],
                        ..Default::default()
                    }),
                ],
                ..Default::default()
            },
        ],
    };
    create_from_ir(&ir, DocumentFormat::Pptx, dir.join("deck.pptx")).unwrap();
}

fn main() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("office");
    std::fs::create_dir_all(&dir).unwrap();
    let png = fixture_png();
    gen_docx(&dir, &png);
    gen_xlsx(&dir, &png);
    gen_pptx(&dir, &png);
    for f in ["basic.docx", "types.xlsx", "deck.pptx"] {
        let meta = std::fs::metadata(dir.join(f)).unwrap();
        println!("wrote {} ({} bytes)", f, meta.len());
    }
}
