//! Render one PDF page to PNG. Usage: render_page <pdf> <page> <out.png> [dpi]
use pdf_oxide::api::Pdf;
use bobine::pdf_source::PdfSource;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdf_path = std::env::args().nth(1).expect("pdf");
    let page: usize = std::env::args().nth(2).expect("page").parse()?;
    let out = std::env::args().nth(3).expect("out");
    let dpi: u32 = std::env::args().nth(4).map(|s| s.parse().unwrap()).unwrap_or(150);
    let mut pdf = Pdf::open(&pdf_path)?;
    std::fs::write(&out, pdf.render_png(page, dpi)?)?;
    println!("wrote {out}");
    Ok(())
}
