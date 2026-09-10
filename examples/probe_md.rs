//! Probe pdf_oxide's fast-path markdown structuring. Usage: probe_md <pdf> [page]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap();
    let page: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(0);
    let mut pdf = pdf_oxide::api::Pdf::open(&path)?;
    let md = pdf.to_markdown(page)?;
    println!("=== {path} page {} — first 25 lines ===", page + 1);
    for l in md.lines().take(25) {
        println!("{l}");
    }
    Ok(())
}
