//! Probe table-primitive paths on a PDF. Usage: probe_paths <pdf>
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap();
    let mut pdf = pdf_oxide::api::Pdf::open(&path)?;
    let lines = pdf.extract_lines(0)?;
    let rects = pdf.extract_rects(0)?;
    println!("{path}: lines={} rects={}", lines.len(), rects.len());
    let prim_lines: Vec<_> = lines.iter().filter(|p| p.is_table_primitive()).collect();
    let prim_rects: Vec<_> = rects.iter().filter(|p| p.is_table_primitive()).collect();
    println!("table primitives: lines={} rects={}", prim_lines.len(), prim_rects.len());
    for p in prim_lines.iter().take(8) {
        println!("  line bbox {:?}", p.bbox);
    }
    for p in prim_rects.iter().take(8) {
        println!("  rect bbox {:?}", p.bbox);
    }
    Ok(())
}
