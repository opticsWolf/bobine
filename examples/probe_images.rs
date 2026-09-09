//! Probe embedded-image placement vs text extents. Usage: probe_images <pdf> [page0]
use pdf_oxide::api::Pdf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap();
    let page: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(0);
    let mut pdf = Pdf::open(&path)?;
    let media = pdf.page_media_box(page)?;
    println!("media box: {media:?}");
    let imgs = pdf.extract_images(page)?;
    for (n, im) in imgs.iter().enumerate() {
        println!("img{n}: {}x{} bbox={:?} rot={}", im.width(), im.height(), im.bbox(), im.rotation_degrees());
    }
    let chars = pdf.extract_chars(page)?;
    let miny = chars.iter().map(|c| c.bbox.y).fold(f32::INFINITY, f32::min);
    let maxy = chars.iter().map(|c| c.bbox.y + c.bbox.height).fold(f32::NEG_INFINITY, f32::max);
    println!("text y range: {miny:.0}..{maxy:.0} (chars={})", chars.len());
    Ok(())
}
