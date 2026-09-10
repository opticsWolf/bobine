//! Office picture extraction.
//!
//! office_oxide renders embedded pictures as `![alt](rIdN)` links (or drops
//! them entirely for pptx/xlsx drawings) — relationship ids that resolve to
//! nothing outside the package. This module walks the [`DocumentIR`] image
//! elements in document order, stages their bytes into the work dir, and
//! splices the markdown links to the staged files. Pictures the markdown
//! never references are appended as a gallery, mirroring the PDF
//! unreferenced-figure flow (`maybe_append_gallery`).
//!
//! Staged layout mirrors the PDF one: `<image_output_dir>/office/img{k}.{ext}`
//! with content-hash file stems (`asset_id`), so re-runs are idempotent.
//! The ingest pipeline's `stage_images` pass later promotes these into
//! `okf-asset://` links — no pipeline changes needed.

use office_oxide::ir::{DocumentIR, Element, ImageFormat};

use crate::assets::asset_id;
use crate::error::Result;

/// One stageable picture: display name, file extension, raw bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficeImage {
    /// Alt text (may be empty).
    pub alt: String,
    /// Lowercase file extension without dot (`png`, `jpg`, …).
    pub ext: String,
    /// Raw image bytes.
    pub data: Vec<u8>,
}

fn format_ext(f: Option<&ImageFormat>) -> String {
    match f {
        Some(ImageFormat::Png) => "png",
        Some(ImageFormat::Jpeg) => "jpg",
        Some(ImageFormat::Gif) => "gif",
        Some(ImageFormat::Tiff) => "tif",
        Some(ImageFormat::Bmp) => "bmp",
        Some(ImageFormat::Emf) => "emf",
        Some(ImageFormat::Wmf) => "wmf",
        None => "bin",
    }
    .to_string()
}

/// Collect stageable pictures from IR in document order.
///
/// Skips decorative images (the Office analogue of the PDF
/// `min_figure_area_pts` decoration rule) and linked-but-not-embedded
/// pictures (`data: None` — nothing to stage).
pub fn collect_office_images(ir: &DocumentIR) -> Vec<OfficeImage> {
    let mut out = Vec::new();
    for section in &ir.sections {
        collect_from_elements(&section.elements, &mut out);
    }
    out
}

fn collect_from_elements(elements: &[Element], out: &mut Vec<OfficeImage>) {
    for el in elements {
        match el {
            Element::Image(img) => {
                if img.decorative {
                    continue;
                }
                if let Some(data) = &img.data {
                    out.push(OfficeImage {
                        alt: img.alt_text.clone().unwrap_or_default(),
                        ext: format_ext(img.format.as_ref()),
                        data: data.clone(),
                    });
                }
            }
            // Pictures can hide inside table cells, text boxes, and notes.
            Element::Table(t) => {
                for row in &t.rows {
                    for cell in &row.cells {
                        collect_from_elements(&cell.content, out);
                    }
                }
            }
            Element::TextBox(tb) => collect_from_elements(&tb.content, out),
            Element::Footnote(n) | Element::Endnote(n) => {
                collect_from_elements(&n.content, out);
            }
            _ => {}
        }
    }
}

/// Package-level media fallback for formats whose IR drops pictures.
///
/// `to_ir()` carries image bytes for docx but yields zero `Image` elements
/// for pptx/xlsx drawings (verified on fixtures). OOXML packages are zips:
/// embedded pictures live under `*/media/*`, so enumerate those parts and
/// read their bytes directly. Alt text is unrecoverable at this level — the
/// filename stem stands in. Returns empty (not an error) for legacy CFB
/// formats and anything unreadable.
pub fn collect_package_images(path: &std::path::Path) -> Vec<OfficeImage> {
    const IMG_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "emf", "wmf"];
    let mut opc = match office_oxide::core::opc::OpcReader::open(path) {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    let mut names: Vec<String> = opc
        .part_names()
        .iter()
        .map(|p| p.as_str().to_string())
        .filter(|n| {
            let lower = n.to_lowercase();
            lower.contains("/media/")
                && lower.rsplit('.').next().is_some_and(|e| IMG_EXTS.contains(&e))
        })
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let part = match office_oxide::core::opc::PartName::new(&name) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let data = match opc.read_part(&part) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let lower = name.to_lowercase();
        let raw_ext = lower.rsplit('.').next().unwrap_or("bin");
        let ext = if raw_ext == "jpeg" { "jpg".to_string() } else { raw_ext.to_string() };
        let stem = name
            .rsplit('/')
            .next()
            .unwrap_or("image")
            .rsplit_once('.')
            .map(|(s, _)| s.to_string())
            .unwrap_or_else(|| "image".to_string());
        out.push(OfficeImage { alt: stem, ext, data });
    }
    out
}

/// Stage `images` into `dest_dir` and rewrite the markdown's local image
/// links to the staged files.
///
/// Pairing is positional: the nth non-URL `![](...)` link takes the nth
/// staged picture (both in document order). Links that look like URLs,
/// `data:`, `okf-asset://`, or already-staged `_assets/` targets are left
/// untouched, as are surplus links when the counts mismatch. Surplus staged
/// pictures (e.g. pptx/xlsx drawings the markdown drops) are appended as a
/// gallery. Returns the rewritten markdown.
pub fn splice_office_images(
    md: &str,
    images: &[OfficeImage],
    dest_dir: &std::path::Path,
    stem: &str,
) -> Result<String> {
    if images.is_empty() {
        return Ok(md.to_string());
    }
    std::fs::create_dir_all(dest_dir).map_err(crate::error::BobineError::Io)?;

    // Pre-stage every picture so link rewriting is a pure substitution.
    // Relative links mirror the PDF scope layout: `<parent>/<dir>/<file>`
    // (e.g. `assets/office/img_<hash>.png`, relative to the work dir).
    let scope = dest_dir
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let dirname = dest_dir.file_name().and_then(|n| n.to_str()).unwrap_or("assets");
    let mut staged: Vec<(String, String)> = Vec::new(); // (alt, rel_path)
    for (k, img) in images.iter().enumerate() {
        let name = format!("{}.{}", asset_id(stem, k + 1, &img.data), img.ext);
        std::fs::write(dest_dir.join(&name), &img.data).map_err(crate::error::BobineError::Io)?;
        let rel = if scope.is_empty() {
            format!("{dirname}/{name}")
        } else {
            format!("{scope}/{dirname}/{name}")
        };
        staged.push((img.alt.clone(), rel));
    }

    let re = regex::Regex::new(r"!\[(?P<alt>.*?)\]\((?P<src>.*?)\)").expect("static regex");
    let mut cursor = 0usize;
    let new_md = re
        .replace_all(md, |caps: &regex::Captures| {
            let alt = caps.name("alt").map(|m| m.as_str()).unwrap_or("");
            let src = caps.name("src").map(|m| m.as_str()).unwrap_or("");
            let target = src.split_whitespace().next().unwrap_or("");
            let lower = target.to_lowercase();
            let external = target.is_empty()
                || lower.starts_with("http://")
                || lower.starts_with("https://")
                || lower.starts_with("data:")
                || lower.starts_with("okf-asset://")
                || lower.starts_with("_assets/");
            if external || cursor >= staged.len() {
                return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
            let (_, rel) = &staged[cursor];
            cursor += 1;
            // Keep the author's alt text; fall back to the IR one.
            let keep_alt = if alt.is_empty() {
                staged[cursor - 1].0.clone()
            } else {
                alt.to_string()
            };
            format!("![{keep_alt}]({rel})")
        })
        .into_owned();

    // Gallery for staged pictures the markdown never referenced.
    // `cursor` advanced once per rewritten link, so the tail is unwedded.
    let mut out = new_md;
    let mut gallery = String::new();
    for (alt, rel) in staged.iter().skip(cursor) {
        if out.contains(&format!("({rel})")) {
            continue;
        }
        gallery.push_str(&format!("![{alt}]({rel})\n"));
    }
    if !gallery.is_empty() {
        out.push_str(&format!("\n\n{gallery}"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use office_oxide::ir::{Image, Note};

    fn img(alt: &str) -> Element {
        Element::Image(Image {
            alt_text: Some(alt.to_string()),
            data: Some(vec![1, 2, 3]),
            format: Some(ImageFormat::Png),
            ..Default::default()
        })
    }

    #[test]
    fn collect_skips_decorative_and_link_only() {
        let deco = Image {
            alt_text: None,
            data: Some(vec![9]),
            format: Some(ImageFormat::Jpeg),
            decorative: true,
            ..Default::default()
        };
        let link_only = Image {
            alt_text: Some("remote".to_string()),
            data: None,
            format: None,
            ..Default::default()
        };
        let ir = DocumentIR {
            sections: vec![office_oxide::ir::Section {
                elements: vec![
                    img("a"),
                    Element::Image(deco),
                    Element::Image(link_only),
                    // Nested inside a table cell still found.
                    Element::Table(office_oxide::ir::Table {
                        rows: vec![office_oxide::ir::TableRow {
                            cells: vec![office_oxide::ir::TableCell {
                                content: vec![img("nested")],
                                col_span: 1,
                                row_span: 1,
                                ..Default::default()
                            }],
                            is_header: false,
                            height_twips: None,
                            allow_break: true,
                            repeat_as_header: false,
                        }],
                        ..Default::default()
                    }),
                    Element::Footnote(Note {
                        id: 1,
                        content: vec![img("note")],
                        marker: None,
                    }),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let got = collect_office_images(&ir);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].alt, "a");
        assert_eq!(got[0].ext, "png");
        assert_eq!(got[1].alt, "nested");
        assert_eq!(got[2].alt, "note");
        assert_eq!(format_ext(None), "bin");
        assert_eq!(format_ext(Some(&ImageFormat::Wmf)), "wmf");
    }

    #[test]
    fn splice_rewrites_positionally_and_galleries_the_rest() {
        let dir = std::env::temp_dir().join("bobine_test").join("office_splice");
        let _ = std::fs::remove_dir_all(&dir);
        // Nest one level so the scope prefix logic is exercised:
        // links must read `scope/office_splice/img_*`.
        let dir = dir.join("scope").join("office_splice");
        let images = vec![
            OfficeImage { alt: "first".into(), ext: "png".into(), data: vec![1] },
            OfficeImage { alt: "second".into(), ext: "jpg".into(), data: vec![2] },
        ];
        let md = "See ![first](rId2) and ![kept](https://x/y.png) end.";
        let out = splice_office_images(md, &images, &dir, "stem").unwrap();
        assert!(out.contains("![first](scope/office_splice/img_"), "{out}");
        assert!(out.contains("![kept](https://x/y.png)"), "{out}");
        // Second picture unreferenced → gallery.
        assert!(out.contains("![second](scope/office_splice/img_"), "{out}");
        // Files staged.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    }
}
