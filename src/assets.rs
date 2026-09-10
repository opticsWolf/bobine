//! `okf-asset://` staging convention for extracted images.
//!
//! Rewrites `![alt](local_path)` links to `![alt](okf-asset://<id>)` and
//! copies bytes into `<out_dir>/_assets/<id>.<ext>`. Does no embedding or
//! database work — that is the consumer's job at ingest time.

use std::path::{Path, PathBuf};

use regex::Regex;

use crate::error::{BobineError, Result};

/// Directory name of the staged asset store, relative to the output dir.
pub const ASSET_STORE_DIRNAME: &str = "_assets";

/// Extensions recognized as loose image files (assets, not documents).
pub const IMAGE_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "svg", "avif", "heic", "heif",
];

/// True when `path` has a loose-image extension.
pub fn is_image_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Deterministic, concept-scoped asset id.
///
/// Scoping by the owning document keeps ids unique per concept; hashing the
/// bytes keeps re-runs on unchanged input idempotent.
pub fn asset_id(concept_stem: &str, occurrence: usize, img_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(format!("{concept_stem}|{occurrence}|"));
    h.update(img_bytes);
    let digest = format!("{:x}", h.finalize());
    format!("img_{}", &digest[..16])
}

/// Return just the path/URL part of a markdown image target
/// (strips whitespace and an optional `<>` wrapper).
fn split_src_and_title(raw: &str) -> &str {
    let raw = raw.trim();
    if raw.is_empty() {
        return raw;
    }
    // First token up to whitespace (or a <...> wrapper).
    if let Some(rest) = raw.strip_prefix('<') {
        if let Some(end) = rest.find('>') {
            return &rest[..end];
        }
    }
    raw.split_whitespace().next().unwrap_or(raw)
}

fn skip_link(src_lower: &str) -> bool {
    src_lower.starts_with("http://")
        || src_lower.starts_with("https://")
        || src_lower.starts_with("okf-asset://")
        || src_lower.starts_with("data:")
}

/// Move extracted images into `<out_dir>/_assets/<id>.<ext>` and rewrite
/// their links to `okf-asset://<id>`.
///
/// Resolution order for each link target: basename inside `image_src_dir`
/// (the extractor's temp dir), then doc-relative, then absolute.
///
/// Returns `(rewritten_md, count)`.
pub fn stage_images_as_okf_assets(
    md: &str,
    image_src_dir: &Path,
    source_path: &Path,
    out_dir: &Path,
    concept_stem: &str,
) -> Result<(String, usize)> {
    let re = Regex::new(r"!\[(?P<alt>.*?)\]\((?P<src>.*?)\)").expect("static regex");
    let assets_dir = out_dir.join(ASSET_STORE_DIRNAME);
    let mut occurrence = 0usize;

    let new_md = re.replace_all(md, |caps: &regex::Captures| {
        let alt = caps.name("alt").map(|m| m.as_str()).unwrap_or("").trim();
        let raw_src = caps.name("src").map(|m| m.as_str()).unwrap_or("");
        let src = split_src_and_title(raw_src);
        if src.is_empty()
            || skip_link(&src.to_lowercase())
            || src.starts_with(&format!("{ASSET_STORE_DIRNAME}/"))
        {
            return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }

        // Resolve the physical file.
        let name = Path::new(src)
            .file_name()
            .map(|n| n.to_os_string())
            .unwrap_or_default();
        let mut cand = image_src_dir.join(&name);
        if !cand.is_file() {
            cand = source_path.parent().unwrap_or(Path::new(".")).join(src);
        }
        if !cand.is_file() && Path::new(src).is_absolute() {
            cand = PathBuf::from(src);
        }
        if !cand.is_file() {
            // work-dir-relative links (e.g. `assets/office/img_x.png` staged
            // by the Office converter) resolve against the output dir.
            cand = out_dir.join(src);
        }
        if !cand.is_file() {
            // unresolved — leave the link untouched
            return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }

        let data = match std::fs::read(&cand) {
            Ok(d) => d,
            Err(_) => return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string(),
        };
        occurrence += 1;
        let aid = asset_id(concept_stem, occurrence, &data);
        let ext = cand
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{}", e.to_lowercase()))
            .unwrap_or_else(|| ".bin".to_string());
        if std::fs::create_dir_all(&assets_dir).is_err() {
            return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
        }
        let dest = assets_dir.join(format!("{aid}{ext}"));
        if !dest.exists() {
            if std::fs::copy(&cand, &dest).is_err() {
                return caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string();
            }
        }
        format!("![{alt}](okf-asset://{aid})")
    });

    Ok((new_md.into_owned(), occurrence))
}

/// Collect loose images from `out_dir` and rewrite links to `okf-asset://`.
///
/// Mirrors the classic "converter dropped images in the work dir" flow:
///
/// 1. every loose image file sitting directly in `out_dir` is moved into
///    `out_dir/_assets/`;
/// 2. [`stage_images_as_okf_assets`] then rewrites `![](local)` links and
///    copies bytes into the store (deduped, concept-scoped ids).
///
/// Returns `(rewritten_md, image_count)`.
pub fn stage_images(
    md: &str,
    source_path: &Path,
    out_dir: &Path,
    concept_stem: &str,
) -> Result<(String, usize)> {
    use tracing::warn;

    let img_dir = out_dir.join(ASSET_STORE_DIRNAME);
    std::fs::create_dir_all(&img_dir).map_err(BobineError::Io)?;

    let entries = std::fs::read_dir(out_dir).map_err(BobineError::Io)?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && is_image_ext(&p) {
            let dest = img_dir.join(entry.file_name());
            if std::fs::rename(&p, &dest).is_err() {
                warn!(
                    "could not move loose image {} into asset store",
                    p.display()
                );
            }
        }
    }

    stage_images_as_okf_assets(md, &img_dir, source_path, out_dir, concept_stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_id_deterministic_and_scoped() {
        let a1 = asset_id("doc_a", 1, b"hello");
        let a2 = asset_id("doc_a", 1, b"hello");
        let b = asset_id("doc_b", 1, b"hello");
        let c = asset_id("doc_a", 2, b"hello");
        assert_eq!(a1, a2, "same inputs → same id");
        assert_ne!(a1, b, "different concept → different id");
        assert_ne!(a1, c, "different occurrence → different id");
        assert!(a1.starts_with("img_"));
        assert_eq!(a1.len(), 4 + 16);
    }

    #[test]
    fn test_split_src_and_title() {
        assert_eq!(split_src_and_title("img.png"), "img.png");
        assert_eq!(split_src_and_title("img.png \"title\""), "img.png");
        assert_eq!(split_src_and_title("<imgs/img.png> \"t\""), "imgs/img.png");
        assert_eq!(split_src_and_title(""), "");
    }

    #[test]
    fn test_stage_skips_remote_and_data_links() {
        let tmp = std::env::temp_dir().join(format!("bobine_t_{}", std::process::id()));
        let md = "![](http://x/y.png) ![d](data:image/png;base64,AAA) ![o](https://z/q.jpg)";
        let (out, n) = stage_images_as_okf_assets(md, &tmp, Path::new("a.pdf"), &tmp, "c").unwrap();
        assert_eq!(n, 0);
        assert_eq!(out, md);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_stage_rewrites_local_image() {
        let tmp = std::env::temp_dir().join(format!("bobine_s_{}", std::process::id()));
        let assets = tmp.join("extracted");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("fig.png"), b"\x89PNG fake").unwrap();

        let md = "# T\n\n![a fig](fig.png \"the title\")\n";
        let out_dir = tmp.join("out");
        let (rewritten, n) = stage_images_as_okf_assets(
            md,
            &assets,
            Path::new(tmp.join("src.pdf").as_os_str()),
            &out_dir,
            "src",
        )
        .unwrap();
        assert_eq!(n, 1);
        assert!(rewritten.starts_with("# T\n\n![a fig](okf-asset://img_"));
        assert!(rewritten.ends_with(")\n"));

        // byte landed in the store with the right extension
        let store = out_dir.join(ASSET_STORE_DIRNAME);
        let stored: Vec<_> = std::fs::read_dir(&store).unwrap().collect();
        assert_eq!(stored.len(), 1);
        let name = stored[0].as_ref().unwrap().file_name();
        assert!(name.to_string_lossy().ends_with(".png"));

        // re-running over already-staged markdown changes nothing
        let (again, n2) =
            stage_images_as_okf_assets(&rewritten, &assets, Path::new("src.pdf"), &out_dir, "src")
                .unwrap();
        assert_eq!(n2, 0);
        assert_eq!(again, rewritten);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_unresolved_link_left_alone() {
        let tmp = std::env::temp_dir().join(format!("bobine_u_{}", std::process::id()));
        let md = "![missing](nope.png)";
        let (out, n) = stage_images_as_okf_assets(md, &tmp, Path::new("a.pdf"), &tmp, "c").unwrap();
        assert_eq!(n, 0);
        assert_eq!(out, md);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_is_image_ext() {
        assert!(is_image_ext(Path::new("a.PNG")));
        assert!(is_image_ext(Path::new("b.jpeg")));
        assert!(!is_image_ext(Path::new("c.pdf")));
        assert!(!is_image_ext(Path::new("d")));
    }
}
