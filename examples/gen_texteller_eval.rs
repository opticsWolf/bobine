//! Build a labeled TexTeller eval set from the ikat corpus ground truth.
//!
//! Usage: gen_texteller_eval [--out DIR] [--limit N] [--dpi D]
//!
//! Every `$$…$$` display span in `ikat-corpus/ground-truth/*.md` is a
//! hand-checked LaTeX formula. This tool cleans those spans, renders each
//! one with `pdflatex` (standalone class, display style) into a PNG crop,
//! and writes `manifest.json` with the GT LaTeX next to the image. The
//! benchmark (`bench_texteller --eval <DIR>`) then scores predictions
//! against that GT.
//!
//! Why synthetic renders: the corpus PDFs are PLOS CID-font papers whose
//! rasterization is corrupted upstream (docs/cid_render_blocker.md), so
//! page crops cannot be used as a formula eval set. Rendering the GT
//! LaTeX with Computer Modern gives clean, correctly typeset formulas and
//! keeps the GT exact. This measures model parity/accuracy, not the
//! renderer bug.
//!
//! Requires `pdflatex` on PATH (TinyTeX works). Output defaults to
//! `target/texteller_eval` (git-ignored). Selection is deterministic:
//! formulas are hashed and taken round-robin across papers, so reruns
//! produce the same set.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use bobine::pdf_source::PdfSource;
use pdf_oxide::api::Pdf;
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_TEX_LEN: usize = 300;

#[derive(Serialize)]
struct Manifest {
    dpi: u32,
    items: Vec<Item>,
}

#[derive(Serialize, Clone)]
struct Item {
    id: String,
    paper: String,
    tex: String,
    /// Path relative to the manifest directory.
    png: String,
}

/// Extract `$$…$$` display spans (multi-line fences joined), matching
/// the extraction in tests/test_bench_ocr.rs.
fn extract_displays(md: &str) -> Vec<String> {
    let mut logical: Vec<String> = Vec::new();
    let mut open = false;
    let mut buf = String::new();
    for line in md.lines() {
        let t = line.trim().trim_end_matches('\r').trim();
        if !open {
            if t.starts_with("$$") {
                if t.len() > 2 && t.ends_with("$$") {
                    logical.push(t.to_string());
                } else {
                    open = true;
                    buf = t.to_string();
                }
            }
        } else {
            buf.push(' ');
            buf.push_str(t);
            if t.ends_with("$$") {
                logical.push(std::mem::take(&mut buf));
                open = false;
            }
        }
    }
    if open && buf.starts_with("$$") {
        logical.push(buf);
    }
    let mut out = Vec::new();
    for t in &logical {
        let no_prefix = t.strip_prefix("$$").unwrap_or(t).trim();
        let inner = no_prefix.strip_suffix("$$").unwrap_or(no_prefix).trim();
        if !inner.is_empty() {
            out.push(inner.to_string());
        }
    }
    out
}

/// Strip `\tag{…}` (display numbering is not part of the formula) and
/// squeeze whitespace. Returns `None` for spans we refuse to score:
/// corpus `?` glyph corruption (lost minus/operators), alignment tabs
/// (illegal outside an alignment environment), or oversized formulas.
fn clean(tex: &str) -> Option<String> {
    let mut s = String::with_capacity(tex.len());
    let chars: Vec<char> = tex.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let is_tag = chars[i] == '\\'
            && chars.get(i + 1) == Some(&'t')
            && chars.get(i + 2) == Some(&'a')
            && chars.get(i + 3) == Some(&'g');
        if is_tag {
            // Skip \tag or \tag* plus its brace group.
            let mut j = i + 4;
            if chars.get(j) == Some(&'*') {
                j += 1;
            }
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if chars.get(j) == Some(&'{') {
                let mut depth = 0usize;
                while j < chars.len() {
                    match chars[j] {
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        s.push(chars[i]);
        i += 1;
    }
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.is_empty()
        || s.contains('?')
        || s.contains('&')
        || s.contains("\\tag")
        || s.chars().count() > MAX_TEX_LEN
    {
        return None;
    }
    Some(s)
}

fn hash_key(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    format!("{:x}", h.finalize())
}

fn latex_doc(tex: &str) -> String {
    format!(
        "\\documentclass[preview,border=2pt]{{standalone}}\n\
         \\usepackage{{amsmath,amssymb}}\n\
         \\begin{{document}}\n\
         $\\displaystyle {tex}$\n\
         \\end{{document}}\n"
    )
}

fn render_one(out: &Path, item: &Item, dpi: u32) -> Result<(), String> {
    let tex_dir = out.join("tex");
    let pdf_dir = out.join("pdf");
    let png_dir = out.join("png");
    let tex_path = tex_dir.join(format!("{}.tex", item.id));
    std::fs::write(&tex_path, latex_doc(&item.tex)).map_err(|e| format!("write tex: {e}"))?;

    let output = Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg("-halt-on-error")
        .arg("-output-directory")
        .arg(&pdf_dir)
        .arg("-jobname")
        .arg(&item.id)
        .arg(&tex_path)
        .current_dir(&tex_dir)
        .output()
        .map_err(|e| format!("run pdflatex: {e}"))?;
    if !output.status.success() {
        let log = String::from_utf8_lossy(&output.stdout);
        let tail = log.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(format!("pdflatex failed: {tail}"));
    }

    let pdf_path = pdf_dir.join(format!("{}.pdf", item.id));
    let mut pdf = Pdf::open(&pdf_path).map_err(|e| format!("open pdf: {e}"))?;
    let png = pdf
        .render_png(0, dpi)
        .map_err(|e| format!("render png: {e}"))?;
    std::fs::write(png_dir.join(format!("{}.png", item.id)), png)
        .map_err(|e| format!("write png: {e}"))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = PathBuf::from("target/texteller_eval");
    let mut limit = 30usize;
    let mut dpi = 300u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out needs a path")?),
            "--limit" => limit = args.next().ok_or("--limit needs a value")?.parse()?,
            "--dpi" => dpi = args.next().ok_or("--dpi needs a value")?.parse()?,
            other => return Err(format!("unknown arg {other}").into()),
        }
    }

    let gt_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ikat-corpus")
        .join("ground-truth");
    if !gt_dir.is_dir() {
        return Err(format!("missing {}", gt_dir.display()).into());
    }

    // paper -> cleaned, deduped formulas
    let mut by_paper: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for entry in std::fs::read_dir(&gt_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let paper = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        let md = std::fs::read_to_string(&path)?;
        let mut cleaned = Vec::new();
        for raw in extract_displays(&md) {
            if let Some(tex) = clean(&raw) {
                if seen.insert(tex.clone()) {
                    cleaned.push(tex);
                }
            }
        }
        if !cleaned.is_empty() {
            by_paper.insert(paper, cleaned);
        }
    }

    // Deterministic round-robin across papers, hash-ordered inside each.
    for v in by_paper.values_mut() {
        v.sort_by_key(|s| hash_key(s));
    }
    let mut selected: Vec<(String, String)> = Vec::new();
    let mut round = 0usize;
    while selected.len() < limit {
        let mut took = false;
        for (paper, formulas) in &by_paper {
            if let Some(tex) = formulas.get(round) {
                selected.push((paper.clone(), tex.clone()));
                took = true;
                if selected.len() >= limit {
                    break;
                }
            }
        }
        if !took {
            break;
        }
        round += 1;
    }

    let out = std::path::absolute(&out)?;
    for sub in ["tex", "pdf", "png"] {
        std::fs::create_dir_all(out.join(sub))?;
    }

    let mut items = Vec::new();
    let mut skipped = 0usize;
    for (idx, (paper, tex)) in selected.into_iter().enumerate() {
        let id = format!("f{idx:03}");
        let item = Item {
            id,
            paper,
            tex,
            png: format!("png/f{idx:03}.png"),
        };
        match render_one(&out, &item, dpi) {
            Ok(()) => items.push(item),
            Err(e) => {
                eprintln!("skip {}: {e}", item.id);
                skipped += 1;
            }
        }
    }

    let manifest = Manifest { dpi, items };
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    println!(
        "wrote {} formulas ({} skipped) to {} at {dpi} dpi",
        manifest.items.len(),
        skipped,
        out.display()
    );
    Ok(())
}
