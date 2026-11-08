//! Version-scoped cfg for the render-health gate (Option C).
//!
//! Reads the resolved `pdf_oxide` version from `Cargo.lock` and emits
//! `pdf_oxide_pre_0_3_79` when it is older than the cutoff. The gate in
//! `src/render_gate.rs` only fires under that cfg, so bumping the
//! dependency past the cutoff deactivates the shim at compile time with
//! no code change (automatic expiry).
//!
//! Pure std, no dependencies — safe for `cargo check --locked` and the
//! Python-free trusted-publish path.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=build.rs");

    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let version = read_pdf_oxide_version(&lock);
    match version {
        Some(v) => {
            if version_lt(&v, (0, 3, 79)) {
                println!("cargo:rustc-cfg=pdf_oxide_pre_0_3_79");
                println!("cargo:warning=render-health gate ACTIVE (pdf_oxide {v} < 0.3.79)");
            } else {
                println!("cargo:warning=render-health gate EXPIRED (pdf_oxide {v} >= 0.3.79)");
            }
        }
        None => {
            // Registry builds (no Cargo.lock shipped for libs): fail safe —
            // assume the old renderer so corrupt pages fall back instead of
            // emitting confident garbage. Flip by bumping Cargo.toml past
            // the cutoff and deleting the gate (see docs/cid_render_blocker.md).
            println!("cargo:rustc-cfg=pdf_oxide_pre_0_3_79");
            println!(
                "cargo:warning=render-health gate ACTIVE (pdf_oxide version unresolved, assuming < 0.3.79)"
            );
        }
    }
}

fn read_pdf_oxide_version(lock: &Path) -> Option<String> {
    let text = std::fs::read_to_string(lock).ok()?;
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() == r#"name = "pdf_oxide""# {
            for follow in lines.by_ref().take(4) {
                let f = follow.trim();
                if let Some(v) = f.strip_prefix("version = ") {
                    return Some(v.trim_matches('"').to_string());
                }
                if f.starts_with("[[package]]") {
                    break;
                }
            }
        }
    }
    None
}

fn version_lt(v: &str, cutoff: (u64, u64, u64)) -> bool {
    let mut parts = v.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
    let got = (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    );
    got < cutoff
}
