// Golden-file corpus regressions.
//
// Converts every fixture PDF through the fully deterministic fast path
// (`RoutingMode::Never`, images off — no ONNX, no rendering) and compares
// the markdown against checked-in golden files under tests/golden/.
//
// Regenerate goldens after an *intentional* output change:
//   BOBINE_UPDATE_GOLDENS=1 cargo test --test test_golden

use std::path::{Path, PathBuf};

use bobine::{ConverterConfig, HybridConverter, RenderOpts, RoutingMode, RoutingOpts, TextOpts};

fn normalize(md: &str) -> String {
    md.replace("\r\n", "\n")
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
        + "\n"
}

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("fixtures dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "pdf").unwrap_or(false))
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no PDF fixtures found");
    out
}

fn first_diff(expected: &str, normalized: &str) -> String {
    // First differing line for a useful failure message.
    expected
        .lines()
        .zip(normalized.lines())
        .position(|(a, b)| a != b)
        .map(|i| {
            format!(
                "line {}: expected {:?}, got {:?}",
                i + 1,
                expected.lines().nth(i).unwrap_or("<eof>"),
                normalized.lines().nth(i).unwrap_or("<eof>")
            )
        })
        .unwrap_or_else(|| "length mismatch".to_string())
}

/// Resolve the golden file for a fixture.
///
/// `ruled_table` is platform-divergent: pdf_oxide 0.3.78's row-banding
/// splits the synthetic grid differently per OS (Windows structures
/// header+A-101 but drops 5 values; Linux structures B-201+B-202 and keeps
/// everything). Both variants are pinned — any third output trips the wire.
/// Scoped to the corpus filename so the figure test is untouched.
fn golden_path_for(golden_path: &Path) -> PathBuf {
    let is_ruled_corpus =
        golden_path.file_name().and_then(|s| s.to_str()) == Some("ruled_table.golden.md");
    if is_ruled_corpus && cfg!(target_os = "linux") {
        golden_path.with_file_name("ruled_table.linux.golden.md")
    } else {
        golden_path.to_path_buf()
    }
}

fn check_fixture(
    name: &str,
    fixture: &Path,
    golden_path: &Path,
    config: &ConverterConfig,
    update: bool,
    work_prefix: &str,
    mismatches: &mut Vec<String>,
) {
    let work = std::env::temp_dir().join(format!("{work_prefix}{name}"));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();

    let mut conv = HybridConverter::new(config.clone(), &work);
    let md = conv.convert_pdf(fixture, &work).unwrap();
    let normalized = normalize(&md);
    let golden_path = golden_path_for(golden_path);

    if update || !golden_path.exists() {
        std::fs::write(golden_path, &normalized).unwrap();
        println!("golden written: {}", golden_path.display());
        return;
    }

    let expected = normalize(&std::fs::read_to_string(golden_path).unwrap());
    if expected != normalized {
        // Full actual (truncated) so CI logs capture platform-divergent
        // output for small fixtures — first_diff alone can't show it.
        const CAP: usize = 4000;
        let shown = if normalized.len() > CAP {
            format!("{}…<{} bytes total>", &normalized[..CAP], normalized.len())
        } else {
            normalized.clone()
        };
        mismatches.push(format!("{name}: {}\n--- actual ---\n{shown}", first_diff(&expected, &normalized)));
    }
}

#[test]
fn corpus_matches_goldens() {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    std::fs::create_dir_all(&golden_dir).unwrap();
    let update = std::env::var("BOBINE_UPDATE_GOLDENS").is_ok();

    let config = ConverterConfig {
        routing: RoutingOpts {
            routing_mode: RoutingMode::Never,
            ..Default::default()
        },
        render: RenderOpts {
            extract_images: false,
            ..Default::default()
        },
        text: TextOpts {
            detect_code_blocks: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut mismatches: Vec<String> = Vec::new();
    for fixture in fixtures() {
        let name = fixture.file_stem().unwrap().to_string_lossy().to_string();
        let golden_path = golden_dir.join(format!("{name}.golden.md"));
        check_fixture(
            &name,
            &fixture,
            &golden_path,
            &config,
            update,
            "bobine_golden_",
            &mut mismatches,
        );
    }

    assert!(
        mismatches.is_empty(),
        "golden regressions detected ({}):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Figure-heavy corpus: same fixtures WITH image extraction and the
/// unreferenced-image gallery enabled. Locks in the interleaved links,
/// page-scoped asset paths (`assets/p{n}/img{k}.{ext}`) and gallery output.
#[test]
fn figure_corpus_matches_goldens() {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    std::fs::create_dir_all(&golden_dir).unwrap();
    let update = std::env::var("BOBINE_UPDATE_GOLDENS").is_ok();

    let config = ConverterConfig {
        routing: RoutingOpts {
            routing_mode: RoutingMode::Never,
            ..Default::default()
        },
        render: RenderOpts {
            extract_images: true,
            append_unreferenced_images: true,
            ..Default::default()
        },
        text: TextOpts {
            detect_code_blocks: false,
            ..Default::default()
        },
        ..Default::default()
    };

    // Only fixtures that actually embed rasters.
    let names = ["2608.05540", "solitons"];

    let mut mismatches: Vec<String> = Vec::new();
    for name in names {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.pdf"));
        let golden_path = golden_dir.join(format!("{name}.figures.golden.md"));
        check_fixture(
            name,
            &fixture,
            &golden_path,
            &config,
            update,
            "bobine_golden_fig_",
            &mut mismatches,
        );
    }

    assert!(
        mismatches.is_empty(),
        "figure-golden regressions detected ({}):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
