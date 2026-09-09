// Golden-file corpus regressions.
//
// Converts every fixture PDF through the fully deterministic fast path
// (`RoutingMode::Never`, images off — no ONNX, no rendering) and compares
// the markdown against checked-in golden files under tests/golden/.
//
// Regenerate goldens after an *intentional* output change:
//   BOBINE_UPDATE_GOLDENS=1 cargo test --test test_golden

use std::path::{Path, PathBuf};

use bobine::{ConverterConfig, HybridConverter, RoutingMode};

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

    if update || !golden_path.exists() {
        std::fs::write(golden_path, &normalized).unwrap();
        println!("golden written: {}", golden_path.display());
        return;
    }

    let expected = normalize(&std::fs::read_to_string(golden_path).unwrap());
    if expected != normalized {
        mismatches.push(format!("{name}: {}", first_diff(&expected, &normalized)));
    }
}

#[test]
fn corpus_matches_goldens() {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    std::fs::create_dir_all(&golden_dir).unwrap();
    let update = std::env::var("BOBINE_UPDATE_GOLDENS").is_ok();

    let config = ConverterConfig {
        routing_mode: RoutingMode::Never,
        extract_images: false,
        detect_code_blocks: false,
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
        routing_mode: RoutingMode::Never,
        extract_images: true,
        append_unreferenced_images: true,
        detect_code_blocks: false,
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
