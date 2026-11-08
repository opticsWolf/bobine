//! Probe: verify TexTeller model resolution uses the shared hub cache
//! without hitting the network (second run).

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let t0 = Instant::now();
    let tt = bobine::TexTeller::from_pretrained(
        "opticsWolf/texteller3-onnx",
        &cache,
        bobine::ModelPrecision::Fp32,
        &[],
    )?;
    println!("loaded in {:?}", t0.elapsed());
    drop(tt);

    // Hub layout: models--<owner>--<name>/snapshots/<rev>/<file>.
    let repo_dir = std::fs::read_dir(&cache)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("models--opticsWolf--texteller3-onnx"))
        })
        .expect("hub repo dir present");
    let snap = std::fs::read_dir(repo_dir.join("snapshots"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .next()
        .expect("snapshot present");
    for f in [
        "encoder_model.onnx",
        "decoder_model_merged.onnx",
        "tokenizer.json",
    ] {
        assert!(snap.join(f).exists(), "{f} missing");
        println!("cached ok: {}", snap.join(f).display());
    }
    Ok(())
}
