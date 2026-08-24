//! Probe: verify TexTeller model resolution uses the flat cache without
//! hitting the network.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let t0 = Instant::now();
    let tt = bobine::TexTeller::from_pretrained(
        "OleehyO/TexTeller",
        &cache,
        bobine::ModelPrecision::Fp32,
        &[],
    )?;
    println!("loaded in {:?}", t0.elapsed());
    drop(tt);

    for f in ["encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json"] {
        assert!(cache.join(f).exists(), "{f} missing");
        println!("cached ok: {}", cache.join(f).display());
    }
    Ok(())
}
