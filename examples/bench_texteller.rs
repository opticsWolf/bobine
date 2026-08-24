//! Benchmark TexTeller formula recognition end-to-end.
//!
//! Usage: bench_texteller [--int8] <image.png> [<image2.png> ...]
//!
//! `--int8` loads the onnx-community quantized exports (no KV-cache;
//! full-sequence recompute) instead of the fp32 merged graph.
//!
//! Prints the recognized LaTeX plus per-image timing (3 reps, min) so the
//! full-recompute and KV-cache decoders can be compared for both speed and
//! output parity.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let int8 = args.iter().position(|a| a == "--int8");
    if let Some(pos) = int8 {
        args.remove(pos);
    }
    if args.is_empty() {
        eprintln!("usage: bench_texteller [--int8] <image.png> [...]");
        std::process::exit(2);
    }

    let t0 = Instant::now();
    let mut tt = if int8.is_some() {
        bobine::TexTeller::from_pretrained_int8(&cache, &[])?
    } else {
        bobine::TexTeller::from_pretrained(
            "OleehyO/TexTeller",
            &cache,
            bobine::ModelPrecision::Fp32,
            &[],
        )?
    };
    println!("model load ({}): {:?}", if int8.is_some() { "int8" } else { "fp32+kv" }, t0.elapsed());

    for path in &args {
        // one warmup, then 3 timed reps
        let warm = tt.recognize(std::path::Path::new(path))?;
        let mut best = f64::MAX;
        let mut last = String::new();
        for _ in 0..3 {
            let s = Instant::now();
            let latex = tt.recognize(std::path::Path::new(path))?;
            best = best.min(s.elapsed().as_secs_f64());
            last = latex;
        }
        println!("--- {} ({:.2}s/run)", path, best);
        println!("warmup: {warm}");
        println!("latex : {last}");
    }
    Ok(())
}
