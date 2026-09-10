//! Probe: does an accelerator provider actually execute compute?
//! Feeds [16,3,488,488] batches through SLANet-plus repeatedly and reports
//! per-run time; watch nvidia-smi during execution to confirm placement.

use std::time::Instant;

fn bench(tag: &str, providers: &[String], path: &std::path::Path, reps: usize) {
    use ort::{inputs, session::Session, value::Tensor};
    let builder = ort::session::Session::builder().expect("builder");
    let mut builder = if providers.is_empty() {
        builder
    } else {
        let eps: Vec<ort::ep::ExecutionProviderDispatch> = providers
            .iter()
            .filter(|p| p.to_lowercase() == "cuda")
            .map(|_| ort::ep::CUDA::default().build())
            .collect();
        match builder.with_execution_providers(&eps) {
            Ok(b) => b,
            Err(e) => {
                println!("{tag}: EP REGISTRATION FAILED: {e}");
                return;
            }
        }
    };
    let mut sess = builder.commit_from_file(path).expect("commit");
    let x = Tensor::from_array(ndarray::Array4::<f32>::zeros((16, 3, 488, 488))).unwrap();
    let _ = sess.run(inputs!["x" => x.clone()]); // warmup
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = sess.run(inputs!["x" => x.clone()]);
    }
    println!("{tag}: {:?}/run", t0.elapsed() / reps as u32);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let path = cache
        .join("models")
        .join("TabRec")
        .join("SlanetPlus")
        .join("slanet-plus.onnx");
    let providers: Vec<String> = std::env::var("BOBINE_ORT_PROVIDERS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    println!("providers requested: {providers:?}");
    let reps = 20;
    bench("target", &providers, &path, reps);
    bench("cpu ref", &[], &path, reps);
    Ok(())
}
