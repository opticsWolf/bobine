//! Probe: TexTeller ViT encoder latency, CUDA vs CPU.

use ort::{inputs, session::Session, value::Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let path = cache.join("encoder_model.onnx");
    let providers: Vec<String> = std::env::var("BOBINE_ORT_PROVIDERS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    println!("providers: {providers:?}");

    let mut builder = Session::builder()?;
    if !providers.is_empty() {
        let eps: Vec<ort::ep::ExecutionProviderDispatch> = providers
            .iter()
            .filter(|p| p.to_lowercase() == "cuda")
            .map(|_| ort::ep::CUDA::default().build())
            .collect();
        builder = builder.with_execution_providers(&eps)?;
    }
    let mut sess = builder.commit_from_file(&path)?;

    // grayscale 448x448, matching TexTeller preprocessing
    let x = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 1, 448, 448)))?;
    let _ = sess.run(inputs!["pixel_values" => x.clone()]); // warmup
    let t0 = Instant::now();
    for _ in 0..10 {
        let out = sess.run(inputs!["pixel_values" => x.clone()])?;
        // force materialization of the output so timing includes D2H copy
        let _h = out["last_hidden_state"].try_extract_array::<f32>()?.to_owned();
    }
    println!("{:?}/run", t0.elapsed() / 10);
    Ok(())
}
