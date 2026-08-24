//! Probe: TexTeller ViT encoder latency, CUDA vs CPU.

use ort::{inputs, session::Session, value::Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = std::env::temp_dir().join("bobine_test").join("cache");
    let path = std::env::var("ENCODER_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| cache.join("encoder_model.onnx"));
    let use_f16 = std::env::var("USE_F16").is_ok();
    let providers: Vec<String> = std::env::var("BOBINE_ORT_PROVIDERS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    println!("providers: {providers:?} | model: {} | f16: {use_f16}", path.display());

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

    let t0 = Instant::now();
    if use_f16 {
        use half::f16;
        let x = Tensor::from_array(
            ndarray::Array4::<f16>::from_elem((1, 1, 448, 448), f16::from_f32(0.5)),
        )?;
        let _ = sess.run(inputs!["pixel_values" => x.clone()]);
        for _ in 0..10 {
            let out = sess.run(inputs!["pixel_values" => x.clone()])?;
            let h = out["last_hidden_state"].try_extract_array::<f16>()?.to_owned();
            // cast back like bobine must do before feeding the f32 decoder
            let _h: ndarray::ArrayD<f32> = h.mapv(|v| v.to_f32());
        }
    } else {
        let x = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 1, 448, 448)))?;
        let _ = sess.run(inputs!["pixel_values" => x.clone()]);
        for _ in 0..10 {
            let out = sess.run(inputs!["pixel_values" => x.clone()])?;
            let _h = out["last_hidden_state"].try_extract_array::<f32>()?.to_owned();
        }
    }
    println!("{:?}/run", t0.elapsed() / 10);
    Ok(())
}
