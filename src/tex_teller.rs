// TexTeller ONNX formula recognition engine.
//
// Architecture: ViT encoder → RoBERTa decoder (autoregressive).
// Models from OleehyO/TexTeller on HuggingFace, downloaded via hf-hub.

use std::path::Path;

use hf_hub::HFClientSync;
use image::DynamicImage;
use ndarray::{s, Array4};
use ort::{inputs, session::Session};
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::config::ModelPrecision;
use crate::error::{BobineError, Result};

// ---------------------------------------------------------------------------
// Constants (from TexTeller Python source)
// ---------------------------------------------------------------------------

const FIXED_IMG_SIZE: u32 = 448;
const IMAGE_MEAN: f32 = 0.9545467;
const IMAGE_STD: f32 = 0.15394445;
const MAX_TOKENS: usize = 1024;

/// Full TexTeller pipeline: encoder + decoder + tokenizer.
pub struct TexTeller {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
    /// `true` when the decoder export carries past_key_values/use_cache_branch
    /// inputs (fp32 merged graph); `false` for the int8 community export,
    /// which decodes by full-sequence recompute.
    kv_cache: bool,
}

impl TexTeller {
    /// Download models from HuggingFace Hub and load.
    ///
    /// * `repo` — e.g. `"OleehyO/TexTeller"`
    /// * `cache_dir` — where to store downloaded files
    /// * `precision` — `Fp32` or `Fp16`
    pub fn from_pretrained(
        repo: &str,
        cache_dir: &Path,
        precision: ModelPrecision,
        providers: &[String],
    ) -> Result<Self> {
        Self::from_pretrained_split(repo, cache_dir, precision, None, providers)
    }

    /// Like [`from_pretrained`] but with a separate provider list for the
    /// encoder session (see [`Self::load_with_provider_split`]).
    pub fn from_pretrained_split(
        repo: &str,
        cache_dir: &Path,
        precision: ModelPrecision,
        encoder_providers: Option<&[String]>,
        decoder_providers: &[String],
    ) -> Result<Self> {
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| BobineError::Ort(format!("invalid repo: {repo}")))?;

        let client = HFClientSync::new()
            .map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?;
        let repo_api = client.model(owner, name);

        let suffix = match precision {
            ModelPrecision::Fp16 => "_fp16",
            ModelPrecision::Fp32 => "",
        };

        // hf-hub's single-file + local_dir path always re-downloads (it never
        // checks the destination), so probe for an existing copy first.
        // Files land flat: <cache_dir>/<filename>.
        let mut fetch = |name: String, what: &'static str| -> Result<std::path::PathBuf> {
            let dest = cache_dir.join(&name);
            if dest.exists() {
                info!("TexTeller {what}: using cached {}", dest.display());
                return Ok(dest);
            }
            info!("Downloading TexTeller {what} from {repo}...");
            repo_api
                .download_file()
                .filename(name)
                .local_dir(cache_dir.to_path_buf())
                .send()
                .map_err(|e| BobineError::Ort(format!("download {what}: {e}")))
        };

        let encoder_path = fetch(format!("encoder_model{suffix}.onnx"), "encoder")?;
        let decoder_path = fetch(format!("decoder_model_merged{suffix}.onnx"), "decoder")?;
        let tokenizer_path = fetch("tokenizer.json".to_string(), "tokenizer")?;

        Self::load_with_provider_split(
            &encoder_path,
            &decoder_path,
            &tokenizer_path,
            true,
            encoder_providers,
            decoder_providers,
        )
    }

    /// Load the int8 quantized TexTeller exports from
    /// `Ji-Ha/TexTeller3-ONNX-dynamic` (~319 MB total). Unlike the
    /// onnx-community export, these merged-decoder weights RETAIN the full
    /// KV-cache interface (past_key_values / use_cache_branch / present),
    /// so decoding uses the fast cached path: ~10 ms/step vs ~26 ms fp32
    /// on CPU (measured), i.e. ~2.4x faster formulas at one quarter the
    /// memory. Math content is unaffected; occasional typographic drift
    /// near argmax ties (lost `\mathbf` bold, epsilon glyph variant).
    pub fn from_pretrained_int8(cache_dir: &Path, providers: &[String]) -> Result<Self> {
        Self::from_pretrained_int8_split(cache_dir, None, providers)
    }

    /// Like [`from_pretrained_int8`] but with a separate provider list for
    /// the encoder session (see [`Self::load_with_provider_split`]).
    pub fn from_pretrained_int8_split(
        cache_dir: &Path,
        encoder_providers: Option<&[String]>,
        decoder_providers: &[String],
    ) -> Result<Self> {
        const OWNER: &str = "Ji-Ha";
        const NAME: &str = "TexTeller3-ONNX-dynamic";

        let client = hf_hub::HFClientSync::new()
            .map_err(|e| BobineError::Ort(format!("hf-hub init: {e}")))?;
        let repo_api = client.model(OWNER, NAME);

        // hf-hub's single-file + local_dir path never checks the destination;
        // probe first. Everything lands under <cache>/texteller_int8/ so the
        // variant's files can never collide with the fp32 layout.
        let sub = cache_dir.join("texteller_int8");
        let mut fetch = |name: String, what: &'static str| -> Result<std::path::PathBuf> {
            let dest = sub.join(&name);
            if dest.exists() {
                info!("TexTeller int8 {what}: using cached {}", dest.display());
                return Ok(dest);
            }
            info!("Downloading TexTeller int8 {what} from {OWNER}/{NAME}...");
            repo_api
                .download_file()
                .filename(name)
                .local_dir(sub.clone())
                .send()
                .map_err(|e| BobineError::Ort(format!("download int8 {what}: {e}")))
        };

        let encoder_path = fetch("onnx/encoder_model_int8.onnx".to_string(), "encoder")?;
        let decoder_path = fetch("onnx/decoder_model_merged_int8.onnx".to_string(), "decoder")?;

        // Tokenizer comes from THIS repository root - never share tokenizer
        // files across model variants.
        let tokenizer_path = fetch("tokenizer.json".to_string(), "tokenizer")?;

        Self::load_with_provider_split(
            &encoder_path,
            &decoder_path,
            &tokenizer_path,
            true,
            encoder_providers,
            decoder_providers,
        )
    }

    /// Load from specific ONNX + tokenizer file paths (merged fp32 graph
    /// with KV-cache support).
    pub fn load_from_paths(
        encoder_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
        providers: &[String],
    ) -> Result<Self> {
        Self::load_with_provider_split(
            encoder_path,
            decoder_path,
            tokenizer_path,
            true,
            None,
            providers,
        )
    }

    /// Load from explicit paths with per-session execution-provider
    /// selection. `encoder_providers` overrides the providers used for the
    /// ViT encoder session only (`None` = same as `decoder_providers`) —
    /// useful to offload the compute-bound encoder to a GPU while keeping
    /// the latency-bound autoregressive decoder on CPU.
    pub fn load_with_provider_split(
        encoder_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
        kv_cache: bool,
        encoder_providers: Option<&[String]>,
        decoder_providers: &[String],
    ) -> Result<Self> {
        info!("Loading TexTeller encoder from {}", encoder_path.display());
        let enc_prov = encoder_providers.unwrap_or(decoder_providers);
        let encoder = crate::engine::apply_providers(Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?, enc_prov)?
            .commit_from_file(encoder_path)
            .map_err(|e| BobineError::Ort(e.to_string()))?;

        info!("Loading TexTeller decoder from {}", decoder_path.display());
        let decoder = crate::engine::apply_providers(Session::builder().map_err(|e| BobineError::Ort(e.to_string()))?, decoder_providers)?
            .commit_from_file(decoder_path)
            .map_err(|e| BobineError::Ort(e.to_string()))?;

        info!("Loading TexTeller tokenizer from {}", tokenizer_path.display());
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| BobineError::Tokenizer(format!("tokenizer load: {e}")))?;

        let bos_token_id = tokenizer
            .token_to_id("<s>")
            .ok_or_else(|| BobineError::Tokenizer("BOS token <s> not found".into()))?;
        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .ok_or_else(|| BobineError::Tokenizer("EOS token </s> not found".into()))?;

        info!(
            bos = bos_token_id,
            eos = eos_token_id,
            vocab = tokenizer.get_vocab_size(true),
            "TexTeller ready"
        );

        Ok(Self { encoder, decoder, tokenizer, bos_token_id, eos_token_id, kv_cache })
    }

    /// Convert a single formula crop image → LaTeX string.
    pub fn recognize(&mut self, image_path: &Path) -> Result<String> {
        let img = image::open(image_path)
            .map_err(|e| BobineError::Ort(format!("image open: {e}")))?;
        let array = Self::preprocess(img)?;

        // Encode (scoped to release &mut self.encoder before decoder)
        let encoder_hidden = {
            let pixel_value = ort::value::Tensor::from_array(array)
                .map_err(|e| BobineError::Ort(format!("build pixel_values: {e}")))?;
            let encoder_outputs = self
                .encoder
                .run(inputs!["pixel_values" => pixel_value])
                .map_err(|e| BobineError::Ort(e.to_string()))?;
            let enc = encoder_outputs["last_hidden_state"]
                .try_extract_array::<f32>()
                .map_err(|e| BobineError::Ort(format!("encoder output: {e}")))?;
            enc.to_owned()
        };

        let token_ids = self.autoregressive_decode(&encoder_hidden)?;
        let latex = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| BobineError::Tokenizer(format!("tokenizer decode: {e}")))?;
        Ok(latex)
    }

    // ------------------------------------------------------------------
    // Preprocessing
    // ------------------------------------------------------------------

    fn preprocess(img: DynamicImage) -> Result<Array4<f32>> {
        let rgb = img.to_rgb8();
        let trimmed = Self::trim_white_border_rgb(&rgb);
        let gray = image::imageops::grayscale(&trimmed);

        let (w, h) = gray.dimensions();
        let scale = FIXED_IMG_SIZE as f32 / w.max(h) as f32;
        let new_w = (w as f32 * scale) as u32;
        let new_h = (h as f32 * scale) as u32;

        let resized = image::imageops::resize(
            &gray, new_w, new_h, image::imageops::FilterType::CatmullRom,
        );

        let mut padded = image::GrayImage::new(FIXED_IMG_SIZE, FIXED_IMG_SIZE);
        for y in 0..new_h {
            for x in 0..new_w {
                padded.put_pixel(x, y, *resized.get_pixel(x, y));
            }
        }

        let mut arr = Array4::<f32>::zeros((1, 1, FIXED_IMG_SIZE as usize, FIXED_IMG_SIZE as usize));
        for y in 0..FIXED_IMG_SIZE as usize {
            for x in 0..FIXED_IMG_SIZE as usize {
                let p = padded.get_pixel(x as u32, y as u32);
                let val = p.0[0] as f32 / 255.0;
                arr[[0, 0, y, x]] = (val - IMAGE_MEAN) / IMAGE_STD;
            }
        }
        Ok(arr)
    }

    fn trim_white_border_rgb(img: &image::RgbImage) -> image::RgbImage {
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return img.clone();
        }
        let bg = img.get_pixel(0, 0);
        let threshold: i32 = 15;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x, y);
                let dr = (p.0[0] as i32 - bg.0[0] as i32).abs();
                let dg = (p.0[1] as i32 - bg.0[1] as i32).abs();
                let db = (p.0[2] as i32 - bg.0[2] as i32).abs();
                if dr > threshold || dg > threshold || db > threshold {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        if max_x <= min_x || max_y <= min_y {
            return img.clone();
        }
        let crop_w = max_x - min_x + 1;
        let crop_h = max_y - min_y + 1;
        image::imageops::crop_imm(img, min_x, min_y, crop_w, crop_h).to_image()
    }

    // ------------------------------------------------------------------
    // Autoregressive decode
    // ------------------------------------------------------------------

    fn autoregressive_decode(
        &mut self,
        encoder_hidden: &ndarray::ArrayD<f32>,
    ) -> Result<Vec<u32>> {
        if self.kv_cache {
            self.autoregressive_decode_kv(encoder_hidden)
        } else {
            self.autoregressive_decode_full(encoder_hidden)
        }
    }

    /// Full-sequence recompute decode for exports without KV-cache inputs
    /// (int8 community export): every step re-runs the whole prefix.
    fn autoregressive_decode_full(
        &mut self,
        encoder_hidden: &ndarray::ArrayD<f32>,
    ) -> Result<Vec<u32>> {
        use ort::value::Tensor;

        let enc_value = Tensor::from_array(encoder_hidden.clone())
            .map_err(|e| BobineError::Ort(format!("build encoder states: {e}")))?;
        let mut token_ids: Vec<i64> = vec![self.bos_token_id as i64];

        for _step in 0..MAX_TOKENS {
            let ids_value = Tensor::from_array(
                ndarray::Array2::<i64>::from_shape_vec(
                    (1, token_ids.len()),
                    token_ids.clone(),
                )
                .map_err(|e| BobineError::Ort(format!("build input_ids: {e}")))?,
            )
            .map_err(|e| BobineError::Ort(format!("build input_ids: {e}")))?;

            let outputs = self
                .decoder
                .run(vec![
                    (
                        "input_ids".to_string(),
                        ort::session::SessionInputValue::from(ids_value),
                    ),
                    (
                        "encoder_hidden_states".to_string(),
                        ort::session::SessionInputValue::from(&enc_value),
                    ),
                ])
                .map_err(|e| BobineError::Ort(e.to_string()))?;

            let logits = outputs["logits"]
                .try_extract_array::<f32>()
                .map_err(|e| BobineError::Ort(format!("decoder output: {e}")))?;
            let seq_len = logits.shape()[1];
            let last = logits.slice(ndarray::s![0, seq_len - 1, ..]);
            let next_token = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(self.eos_token_id as i64);

            token_ids.push(next_token);
            if next_token == self.eos_token_id as i64 {
                break;
            }
        }
        Ok(token_ids.into_iter().map(|t| t as u32).collect())
    }

    /// KV-cache decode (fp32 merged graph). See the module comment on
    /// `autoregressive_decode`'s prefill/true-step split and the pinned
    /// encoder caches.
    fn autoregressive_decode_kv(
        &mut self,
        encoder_hidden: &ndarray::ArrayD<f32>,
    ) -> Result<Vec<u32>> {
        use ort::value::Tensor;

        // KV-cache decode via decoder_model_merged.onnx:
        //
        // Step 0 (prefill): feed the whole prompt with empty past and
        // `use_cache_branch=false`; the graph computes every position from
        // scratch AND emits `present.*` caches.
        //
        // Steps 1..: feed only the newest token plus the previous `present.*`
        // as `past_key_values.*` with `use_cache_branch=true` — O(1) per step
        // instead of re-running the whole prefix.
        const LAYERS: usize = 12;

        let mut token_ids: Vec<i64> = vec![self.bos_token_id as i64];

        // Encoder hidden states never change across steps — build once, borrow
        // every step (the old code re-copied ~2.4 MB per step).
        let enc_value = Tensor::from_array(encoder_hidden.clone())
            .map_err(|e| BobineError::Ort(format!("build encoder states: {e}")))?;
        let true_flag = Tensor::from_array(ndarray::arr1(&[true]))
            .map_err(|e| BobineError::Ort(format!("build flag: {e}")))?;
        let false_flag = Tensor::from_array(ndarray::arr1(&[false]))
            .map_err(|e| BobineError::Ort(format!("build flag: {e}")))?;

        // KV state, keyed exactly like the graph's 48 past/present IOs.
        //
        // CAUTION (verified against the real model): the true branch emits a
        // *broken* encoder cache (zero-batch tensors) — cross-attention K/V
        // are therefore harvested from the false-branch prefill ONCE and
        // pinned for the whole decode; only the decoder self-attention caches
        // roll forward step by step.
        let mut dec_cache: Vec<(String, ort::value::Value<ort::value::TensorValueType<f32>>)> =
            Vec::with_capacity(LAYERS * 2);
        let mut enc_cache: Vec<(String, ort::value::Value<ort::value::TensorValueType<f32>>)> =
            Vec::with_capacity(LAYERS * 2);

        for _step in 0..MAX_TOKENS {
            // Feed only the new tokens on cached steps; everything on prefill.
            let prefill = dec_cache.is_empty();
            let (feed_ids, use_cache) = if prefill {
                (token_ids.as_slice(), &false_flag)
            } else {
                (&token_ids[token_ids.len() - 1..], &true_flag)
            };
            let ids_value = Tensor::from_array(
                ndarray::Array2::<i64>::from_shape_vec((1, feed_ids.len()), feed_ids.to_vec())
                    .map_err(|e| BobineError::Ort(format!("build input_ids: {e}")))?,
            )
            .map_err(|e| BobineError::Ort(format!("build input_ids: {e}")))?;

            let mut inputs: Vec<(String, ort::session::SessionInputValue)> =
                Vec::with_capacity(3 + LAYERS * 4);
            inputs.push(("input_ids".into(), ids_value.into()));
            inputs.push(("encoder_hidden_states".into(), (&enc_value).into()));
            inputs.push(("use_cache_branch".into(), (&*use_cache).into()));

            for (name, val) in enc_cache.iter().chain(dec_cache.iter()) {
                inputs.push((name.clone(), val.into()));
            }

            let outputs = self
                .decoder
                .run(inputs)
                .map_err(|e| BobineError::Ort(e.to_string()))?;

            // Argmax over the last position.
            let logits = outputs["logits"]
                .try_extract_array::<f32>()
                .map_err(|e| BobineError::Ort(format!("decoder output: {e}")))?;
            let seq_len = logits.shape()[1];
            let last = logits.slice(ndarray::s![0, seq_len - 1, ..]);
            let next_token = last
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(self.eos_token_id as i64);

            token_ids.push(next_token);

            // Harvest present.* decoder caches for the next step. The
            // encoder caches are NOT refreshed (see note above); they were
            // captured once from the prefill below.
            dec_cache.clear();
            for i in 0..LAYERS {
                for kv in ["key", "value"] {
                    let name = format!("present.{i}.decoder.{kv}");
                    let arr = outputs[name.as_str()]
                        .try_extract_array::<f32>()
                        .map_err(|e| BobineError::Ort(format!("{name}: {e}")))?
                        .to_owned();
                    let val = Tensor::from_array(arr)
                        .map_err(|e| BobineError::Ort(format!("{name}: {e}")))?;
                    dec_cache.push((format!("past_key_values.{i}.decoder.{kv}"), val));
                }
            }
            if enc_cache.is_empty() && prefill {
                // Prefill: capture the cross-attention K/V permanently.
                for i in 0..LAYERS {
                    for kv in ["key", "value"] {
                        let name = format!("present.{i}.encoder.{kv}");
                        let arr = outputs[name.as_str()]
                            .try_extract_array::<f32>()
                            .map_err(|e| BobineError::Ort(format!("{name}: {e}")))?
                            .to_owned();
                        let val = Tensor::from_array(arr)
                            .map_err(|e| BobineError::Ort(format!("{name}: {e}")))?;
                        enc_cache.push((format!("past_key_values.{i}.encoder.{kv}"), val));
                    }
                }
            }

            if next_token == self.eos_token_id as i64 {
                break;
            }
        }
        Ok(token_ids.into_iter().map(|t| t as u32).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal test image (white 100x100, draws a black rectangle)
    fn test_image() -> image::DynamicImage {
        let mut img = image::RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                img.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        // Draw a black formula-like rectangle
        for y in 20..80 {
            for x in 10..90 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        image::DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn preprocess_output_shape() {
        let img = test_image();
        let tensor = TexTeller::preprocess(img).unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 448, 448]);
    }

    #[test]
    fn preprocess_values_in_range() {
        let img = test_image();
        let tensor = TexTeller::preprocess(img).unwrap();
        for v in tensor.iter() {
            assert!(*v > -10.0 && *v < 10.0, "normalized value {v} out of range");
        }
    }

    #[test]
    fn trim_white_border_crops() {
        let mut img = image::RgbImage::new(200, 200);
        for y in 0..200 {
            for x in 0..200 {
                img.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        // Black blob in center
        for y in 50..150 {
            for x in 50..150 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let trimmed = TexTeller::trim_white_border_rgb(&img);
        assert!(trimmed.width() < 200);
        assert!(trimmed.height() < 200);
        assert!(trimmed.width() >= 100);
        assert!(trimmed.height() >= 100);
    }

    #[test]
    fn trim_white_border_all_white() {
        let mut img = image::RgbImage::new(50, 50);
        for y in 0..50 {
            for x in 0..50 {
                img.put_pixel(x, y, image::Rgb([255, 255, 255]));
            }
        }
        let trimmed = TexTeller::trim_white_border_rgb(&img);
        assert_eq!(trimmed.width(), 50); // unchanged when all same color
        assert_eq!(trimmed.height(), 50);
    }
}
