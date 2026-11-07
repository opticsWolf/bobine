// POTATR — page-level table extraction via ONNX (P0: model slot + preprocess).
//
// Reference: kensho-technologies/POTATR (`potatr.inference`, `potatr.datasets`).
// Unlike SLANet-plus (`rapid_table.rs`, crop-level), POTATR takes the FULL
// PAGE as input and produces detections + structure + text-bearing cells in
// one forward pass. Page words are consumed downstream by the post-processor
// (P1) — they never enter the network (`POTATR.forward` takes images only).
//
// Model: potatr.onnx (116 MB) from the first-party serving repo
// `opticsWolf/potatr-onnx` (MIT end to end; exported from
// `kensho/POTATR-v1.1-Pub` safetensors at 4f853618…, opset 18, torch 2.15,
// dynamic H/W axes; torch-vs-ORT max|d| ≤ 6e-5; see manifest.json upstream).
// Gate: ikat micro GriTS-Con F1 0.900, TEDS 0.873 (torch side, 39 GT tables).

use std::path::Path;

use image::DynamicImage;
use ndarray::Array4;
use ort::{inputs, session::Session};
use tracing::info;

use crate::error::{BobineError, Result};

/// Longest side of the preprocess canvas (eval transform: `RandomMaxResize(1000, 1000)`).
pub const POTATR_LONGEST_SIDE: u32 = 1000;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Download location of the POTATR ONNX model.
pub const POTATR_REPO: (&str, &str) = ("opticsWolf", "potatr-onnx");
/// First-party serving repo: exported from `kensho/POTATR-v1.1-Pub`
/// safetensors at upstream `4f853618…` (MIT); torch-vs-ORT parity ≤ 6e-5.
pub const POTATR_FILENAME: &str = "potatr.onnx";
/// Pinned hub revision of the serving repo.
pub const POTATR_REV: &str = "e7f1ce67372e5f3972d7c93c8dc639258c6f1944";

/// Fetch potatr.onnx (~116 MB) through the shared hub cache (see
/// [`crate::hub_cache`]), returning its snapshot path to load.
/// `cache_dir` is the hub-cache root. Skips the network entirely when
/// the file is already cached ("using cached").
pub fn download_potatr(cache_dir: &Path) -> Result<std::path::PathBuf> {
    crate::hub_cache::hub_fetch(
        cache_dir,
        POTATR_REPO.0,
        POTATR_REPO.1,
        POTATR_FILENAME,
        "potatr",
        Some(POTATR_REV),
    )
}

/// Geometry of one preprocessed page: P1 needs the exact scale to map the
/// normalized `(cx,cy,h,w)` outputs back onto the original canvas
/// (same convention as `rapid_table.rs::recognize`).
#[derive(Debug, Clone, Copy)]
pub struct PotatrGeometry {
    /// `POTATR_LONGEST_SIDE / max(orig_w, orig_h)`.
    pub scale: f32,
    /// Resized canvas dims actually fed to the network.
    pub resized_w: u32,
    /// Resized canvas dims actually fed to the network.
    pub resized_h: u32,
    /// Original page-render dims.
    pub orig_w: u32,
    /// Original page-render dims.
    pub orig_h: u32,
}

/// Resize longest side to 1000 (aspect-preserving, PIL-rounding), RGB/255,
/// ImageNet-normalize, CHW output — mirroring the reference `eval` transform
/// (`RandomMaxResize(1000,1000)` + `ToTensor` + `Normalize`, `datasets.py:226,244`).
///
/// Resample: PIL `Image.resize` with default resample is BICUBIC on Pillow
/// ≥ 9.1 (verified empirically: default == BICUBIC, != NEAREST); the closest
/// `image`-crate filter is `CatmullRom`. The two kernels differ in the
/// far-lobe weights, so bit parity is NOT expected — the parity test allows
/// max|d| ≤ 3/255 pre-normalize. Sub-pixel shifts are structure-irrelevant
/// (GriTS match radii are far coarser); end-to-end parity is P1's gate.
pub fn preprocess(img: &DynamicImage) -> Result<(Array4<f32>, PotatrGeometry)> {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(BobineError::Other("empty page render".into()));
    }
    let scale = POTATR_LONGEST_SIDE as f32 / w.max(h) as f32;
    // PIL: `int(round(scale * dim))` per axis (`datasets.py:134`).
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);

    let rgb = img.to_rgb8();
    let resized =
        image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::CatmullRom);

    let mut arr = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let p = resized.get_pixel(x as u32, y as u32).0;
            arr[[0, 0, y, x]] = (p[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            arr[[0, 1, y, x]] = (p[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            arr[[0, 2, y, x]] = (p[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }
    }
    let geo = PotatrGeometry {
        scale,
        resized_w: new_w,
        resized_h: new_h,
        orig_w: w,
        orig_h: h,
    };
    Ok((arr, geo))
}

/// Raw forward outputs: node logits `[1,250,17]`, normalized boxes
/// `[1,250,4]` (cx,cy,h,w over the *resized* canvas), edge logits
/// `[1,250,250,2]`. Shapes are dynamic (`ArrayD`) — P1 indexes them by
/// rank. Post-processing (P1) turns these into tables.
pub struct PotatrOutputs {
    /// Class logits per query (16 classes + no-object).
    pub node_logits: ndarray::ArrayD<f32>,
    /// Normalized boxes per query.
    pub boxes: ndarray::ArrayD<f32>,
    /// Edge logits per query pair.
    pub edge_logits: ndarray::ArrayD<f32>,
    /// Geometry to map boxes back to the original canvas.
    pub geometry: PotatrGeometry,
}

/// POTATR page-level table extractor.
pub struct Potatr {
    session: Session,
}

impl Potatr {
    pub fn load(model_path: &Path, providers: &[String]) -> Result<Self> {
        info!("Loading POTATR from {}", model_path.display());
        let session = crate::engine::session_builder(providers)?
            .commit_from_file(model_path)
            .map_err(|e| BobineError::Ort(e.to_string()))?;
        info!("POTATR ready");
        Ok(Self { session })
    }

    /// Run one page: preprocess → forward → raw outputs (P1 post-processes).
    pub fn forward(&mut self, img: &DynamicImage) -> Result<PotatrOutputs> {
        let (input, geometry) = preprocess(img)?;
        let tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| BobineError::Ort(format!("potatr input: {e}")))?;

        // Identify outputs by name (robust to session output ordering),
        // same pattern as `rapid_table.rs::recognize`.
        let (mut node_name, mut box_name, mut edge_name) =
            (String::new(), String::new(), String::new());
        for o in self.session.outputs() {
            let name = o.name().to_string();
            if name.contains("pred_node_logits") {
                node_name = name;
            } else if name.contains("pred_boxes") {
                box_name = name;
            } else if name.contains("pred_edge_logits") {
                edge_name = name;
            }
        }
        if node_name.is_empty() || box_name.is_empty() || edge_name.is_empty() {
            return Err(BobineError::ModelNotAvailable(
                "unexpected POTATR output layout".into(),
            ));
        }

        let outputs = self
            .session
            .run(inputs!["image" => tensor])
            .map_err(|e| BobineError::Ort(format!("potatr run: {e}")))?;

        let node_logits = outputs[node_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr nodes: {e}")))?
            .into_owned();
        let boxes = outputs[box_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr boxes: {e}")))?
            .into_owned();
        let edge_raw = outputs[edge_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| BobineError::Ort(format!("potatr edges: {e}")))?
            .into_owned();
        let edge_logits = edge_raw.into_owned();

        Ok(PotatrOutputs {
            node_logits,
            boxes,
            edge_logits,
            geometry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_geometry_pillow_rounding() {
        // 1500×900: scale = 1000/1500; PIL rounds each dim independently.
        let img = DynamicImage::new_rgb8(1500, 900);
        let (_, geo) = preprocess(&img).expect("preprocess");
        assert_eq!((geo.resized_w, geo.resized_h), (1000, 600));
        assert!((geo.scale - 1000.0 / 1500.0).abs() < 1e-6);
        assert_eq!((geo.orig_w, geo.orig_h), (1500, 900));
    }

    #[test]
    fn test_preprocess_rejects_empty() {
        // Zero-size images cannot exist via DynamicImage constructors, so
        // exercise the guard through a degenerate 1px input instead: it
        // must upscale, not divide by zero.
        let img = DynamicImage::new_rgb8(1, 1);
        let (arr, geo) = preprocess(&img).expect("preprocess");
        assert_eq!((geo.resized_w, geo.resized_h), (1000, 1000));
        assert_eq!(arr.dim(), (1, 3, 1000, 1000));
    }
}
