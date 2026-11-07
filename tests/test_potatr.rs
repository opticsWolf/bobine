// P0 gate: POTATR preprocess parity against the reference `eval` transform.
//
// Fixture `data/potatr_preprocess.png` + reference tensor
// `data/potatr_preprocess_ref.bin` (raw LE f32, shape in
// `data/potatr_preprocess_ref.json`) were produced by the actual reference
// pipeline (`potatr.datasets.get_transform("eval")`: PIL default resample +
// ToTensor + ImageNet normalize) — see `potatr-export/` (untracked local).
//
// PIL's default resample is BICUBIC (verified empirically on Pillow 13);
// the closest `image`-crate filter (`CatmullRom`) uses different far-lobe
// weights, so bit parity is not expected. The gate allows max|d| <= 0.03
// post-normalize (measured 0.0175, i.e. ~1/255 pre-norm); geometry (scale,
// dims) is exact. Sub-pixel resampler shifts are structure-irrelevant; end-to-end parity
// (GriTS on real pages) is P1's ignored gate.

use bobine::rapid_potatr;

fn load_ref(expected: usize) -> Vec<f32> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let raw = std::fs::read(format!("{dir}/potatr_preprocess_ref.bin"))
        .expect("reference tensor fixture");
    assert_eq!(raw.len() % 4, 0, "f32-aligned fixture");
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(vals.len(), expected, "fixture matches canvas");
    vals
}

#[test]
fn potatr_preprocess_matches_reference_transform() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let img = image::open(format!("{dir}/potatr_preprocess.png"))
        .expect("fixture loads")
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let (arr, geo) = rapid_potatr::preprocess(&image::DynamicImage::ImageRgb8(img))
        .expect("preprocess");

    // Geometry is exact: PIL rounds each dim independently (datasets.py:134).
    let scale = 1000.0 / w.max(h) as f32;
    assert_eq!(geo.resized_w, (w as f32 * scale).round() as u32);
    assert_eq!(geo.resized_h, (h as f32 * scale).round() as u32);
    assert!((geo.scale - scale).abs() < 1e-6);

    let (c_dim, h_dim, w_dim) = (3, geo.resized_h as usize, geo.resized_w as usize);
    assert_eq!(arr.dim(), (1, c_dim, h_dim, w_dim));
    let vals = load_ref(c_dim * h_dim * w_dim);

    // Values within resampler tolerance (see header).
    let (mut max_d, mut max_i) = (0.0f32, 0usize);
    for (i, &v) in vals.iter().enumerate() {
        let c = i / (h_dim * w_dim);
        let y = (i / w_dim) % h_dim;
        let x = i % w_dim;
        let d = (arr[[0, c, y, x]] - v).abs();
        if d > max_d {
            max_d = d;
            max_i = i;
        }
    }
    println!("preprocess parity: max|d| = {max_d:.5} at flat index {max_i}");
    assert!(
        max_d <= 0.03,
        "resampler divergence too large: max|d| = {max_d}"
    );
}
