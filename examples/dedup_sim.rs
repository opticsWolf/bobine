//! Simulate the claim-order + rect_subtract dedup from
//! `full_structure_page_markdown` on real layout regions and report where it
//! breaks down (piece-count bailouts, residual char multiplicity).
//!
//! Usage: dedup_sim <pdf> <page|all>

use bobine::ConverterConfig;
use bobine::pdf_source::PdfSource;
use pdf_oxide::api::Pdf;

#[derive(Clone, Copy)]
struct R {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

fn prio(label: &str) -> u8 {
    let l = label.to_lowercase();
    if l.contains("caption") || l.contains("footnote") {
        2
    } else if l.contains("table") {
        1
    } else if l.contains("equation")
        || l.contains("display_formula")
        || l.contains("inline_formula")
        || l.contains("isolate_formula")
        || l.contains("formula")
    {
        0
    } else if l.contains("figure") || l.contains("image") {
        9
    } else {
        3
    }
}

const EPS: f32 = 1e-3;
static mut BAILS: usize = 0;

fn rect_subtract(base: R, subs: &[R]) -> Vec<R> {
    let mut pieces: Vec<R> = vec![base];
    for s in subs {
        if s.w <= 0.0 || s.h <= 0.0 {
            continue;
        }
        let mut next: Vec<R> = Vec::with_capacity(pieces.len() + 4);
        for p in pieces.drain(..) {
            if s.x >= p.x + p.w - EPS
                || s.x + s.w <= p.x + EPS
                || s.y >= p.y + p.h - EPS
                || s.y + s.h <= p.y + EPS
            {
                next.push(p);
                continue;
            }
            if s.y > p.y + EPS {
                next.push(R { x: p.x, y: p.y, w: p.w, h: s.y - p.y });
            }
            if s.y + s.h < p.y + p.h - EPS {
                next.push(R { x: p.x, y: s.y + s.h, w: p.w, h: p.y + p.h - s.y - s.h });
            }
            let sy0 = s.y.max(p.y);
            let sy1 = (s.y + s.h).min(p.y + p.h);
            if s.x > p.x + EPS {
                next.push(R { x: p.x, y: sy0, w: s.x - p.x, h: sy1 - sy0 });
            }
            if s.x + s.w < p.x + p.w - EPS {
                let x0 = (s.x + s.w).max(p.x);
                next.push(R { x: x0, y: sy0, w: p.x + p.w - x0, h: sy1 - sy0 });
            }
        }
        pieces = next.into_iter().filter(|r| r.w > 0.5 && r.h > 0.5).collect();
        if pieces.len() > 64 {
            unsafe { BAILS += 1 };
            return vec![base];
        }
    }
    pieces
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pdf_path = std::env::args().nth(1).expect("pdf");
    let which = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    let cache = std::env::temp_dir().join("bobine_test").join("cache");

    let mut engine = bobine::engine::OnnxEngine::new(&ConverterConfig::default(), &cache);
    engine.ensure_models()?;

    let bytes = std::fs::read(&pdf_path)?;
    let mut pdf = Pdf::from_bytes(bytes).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let pages: Vec<usize> = if which == "all" {
        (0..pdf.page_count()?).collect()
    } else {
        vec![which.parse()?]
    };

    for page in pages {
        unsafe { BAILS = 0 };
        let img = image::load_from_memory(&pdf.render_png(page, 300)?)?;
        let regions = engine.layout_regions(&img)?;
        let scale = 300.0_f32 / 72.0;

        // sort by reading order like the converter
        let mut sorted = regions.clone();
        sorted.sort_by(|a, b| {
            a.y0.partial_cmp(&b.y0).unwrap().then(a.x0.partial_cmp(&b.x0).unwrap())
        });

        // claim order: priority, then y0, x0
        let mut order: Vec<usize> = (0..sorted.len()).collect();
        order.sort_by(|&a, &b| {
            prio(&sorted[a].label).cmp(&prio(&sorted[b].label)).then(
                sorted[a].y0.partial_cmp(&sorted[b].y0).unwrap().then(
                    sorted[a].x0.partial_cmp(&sorted[b].x0).unwrap(),
                ),
            )
        });

        let mut claimed: Vec<R> = vec![];
        // eff_rects indexed by position in `sorted`
        let mut eff: Vec<Vec<R>> = vec![vec![]; sorted.len()];
        let mut max_pieces = 0usize;
        for &i in &order {
            let r = &sorted[i];
            let base = R { x: r.x0, y: r.y0, w: r.x1 - r.x0, h: r.y1 - r.y0 };
            if prio(&r.label) == 9 || base.w <= 0.0 || base.h <= 0.0 {
                eff[i] = vec![base];
                continue;
            }
            let pieces = rect_subtract(base, &claimed);
            max_pieces = max_pieces.max(pieces.len());
            claimed.extend(pieces.iter().cloned());
            eff[i] = pieces;
        }

        // residual char multiplicity under Intersects semantics
        // (skip CLAIM_NONE regions — they emit crops, not text)
        let chars = pdf.chars(page)?;
        let mut emitted = 0usize;
        let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
        for c in &chars {
            let cb = &c.bbox;
            let n = sorted.iter().enumerate()
                .filter(|(i, r)| prio(&r.label) != 9)
                .flat_map(|(i, _)| eff[i].iter())
                .filter(|p| {
                    let px1 = (p.x + p.w) * scale;
                    let py1 = (p.y + p.h) * scale;
                    cb.x < px1 && cb.x + cb.width > p.x * scale
                        && cb.y < py1 && cb.y + cb.height > p.y * scale
                })
                .count();
            if n > 0 { emitted += n; }
            *hist.entry(n).or_default() += 1;
        }
        println!(
            "page {page}: {} regions, max_pieces={max_pieces}, bailouts={} \
             emitted={} chars={} ratio={:.2} hist={:?}",
            sorted.len(), unsafe { BAILS }, emitted, chars.len(),
            emitted as f32 / chars.len().max(1) as f32, hist,
        );
        for (i, r) in sorted.iter().enumerate() {
            println!(
                "  [{i:2}] {:<18} prio={} pt=({:4.0},{:4.0}) {:4.0}x{:4.0} -> {} pieces",
                r.label, prio(&r.label), r.x0 / scale, r.y0 / scale,
                (r.x1 - r.x0) / scale, (r.y1 - r.y0) / scale, eff[i].len(),
            );
        }
    }
    Ok(())
}
