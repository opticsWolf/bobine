# bobine — Benchmark & Test Results

Measured on the reference dev machine (see [Environment](#environment) below).
All numbers are from release builds (`cargo run --release --example …`) with
models pre-cached in `%TEMP%/bobine_test/cache`.

> **Runtime note (0.9.1):** these numbers were taken on ONNX Runtime **1.28.1**;
> bobine now pins **1.31.1** (`bobine[cpu]` / `bobine[gpu]`, and the CI copy).
> Read them as ratios between CPU and CUDA, quantization tiers and table
> routing — the routing defaults they justify are unchanged — not as absolute
> latency for a 1.31.1 install.

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (driver 591.86) |
| ONNX Runtime builds | `D:/User/Documents/Rust/onnxruntime/onnxruntime-win-x64-1.28.1` (CPU) and `onnxruntime-win-x64-gpu_cuda13-1.28.1` (GPU) |
| CUDA toolkit | v13.3 (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64`) |
| cuDNN | v9.24, CUDA-13 build (`C:/Program Files/NVIDIA/CUDNN/v9.24/bin/13.3/x64`) |
| Fixture | `tests/fixtures/formula_sample.png` (TexTeller), synthetic 1024×1024 page (bench_rapid) |

### Reproducing

```bash
export ORT_DYLIB_PATH=D:/User/Documents/Rust/onnxruntime/onnxruntime-win-x64-gpu_cuda13-1.28.1/lib/onnxruntime.dll

# CPU runs
BOBINE_ORT_PROVIDERS="cpu" cargo run --release --example bench_rapid
cargo run --release --example bench_texteller -- tests/fixtures/formula_sample.png           # fp32 + KV-cache
cargo run --release --example bench_texteller -- --int8 tests/fixtures/formula_sample.png    # int8

# Variant matrix (fp32 / fp16 / q8 / int8) on the labeled formula eval set.
# gen_texteller_eval renders the ikat ground-truth LaTeX with pdflatex
# (TinyTeX works) and writes target/texteller_eval/manifest.json.
cargo run --release --example gen_texteller_eval
cargo run --release --example bench_texteller -- --eval target/texteller_eval --all

# GPU (CUDA) runs — PATH must expose cudart/cublas/cudnn first:
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA/CUDNN/v9.24/bin/13.3/x64:$PATH"
BOBINE_ORT_PROVIDERS="cuda"     cargo run --release --example bench_rapid
BOBINE_ORT_PROVIDERS="cuda,cpu" cargo run --release --example bench_texteller -- --eval target/texteller_eval --all
```

## Test results (2026-08-25, bobine 0.4.9)

With a valid `ORT_DYLIB_PATH` (≥ 1.19) set:

| Suite | Result |
|---|---|
| `cargo test --lib` | **89 passed / 0 failed** |
| `tests/test_converter.rs` (PDF integration) | **7 passed** (~145 s; downloads/loads real models) |
| `tests/test_golden.rs` | **1 passed** |
| `tests/test_rapid_table.rs` | **1 passed** |

Without `ORT_DYLIB_PATH`, one lib test fails
(`engine::tests::table_slot_defaults_to_cpu_even_with_cuda_base`) because `ort`
falls back to whatever `onnxruntime.dll` it finds on `PATH` — see the
[pitfall](#pitfall-stale-onnxruntimedll-in-system32) below.

## Rapid pipeline (layout / OCR / table)

Synthetic 1024×1024 page, best-of-30 reps for table, avg-of-N otherwise.

| Model | Load | CPU | CUDA (RTX 3090) | Speedup |
|---|---|---|---|---|
| Layout (PP-DocLayout_plus-L, 1024×1024) | 0.60 s CPU / 0.76 s CUDA | 320 ms/run | **44.6 ms/run** (1 region) | **≈ 7.2×** |
| OCR det+rec (PP-OCRv5, 1024×1024) | 0.20 s CPU / 0.17 s CUDA | 143 ms/run | **38.0 ms/run** | **≈ 3.8×** |
| Table SLANet-plus 700×400 | 0.10–0.14 s both | **28.8 ms/run best** | 49 ms/run best | 0.6× (slower on CUDA) |
| Table SLANet-plus 1024×1024 | — | **133 ms/run best** | 893 ms/run best | 0.15× (much slower on CUDA) |

Confirms the documented routing policy: CUDA is auto-enabled for the layout and
OCR slots (measured here at 7.2× / 3.8×), while table recognition stays pinned
to CPU — SLANet's graph fragments across devices and measures **2–9× slower on
CUDA** (here: 1.7× slower at 700×400, 6.7× slower at 1024×1024).

Note the DETR trade on the layout slot: PP-DocLayout_plus-L is 1.5× faster
than YOLO on CPU (320 vs 480 ms) but slightly slower on CUDA (44.6 vs
37–38 ms) — transformer attention parallelizes less cleanly and parts of
the graph stay on CPU. Either way CUDA wins 7× over CPU, so the routing
policy is unchanged.

### OCR recognition batching (v0.4.28)

`detect_and_recognize` used to run one rec session per detected line. It now
preprocesses all line crops once and recognizes them in chunks of 32
(width-bucketed, zero-padded to the chunk max — exactly what PaddleOCR's own
batch inference feeds), with the greedy CTC decode extracted verbatim per row.
Box geometry is untouched, so positioning is preserved by construction.

Batching is **provider-gated**: CPU-only sessions use chunk size 1, which is
byte-identical to the old path (no padding possible). Rationale, measured on
a dense 47-line page render (`BOB_PAGE_IMG`, release, RTX 3090):

| | Per-line (baseline) | Batched (v0.4.28) |
|---|---|---|
| CPU det+rec | 1.02 s / 47 lines | 0.79 s / **47 lines exact** |
| CUDA det+rec | 2.00 s / 47 lines | **0.32 s** / 46 lines |

Unconditional batching was measured first and rejected: 6.3× faster on CUDA
but **4.8× slower on CPU** (1.02 s → 4.90 s) — padding inflates attention
FLOPs past any launch savings where compute, not launch overhead, dominates.
Bucketing alone did not recover it, hence the gate.

Caveat: padded context participates in attention, so one weak line in 47
decoded to blank under batching on CUDA (46 vs 47 lines; the dropped line
and the e2e diffs on `scanned_page`/`ruled_table` stay within normal OCR
noise — e.g. `Batl B shows systematicaly` vs `Batll B shows sys tema ticaly`
on the same hard table crop). The GPU baseline was verified self-consistent
across processes, so the residual diffs are batching-attributable, not run
noise.

## TexTeller formula OCR

Two measurements on the reference machine (2026-11-01, bobine 0.7.5), both
from release builds:

- **Short fixture** — `tests/fixtures/formula_sample.png`, 3 reps, min
  reported. Fixed cost: one encoder pass plus a short decode.
- **Labeled eval set** — 30 display formulas from the ikat ground-truth
  Markdown, rendered at 300 dpi with `pdflatex` by `gen_texteller_eval`
  (manifest in `target/texteller_eval/`), one timed pass each after a warmup.
  This exercises the long decodes that dominate real crops.

All four variants reproduce the fixture exactly on both providers:
`\[\frac{x^{2}}{a^{2}}- \frac{y^{2}}{b^{2}}=1\]`.

### Short fixture (min of 3 runs)

| Variant | Weights | Load CPU / CUDA | CPU | CUDA | CUDA vs CPU |
|---|---|---|---|---|---|
| fp32 + KV-cache | ~1.19 GB | 2.05 s / 1.72 s | 1.00 s | **0.42 s** | 2.4× |
| fp16 | ~598 MB | 1.42 s / 1.22 s | 2.96 s | 0.53 s | 5.6× |
| q8 (bench only) | ~219 MB | 1.18 s / 1.28 s | 0.67 s | 0.49 s | 1.4× |
| int8 | ~303 MB | 1.46 s / 1.95 s | **0.54 s** | 1.07 s | 0.5× (slower) |

### Labeled eval set (30 formulas, one pass)

| Variant | CPU median | CPU total | CUDA median | CUDA total |
|---|---|---|---|---|
| fp32 | 2.01 s | 69.4 s | 1.49 s | **45.7 s** |
| fp16 | 6.10 s | 228.1 s | **1.21 s** | 47.2 s |
| q8 (bench only) | 1.67 s | 65.3 s | 1.55 s | 69.8 s |
| int8 | **1.05 s** | **39.5 s** | 2.70 s | 95.5 s |

### Accuracy

GT similarity is normalized Levenshtein against the source LaTeX (display
fences, `\left`/`\right`, `\limits`, and whitespace stripped). Exact GT match
is structurally low because TexTeller's house style differs from the
MathML-derived ground truth (`\mathrm`, `\mbox`, `\,` placement), so **parity
against fp32 within the same provider** is the meaningful quantization metric.

| Variant | GT exact (CPU / CUDA) | GT similarity (CPU / CUDA) | Parity vs fp32 CPU | Parity vs fp32 CUDA |
|---|---|---|---|---|
| fp32 | 1/30 / 1/30 | 0.579 / 0.579 | — | — |
| fp16 | 1/30 / 1/30 | 0.579 / 0.579 | 28/30 (0.998) | 29/30 (1.000) |
| q8 (bench only) | 1/30 / 1/30 | 0.459 / 0.459 | 4/30 (0.644) | 4/30 (0.644) |
| int8 | 2/30 / 1/30 | 0.553 / 0.557 | 10/30 (0.830) | 10/30 (0.843) |

Findings:

- **CPU: int8 is the opt-in speed tier** — 1.9× faster than fp32 at the
  median (1.05 s vs 2.01 s) on a quarter of the weights. fp32 stays the
  default because int8 is *not* output-identical: 10/30 exact parity, 0.83
  mean similarity. Most drift is stylistic (`\rm` dropped, `\left(` → `(`,
  `\tfrac`), but one formula collapses to `\[t_{0}=0,e_{0}\]` where fp32
  decodes the full sum.
- **q8 is rejected.** 4/30 parity, and one formula degenerates into a
  repetition loop (2318 characters, 17–20 s) — an unbounded decode is a
  latency hazard, not just an accuracy miss. It stays benchmark-only; the
  library has no q8 constructor.
- **CUDA: fp16 is the opt-in speed tier** — 1.2× faster than fp32 at the
  median on the eval set with half the weights, and 29/30 parity at 1.000
  similarity (the single difference is `\,` placement). fp32 still wins the
  short fixture (0.42 s vs 0.53 s), so the fp16 gain grows with decode length.
- **int8 on CUDA is 1.8× slower than fp32** (2.70 s vs 1.49 s median): ORT
  inserts 171–267 memcpy nodes around the quantized ops. The engine already
  warns when int8 meets a CUDA provider.
- **CUDA helps long decodes far less than the fixture suggests** — 1.35× at
  the median on the eval set vs 2.4× on the fixture — because autoregressive
  steps are launch-latency bound, not FLOP bound.
- **fp16 on CPU is 3× slower than fp32** (no fast f16 path; cast overhead).
  Do not select it there.
- **int8 output is provider-dependent**: only 15/30 formulas match between
  CPU and CUDA, versus 30/30 for fp32 and 29/30 for fp16. int8 results are not
  reproducible across hardware.

Recommendation: keep `model_quantization=Fp32` (the default) for reference
output on either provider. Choose `Int8` only when CPU throughput matters more
than exact reproduction, and `ModelPrecision::Fp16` only on CUDA. The engine
warns on int8 + CUDA and on fp16 without CUDA.

## OCR accuracy vs ikat ground truth (P-D harness — 0.7.1)

Harness: `tests/test_bench_ocr.rs` (`#[ignore]`-gated; needs model cache +
`ORT_DYLIB_PATH`). Forces `RoutingMode::Always`, converts each
`ikat-corpus/pdf/*.pdf`, extracts GT display spans (`$$…$$` + `\tag{N}`)
from `ikat-corpus/ground-truth/*.md`, extracts OCR `$$` spans (multi-line
join + unterminated-fence salvage), aligns by document order, and scores
normalized exact match (whitespace squeeze; `\left`/`\right` identity;
space-insensitive around `{`/`}`/`(`/`)`/`[`/`]`). Env: `BOBINE_BENCH_OCR=1`
to run, `BOBINE_BENCH_PAPERS=` subset, `BOBINE_BENCH_CACHE=` to reuse
`target/bench_ocr/*.md`, `BOBINE_BENCH_RESCORE=1` to re-score saved
outputs without re-running OCR. Writes `report.md` + `mismatches.md`.

Validated on GPU (RTX 3090, ORT 1.28.1 cuda13 + CUDA 13.3 + cuDNN 9.24):
`ORT_DYLIB_PATH` at the `-gpu_cuda13` build, CUDA/cuDNN `bin` dirs on
`PATH`; layout/OCR slots auto-enable CUDA, table stays on CPU.

Two findings, one fixed and one blocking:

1. **Fixed — `rec_input` normalization** (`src/rapid_ocr.rs`): the rec path
   computed `x/255 - 0.5` while the comment and the PP-OCRv4 spec require
   `(x/255 - 0.5) / 0.5` (the det path already did this correctly).
   Half-range inputs reached the rec model. Fixed + unit-test pinned.
2. **Blocker — pdf_oxide 0.3.78 renderer corrupts CID fonts.** The PLOS
   ONE papers that make up the ikat corpus use Type0/CIDFontType2 subset
   fonts; the renderer draws them with shifted glyph IDs, so every render
   carries corrupted text (`RESECRCJ CRTKCLE` for `RESEARCH ARTICLE`).
   All downstream ONNX consumers (RapidOCR text, TexTeller formula crops)
   inherit the corruption, so corpus accuracy numbers would measure the
   renderer bug, not the models. Counter-proof: arXiv-style Type1-only
   PDFs (`tests/fixtures/solitons.pdf`) render and OCR correctly
   (page-0 lines at 0.96–0.99 confidence, verbatim correct). The fast path
   is unaffected (it reads the text layer, whose ToUnicode map is fine).

Status: the harness is implemented and validated; the accuracy table is
**withheld until the upstream renderer is fixed** — publishing the current
0%-exact numbers would misattribute a rasterizer bug to the OCR models.
Re-run `BOBINE_BENCH_OCR=1` (GPU recommended) once renders are correct.
(Full record: `docs/cid_render_blocker.md`.)

## Pitfall: stale onnxruntime.dll in System32

This machine has an old `C:\Windows\System32\onnxruntime.dll` (**v1.17.1**) that
`ort` picks up when `ORT_DYLIB_PATH` is unset, failing with:

```text
Failed to load ONNX Runtime dylib: BadVersion { version_str: "1.17.1", path: "onnxruntime.dll" }
```

The test binary then aborts during shutdown (`STATUS_STACK_BUFFER_OVERRUN`,
"panic in a function that cannot unwind") as `ort`'s exit handler trips over the
poisoned state — the abort is fallout, not the root cause.

Fix options:
1. Always set `ORT_DYLIB_PATH` (recommended; see quickref).
2. Remove/rename the System32 copy (requires admin; other apps may depend on it).

## Summary of machine-local paths

| Purpose | Path |
|---|---|
| ORT CPU 1.28.1 | `D:/User/Documents/Rust/onnxruntime/onnxruntime-win-x64-1.28.1/lib/onnxruntime.dll` |
| ORT GPU (CUDA 13) 1.28.1 | `D:/User/Documents/Rust/onnxruntime/onnxruntime-win-x64-gpu_cuda13-1.28.1/lib/onnxruntime.dll` |
| CUDA 13 runtime DLLs | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64` |
| cuDNN 9.24 (CUDA 13) DLLs | `C:/Program Files/NVIDIA/CUDNN/v9.24/bin/13.3/x64` |
| Model cache | `%TEMP%/bobine_test/cache` |
