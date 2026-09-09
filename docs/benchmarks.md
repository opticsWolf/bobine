# bobine — Benchmark & Test Results

Measured on the reference dev machine (see [Environment](#environment) below).
All numbers are from release builds (`cargo run --release --example …`) with
models pre-cached in `%TEMP%/bobine_test/cache`.

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
BOB_ORT_PROVIDERS="cpu" cargo run --release --example bench_rapid
cargo run --release --example bench_texteller -- tests/fixtures/formula_sample.png           # fp32 + KV-cache
cargo run --release --example bench_texteller -- --int8 tests/fixtures/formula_sample.png    # int8

# GPU (CUDA) runs — PATH must expose cudart/cublas/cudnn first:
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA/CUDNN/v9.24/bin/13.3/x64:$PATH"
BOB_ORT_PROVIDERS="cuda"     cargo run --release --example bench_rapid
BOBINE_ORT_PROVIDERS="cuda,cpu" cargo run --release --example bench_texteller -- tests/fixtures/formula_sample.png
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
| Layout (DocLayout-YOLO, 1024×1024) | 0.36 s CPU / 0.48–0.56 s CUDA | 480 ms/run | **37–38 ms/run** (11 regions) | **≈ 12.8×** |
| OCR det+rec (PP-OCRv4, 1024×1024) | 0.20 s both | 131 ms/run | **36 ms/run** | **≈ 3.6×** |
| Table SLANet-plus 700×400 | 0.10–0.14 s both | **28.8 ms/run best** | 49 ms/run best | 0.6× (slower on CUDA) |
| Table SLANet-plus 1024×1024 | — | **133 ms/run best** | 893 ms/run best | 0.15× (much slower on CUDA) |

Confirms the documented routing policy: CUDA is auto-enabled for the layout and
OCR slots (measured here at 12.8× / 3.6×), while table recognition stays pinned
to CPU — SLANet's graph fragments across devices and measures **2–9× slower on
CUDA** (here: 1.7× slower at 700×400, 6.7× slower at 1024×1024).

## TexTeller formula OCR

Per-image timing over 3 reps, min reported; fixture
`tests/fixtures/formula_sample.png`. Recognized LaTeX was identical across all
runs: `\[\frac{x^{2}}{a^{2}}- \frac{y^{2}}{b^{2}}=1\]`.

| Variant | Model load | CPU | CUDA | Notes |
|---|---|---|---|---|
| fp32 merged graph + KV-cache | 2.0 s CPU / 2.4 s CUDA | 0.95 s/run | **0.42 s/run** | ≈ 2.3× on CUDA |
| int8 quantized (split enc/dec) | 1.4 s CPU / 5.0 s CUDA | **0.50 s/run** | 1.32 s/run | int8 wins on CPU; loses on CUDA |

Takeaways:
- On **CPU**, prefer the int8 exports (≈ 1.9× faster than fp32, identical output).
- On **CUDA**, prefer the fp32 + KV-cache graph; the quantized model is slower
  under the CUDA EP (dequant overhead dominates) *and* pays a much larger
  session-init cost.
- The int8 export has no KV-cache (full-sequence recompute), which further
  penalizes it despite being smaller.

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
