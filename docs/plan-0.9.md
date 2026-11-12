# bobine 0.9 implementation plan

Status: plan, 2026-11-11. Branch `dev_silk` at `34ea72a` (0.8.1
released). Covers items **B2–B6** of the programme plan
(`weave/plugins/silk/docs/ONNX_UNIFICATION_PLAN.md` in Weave, §4.3) plus
the follow-ups 0.8.1 left open.

Why 0.9 exists: three consumers need bobine to speak the same words as
everyone else and to stop reloading its models per file.

- **okfgraph 0.14.1** (its `docs/implementation-plan-0.14.md`, steps E
  and F) is blocked on B3 and B4: `pdf_device` / `pdf_precision`
  settings, and one warm converter per command with a bounded lifetime.
- **Silk** (S4) wants its Converter Settings node to use the same
  `device` / `precision` vocabulary as the ONNX Loader, and a warm
  `bobine.serve` pool instead of one child process per conversion.
- **Weave** (S6) raises the floors once this ships: `bobine>=0.9`.

| Step | Size | Depends on | Unblocks |
|---|---|---|---|
| B2. One runtime locator | small | embroider ≥ 0.15 (`locate_runtime`, `prepare_cuda_dirs`) | one-runtime story everywhere; Silk S5 |
| B3. Shared vocabulary (`device` / `precision`) | small | — | **okfgraph step E**; Silk S4 |
| B4. Warm converter: in-process reuse + host | medium | B3 | **okfgraph step F**; Silk's per-call reload |
| B5. Tests per step | small | each step | — |
| B6. CLI + harness skill (deferred) | medium | B2–B4 | a shell story; `bobine` from a skill |

B2 and B3 are independent and small; B4 is the release's reason. Ship
**0.9.0** when B2–B5 are green. B6 follows in a later release.

Standing rules for every step:

- **The ORT pins stay `==1.31.1`** (`cpu` / `gpu[cuda,cudnn]`). B1 fixed
  a real install conflict; nothing in this plan may drift them again.
- **`import bobine` must not import `onnxruntime`.** After B2 that is a
  test, not a hope.
- **The conversion output is golden-tested.** None of this changes what
  lands in the Markdown; the golden suite must stay green without
  re-baselining. A re-baseline means the change altered output — stop and
  find out why.
- Every step ends with `cargo test` + the Python suite green **and** a
  line in `CHANGELOG.md` and `docs/quickref.md`.

---

## B2. One runtime locator

### Today (measured)

`python/bobine/__init__.py:29-54` — `_ensure_ort_dylib()` **imports
`onnxruntime` at module import** to find `capi/onnxruntime.dll` (or the
`.so`/`.dylib`), then `_warm_gpu_dlls()` imports it again to call
`preload_dlls()`. That is the same import that puts a second runtime into
a host process, and it is the second resolver in a stack that now has a
canonical one: okfgraph delegates to `embroider.locate_runtime()` since
0.13 (its `components/embedding.py:1067`), which walks dist-info and
never imports the module.

### Change

- Delete `_ensure_ort_dylib` and `_warm_gpu_dlls`.
- Before importing `bobine._native`, call the canonical locator's APIs from
  **embroider's Rust core** (bump the dependency to `embroider = "0.15"`):

  ```rust
  embroider::diag::locate_runtime(&hints, /*search_prefix=*/true, /*set_env=*/true)
  embroider::diag::prepare_cuda_dirs(None)
  ```

  `locate_runtime` honours an explicit `ORT_DYLIB_PATH` first, then a
  pip-installed `onnxruntime` / `onnxruntime-gpu` resolved without
  importing it; `prepare_cuda_dirs` folds what `_warm_gpu_dlls` did
  (`cudnn64_9.dll` and CUDA siblings resolve for the GPU build).
- Keep the public knob: **`bobine.ORT_DYLIB_PATH`** stays a module
  attribute, now reporting what the locator found (or `None`).
- Expose the locator's report too, so Silk's S5 status can print where
  the runtime came from: `bobine.runtime_info() -> {"dylib", "source",
  "version"}` (thin wrapper over the native call, mirroring
  `embroider.runtime_info()`).

### Tests (B5 half)

- `import bobine` then `assert "onnxruntime" not in sys.modules` — with
  a runtime installed, in the normal test env.
- A venv with `oncudnn`/`onnxruntime-gpu` absent still imports bobine and
  reports `dylib=None` without raising.
- An explicit `ORT_DYLIB_PATH` wins (spoof the env var, read
  `bobine.ORT_DYLIB_PATH`).
- One conversion test (behind the existing model-gated marker) still
  runs, proving the locator found a working library.

### Done when

`import bobine` costs no `onnxruntime` import, `bobine.ORT_DYLIB_PATH`
and `bobine.runtime_info()` answer, and the golden suite is unchanged.

---

## B3. Shared vocabulary: `device` / `precision`

### Today (measured)

`src/config.rs` speaks **provider lists**: `ort_providers` (~line 127)
plus six per-pass overrides (`encoder_`/`decoder_`/`layout_`/`ocr_`/
`table_`/`potatr_ort_providers`), and separates `ModelPrecision`
(~line 74) from `ModelQuantization` (~line 83). Every caller therefore
translates by hand — Silk's `functions/bobine_convert.py` does it in
Python, okfgraph passes provider lists too.

### The words

Same as embroider and okfgraph, so one row means one thing everywhere:

| word | values | default |
|---|---|---|
| `device` | `auto` \| `cpu` \| `cuda` | `auto` |
| `precision` | `auto` \| `fp32` \| `fp16` \| `q8` \| `int8` | `auto` |

### Change

- `ConverterConfig(device=..., precision=...)` in `src/config.rs`, both
  parsed through the shared vocabulary:
  - `device` → `embroider::policy::DeviceReq::{Auto,Cpu,Cuda}`
    (`DeviceReq::parse` already exists and rejects anything else with a
    readable message), mapped onto the provider lists exactly as today so
    the existing routing policy — **CUDA helps layout/OCR, SLANet-plus
    tables stay CPU-pinned, TexTeller defaults to int8 CPU** — is
    preserved by construction, not re-decided.
  - `precision` → the existing `ModelPrecision` / `ModelQuantization`
    pair, with the same per-pass defaults (TexTeller int8).
- **Precedence, explicit:** a per-pass provider list (`layout_ort_providers`
  etc.) *wins* over the `device`-derived default whenever it is set. That
  keeps every expert override working and makes the mapping testable as a
  table rather than an argument.
- Reject a combination that cannot work (e.g. `precision="fp16"` with
  `device="cpu"` for a slot whose graph has an fp16-only contract) with
  `BobineError::Other` naming the pass and the reason — fail at
  construction, not mid-page.
- `describe()` on `HybridConverter` returns, per pass: the landed provider
  and the precision actually loaded, plus the byte-identical `device` /
  `precision` the caller asked for. Python shape:

  ```python
  {"device_requested": "auto", "precision_requested": "auto",
   "passes": {"layout": {"provider": "CUDAExecutionProvider",
                         "precision": "fp32"},
              "ocr": {...}, "table": {"provider": "CPUExecutionProvider", ...},
              "formula": {...}}}
  ```

### Tests (B5 half)

- The mapping table, in Rust: for each `(device, precision)` pair, the
  provider list each pass receives — including the CPU-pinned table pass
  and the int8 formula default.
- Precedence: a set per-pass list beats the derived default.
- The refusals: unknown word, and the impossible combination above.
- Python: `ConverterConfig(device="cuda", precision="fp16")` constructs;
  `device="gpu2"` raises with the word in the message.

### Done when

Silk's `clean_settings` can pass `device` / `precision` straight through,
and okfgraph can add `pdf_device` / `pdf_precision` without bobine-side
translation tables.

---

## B4. Warm converter: in-process reuse and a persistent host

### Today (measured)

- `bobine.HybridConverter` (Python) exposes `convert`, `convert_pdf`,
  `convert_jats` — **no `ingest_document`**, so a caller that holds one
  object still cannot stage images, count pages or report progress; it
  can only get a Markdown string or a file.
- `src/pipeline.rs:83 convert_to_markdown` builds a **fresh**
  `HybridConverter::new(config.clone(), cache_dir)` per call (lines
  100/105), and `ingest_document` goes through it. So layout + OCR
  sessions are rebuilt for every file — the measured reason Silk runs a
  child process per conversion and okfgraph pays a reload per PDF.

### Change, part 1 — `HybridConverter.ingest_document` (in-process reuse)

- Move the pipeline body into a method the object owns:

  ```python
  HybridConverter(config, cache_dir="...").ingest_document(
      path, output_dir, on_page=None, should_continue=None,
  ) -> ConvertedDocument   # md_path, image_dir, page_count
  ```

  Full pipeline: convert → stage images into `output_dir` → lint →
  page count → progress callback per page (`on_page(index, total)`, the
  same contract `bobine.ingest_document` already accepts).
- **Release:** dropping the object frees the sessions (document that
  explicitly — callers set the lifetime; no hidden global pool). Add
  `close()` as a named way to say the same, and make `__exit__` call it,
  so a caller can be deterministic.
- Keep the module-level `bobine.ingest_document(...)` working unchanged
  (it builds, uses and drops one converter) — plus a note in its
  docstring that a caller converting several files should hold a
  `HybridConverter` instead.
- **Thread safety, stated:** one `HybridConverter` runs one document at a
  time. Callers serialise (okfgraph takes a lock; the host below owns
  one converter per settings key and serialises per key). Document it in
  the quickref rather than discovering it under load.
- `describe()` from B3 rides this object, so a caller can report what
  the warm sessions landed on.

### Change, part 2 — `python -m bobine.serve`

Standard library only (the shape embroider's `serve` established):

| Route | Body / answer |
|---|---|
| `POST /v1/convert` | `{path, output_dir, settings}` → NDJSON: one progress line per page, then the result `{md_path, image_dir, page_count, describe}` |
| `GET /v1/models` | `model_status()` + the landed devices/precisions (offline: reports the cache, never downloads) |
| `GET /health` | `{"status": "ok"}` |

- Args: `--host --port --cache-dir [--config key=value ...]`.
- **One `HybridConverter` per settings key**, kept warm; a new key
  constructs one; idle keys are evicted after `--idle-seconds`
  (default 300, `0` = never keep one warm). This is what removes Silk's
  per-call reload.
- Cancel: killing the process remains the hard stop; a per-job route is
  explicitly out of scope for 0.9.
- Errors: 400 for a bad request/settings, 409 when the requested key's
  converter is busy (one document at a time), 500 for a conversion
  failure with `{"error": "<bobine message>", "path": ...}` — the code
  okfgraph will map onto `CONVERSION_FAILED`.

### Tests (B5 half)

- Counting fake converter (Python): three files through one
  `HybridConverter` → one construction; `close()` releases once;
  a failing file in the middle still releases; `__exit__` releases.
- Concurrency: two threads against one converter serialise (the fake
  records overlap) or raise the documented busy error — whichever the
  implementation states.
- Host: `/v1/convert` over a **text file** (no models) returns NDJSON
  progress + a result; two calls with the same settings key reuse one
  converter (spy on the class); a different key builds a second; idle
  eviction with a fake clock (`--idle-seconds 0` releases immediately);
  `/v1/models` answers offline; 400/409/500 shapes.
- One live model-gated conversion through the host (skipped when the
  models are not cached).

### Done when

okfgraph's step F can hold one converter for the files of a command and
release it after `pdf_idle_seconds`, and Silk can replace its child
process per file with a warm pool — without either of them translating
settings by hand.

---

## B5. Test placement (summary)

| Area | Where |
|---|---|
| Locator without `onnxruntime` imported, `ORT_DYLIB_PATH` precedence | `python/tests/test_runtime_locator.py` (new) |
| Vocabulary table + precedence + refusals | `tests/test_config_vocabulary.rs` (new, pure Rust) |
| Reuse / release / serialise | `python/tests/test_warm_converter.py` (new) |
| Host routes, warm keys, idle eviction | `python/tests/test_serve.py` (new) |
| Output unchanged | the existing golden suite, untouched |

---

## B6. CLI and harness skill (deferred, after 0.9)

Not part of 0.9.0. When it lands:

- `bobine convert FILE... [--output-dir] [--device] [--precision]
  [--routing-mode]` — many files in one process on **one** warm
  `HybridConverter` (the B4 API, which is why this waits).
- `bobine models` (status, offline), `bobine serve` (the B4 host).
- A harness-neutral `skills/bobine-cli/SKILL.md` shipped **inside the
  wheel** as package data — the pattern okfgraph adopted in 0.13.1
  (`okfgraph/skills/*/SKILL.md` + `okf setup skills`); bobine should copy
  the pattern rather than invent one.

---

## Release checklist (0.9.0)

1. `Cargo.toml` + `pyproject.toml` → `0.9.0`; `embroider = "0.15"`;
   **pins stay `onnxruntime==1.31.1` / `onnxruntime-gpu[cuda,cudnn]==1.31.1`**.
2. `cargo test` (lib + integration) and the Python suite green; golden
   suite **not** re-baselined.
3. `CHANGELOG.md` `## [0.9.0]` — the three user-visible changes (no
   `onnxruntime` import, `device`/`precision`, warm converter + host) and
   the release's one-line summary.
4. `docs/quickref.md` + `docs/architecture.md`: the vocabulary table, the
   in-process reuse rule ("one converter, one document at a time, caller
   owns the lifetime"), the host routes.
5. `pages/index.html`: badge + published-stack row.
6. Tag **on `main`** after a fast-forward from `dev_silk` — same rule the
   other two repos use. Then crates.io + PyPI (0.8.1 published both).
7. Report the release to Weave so S6 can raise `bobine>=0.9`, and to
   okfgraph so its `pdf` extra floor rises with E/F.

---

## Open questions

1. **`precision="q8"` for which passes?** `ModelQuantization` exists, but
   which slots ship a measured q8 artifact today is a table nobody has
   written down. B3 should refuse a q8 request for a slot without one,
   naming the slot — get the artifact list from the model cache rather
   than inventing it.
2. **Host settings key.** One converter per settings key means the key
   must be canonical (sorted, defaults filled). Agree the key string
   shape with okfgraph before implementing, so both sides hash the same
   thing.
3. **`--idle-seconds` default.** 300 matches okfgraph's planned
   `pdf_idle_seconds`; fine to keep both and let the host's own default
   govern when it runs standalone.
4. **Does `ingest_document` need a `lint_fn` hook?** okfgraph lints with
   mordant before import; bobine's own lint callback exists in the Rust
   pipeline. For 0.9 keep bobine's internal lint and leave the hook
   unexposed — revisit if okfgraph wants one lint pass instead of two.
