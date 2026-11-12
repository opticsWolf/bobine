# bobine 0.9 implementation plan

Status: plan, reviewed against both consumer plans 2026-11-11. Branch
`dev_silk` at `34ea72a` (0.8.1 released). Covers items **B2–B6** of the
programme plan (`weave/plugins/silk/docs/ONNX_UNIFICATION_PLAN.md` in
Weave, §4.3) plus the follow-ups 0.8.1 left open.

Sources this plan is accountable to:

- Weave, `weave/plugins/silk/docs/ONNX_UNIFICATION_PLAN.md` §4.3 (B1–B6),
  §3.1 (host wire), §3.2 (identity), §5 (sequence).
- OKFgraph, `docs/implementation-plan-0.14.md` §E and §F — the two steps
  that are blocked on this release, with acceptance criteria written from
  okfgraph's side.

Why 0.9 exists: three consumers need bobine to speak the same words as
everyone else and to stop reloading its models per file.

- **okfgraph 0.14.1** (steps E and F) is blocked on B3 and B4:
  `pdf_device` / `pdf_precision` settings, and one warm converter per
  command with a bounded lifetime.
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
- **Nothing public is removed in 0.9.** `bobine.ingest_document`,
  `HybridConverter.convert/convert_pdf/convert_jats`,
  `bobine.model_status(cache_dir=…)` and `ConverterConfig`'s existing
  fields all keep working; the new surface is additive. okfgraph 0.14.0
  shipped against exactly these.
- Every step ends with `cargo test` + the Python suite green **and** a
  line in `CHANGELOG.md` and `docs/quickref.md`.

---

## 0.5. What the review against the two plans changed

Ten findings, each folded into the step below. Two of them are
corrections to the *other* documents, recorded here so the next reader
does not re-litigate them.

| # | Finding | Resolution |
|---|---|---|
| 1 | Weave §4.3 B3 lists `precision="auto\|fp32\|fp16\|q8\|int8"`; okfgraph §E lists `auto\|fp32\|fp16\|int8` (no `q8`). bobine's enums are `ModelPrecision{Fp32,Fp16}` + `ModelQuantization{Fp32,Int8}` — **there is no q8 anywhere in the stack**. | Canonical list is **`auto\|fp32\|fp16\|int8`**. `q8` is rejected by name with a message saying so. Weave's line is ahead of the artifacts; correct it there. |
| 2 | okfgraph §E says it will read the allowed values "from bobine if it exposes them, else from a local tuple that a test checks against bobine". | **B3 exposes them**: `bobine.DEVICES`, `bobine.PRECISIONS`, so okfgraph keeps no duplicate tuple and no drift test. |
| 3 | okfgraph §E wants `converter_status()` to report "the resolved device and precision". Its `converter_status()` calls `bobine.model_status(cache_dir=…)`, which is **cache** status. | `model_status()` stays cache-only (no signature change, no new fields). Device/precision come from **`describe()` on the converter object** — a session property, not a cache property. okfgraph reads it from the converter it holds. |
| 4 | okfgraph §F holds a `threading.Lock` because "one bobine converter must not run two documents at once"; my first draft had the host answer **409 busy**. | The host **serialises per settings key** (queues). A 409 turns a slow PDF into a spurious client failure, and every consumer already holds its own lock. 409 removed from the route table. |
| 5 | okfgraph §F defines `pdf_idle_seconds=0` as "release after every call". | Same meaning here, same words: `--idle-seconds 0` = release after every conversion, never keep one warm. |
| 6 | okfgraph §F wants a `pdf_converter` doctor row "held or idle, and for how long"; Silk's status wants the same for its pool. | `/v1/models` reports the **warm keys** with `last_used` and `documents` converted, so neither client has to invent a lifetime view. |
| 7 | okfgraph 0.14.0 **step D** maps bobine failures onto `CONVERSION_FAILED` by catching `RuntimeError` (bobine collapses everything into `PyRuntimeError`). A typed exception in 0.9 would silently break that mapping. | **0.9 introduces no new exception type.** If one is ever added it must subclass `RuntimeError`; recorded as a contract, not a preference. |
| 8 | okfgraph §F consumes `md_path` / `image_dir` / `page_count` from the returned object. | Those already exist on `ConvertedDocument` (`src/pipeline.rs:67`), along with `md_text`, `image_count` and `data_files` (Excel's per-sheet CSVs). `ingest_document` returns that type unchanged — no new struct. |
| 9 | B2 bumps `embroider = "0.14"` → `"0.15"` (`Cargo.toml:46`). Weave §4.1 asserts 0.15 left every call bobine makes unchanged. | B2 gains an explicit verification task: bump, `cargo update -p embroider`, `cargo test`, golden suite. If a call moved, stop and report rather than paper over it. |
| 10 | Weave §4.3 B6 says okfgraph's skills "are not yet [in the wheel]; fix both the same way". **Stale** — okfgraph 0.13.1 ships them (`pyproject.toml:85-86`, `okfgraph = ["skills/*/SKILL.md"]`, `okf setup skills`). | B6 copies the *existing* okfgraph pattern. Weave's line should be corrected. |

---

## 1. Cross-repo contract (one page both sides can check)

| Caller | Sends | Expects back |
|---|---|---|
| **okfgraph 0.14.1 E** | `ConverterConfig(device=…, precision=…)`, values validated against `bobine.DEVICES` / `bobine.PRECISIONS` | a construction error naming the word when invalid; `describe()` reporting requested + landed per pass |
| **okfgraph 0.14.1 F** | `HybridConverter(config, cache_dir=…)` held across files; `ingest_document(path, output_dir, on_page=, should_continue=)`; `close()` / `with` | `ConvertedDocument(md_path, image_dir, page_count, …)`; `RuntimeError` on failure (mapped to `CONVERSION_FAILED`); release actually frees sessions |
| **Silk S4** | the same settings words through `bobine_convert.clean_settings`; a warm pool against `python -m bobine.serve` | `/v1/convert` NDJSON progress + result; `/v1/models` with warm keys and `describe()`; `/health` |
| **Silk S5** | nothing | `embroider.runtime_info()`; bobine's own answer: `bobine.ORT_DYLIB_PATH` + `bobine.runtime_info()` |

Rules that fall out of this table:

- **No `onnxruntime` import, ever, at bobine import time** — both Silk
  (S5 reports one runtime for the whole app) and okfgraph (one runtime per
  process) depend on it.
- **One converter, one document at a time**, and the *caller* owns the
  lifetime. bobine never keeps a hidden global pool in-process.
- **Failure stays `RuntimeError`**, so `CONVERSION_FAILED` keeps working.

---

## 2. B2. One runtime locator

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
- Bump `Cargo.toml:46` to `embroider = "0.15"` and call the canonical
  locator before importing `bobine._native`:

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
- Add `bobine.runtime_info() -> {"dylib", "source", "version"}` — the
  same shape as `embroider.runtime_info()` and okfgraph's
  `locate_ort_runtime()`, so Silk's S5 can print one answer per process.

### Verification (finding 9)

1. Bump the dependency, `cargo update -p embroider`, `cargo build`.
2. `cargo test` — if any call site moved in 0.15, stop and report; do not
   adapt silently.
3. Python suite + golden suite unchanged (no re-baseline).

### Tests

- `import bobine` then `assert "onnxruntime" not in sys.modules` — with a
  runtime installed, in the normal test env.
- With no `onnxruntime` present: bobine still imports and reports
  `dylib=None` without raising.
- An explicit `ORT_DYLIB_PATH` wins (spoof the env var, read
  `bobine.ORT_DYLIB_PATH` and `bobine.runtime_info()["source"]`).
- One model-gated conversion still runs, proving the locator found a
  working library.

### Done when

`import bobine` costs no `onnxruntime` import, both accessors answer, and
the golden suite is unchanged.

---

## 3. B3. Shared vocabulary: `device` / `precision`

### Today (measured)

`src/config.rs` speaks **provider lists**: `ort_providers` (~line 127)
plus six per-pass overrides (`encoder_`/`decoder_`/`layout_`/`ocr_`/
`table_`/`potatr_ort_providers`), and separates `ModelPrecision`
(~line 74: `Fp32`, `Fp16`) from `ModelQuantization` (~line 83: `Fp32`,
`Int8`). Every caller therefore translates by hand — Silk's
`functions/bobine_convert.py` does it in Python, okfgraph passes provider
lists too.

### The words (finding 1)

| word | values | default |
|---|---|---|
| `device` | `auto` \| `cpu` \| `cuda` | `auto` |
| `precision` | `auto` \| `fp32` \| `fp16` \| `int8` | `auto` |

`q8` is **not** a bobine word: no artifact in the stack ships q8, and
`ModelQuantization` has no such variant. A `q8` request is refused by
name ("bobine has no q8 artifact; use int8"), so a caller cannot think it
asked for something real. `int8` is today the **TexTeller** compact pair
only; asking for it on a pass without an int8 artifact is refused naming
the pass (open question 1 below is therefore answered, not parked).

Exposed to Python (finding 2), so okfgraph validates against the source:

```python
bobine.DEVICES    # ("auto", "cpu", "cuda")
bobine.PRECISIONS # ("auto", "fp32", "fp16", "int8")
```

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
  `device="cpu"` for a slot whose graph has an fp16-only contract) with a
  `RuntimeError` naming the pass and the reason — fail at construction,
  not mid-page. No new exception class (finding 7).
- `describe()` on `HybridConverter` (finding 3):

  ```python
  {"device_requested": "auto", "precision_requested": "auto",
   "passes": {"layout": {"provider": "CUDAExecutionProvider",
                         "precision": "fp32"},
              "ocr": {...}, "table": {"provider": "CPUExecutionProvider", ...},
              "formula": {...}}}
  ```

  Answerable **before** the first conversion (requested filled, landed
  `None` until sessions open) and after (both). That is what okfgraph's
  `pdf_converter` row and Silk's tooltip read; `model_status()` is not
  touched (finding 3).

### Tests

- The mapping table, in Rust: for each `(device, precision)` pair, the
  provider list each pass receives — including the CPU-pinned table pass
  and the int8 formula default.
- Precedence: a set per-pass list beats the derived default.
- The refusals: unknown word, `q8`, int8 on a pass without an artifact,
  and the impossible combination above.
- Python: `ConverterConfig(device="cuda", precision="fp16")` constructs;
  `device="gpu2"` raises with the word in the message;
  `bobine.PRECISIONS == ("auto", "fp32", "fp16", "int8")` and a
  `describe()` before any conversion has `passes[*]["provider"] is None`.

### Done when (okfgraph §E acceptance)

- `BobineConverter(routing_mode, extract_images, device, precision)`
  passes both values straight into `ConverterConfig` with no translation
  table in okfgraph.
- okfgraph's settings validation reads `bobine.DEVICES` /
  `bobine.PRECISIONS` rather than a local tuple.
- okfgraph's status row can show the landed device and precision from
  `describe()`.

---

## 4. B4. Warm converter: in-process reuse and a persistent host

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
  ) -> ConvertedDocument
  ```

  Full pipeline: convert → stage images into `output_dir` → lint → page
  count → progress callback per page (`on_page(index, total)`, the same
  contract `bobine.ingest_document` already accepts; `should_continue`
  returns `False` to stop between pages). The return type is the existing
  `ConvertedDocument` (`src/pipeline.rs:67`: `md_path`, `md_text`,
  `image_dir`, `image_count`, `page_count`, `data_files`) — no new struct
  (finding 8).
- **Release:** dropping the object frees the sessions (document that
  explicitly — callers set the lifetime; no hidden global pool). Add
  `close()` as a named way to say the same, and make `__enter__` /
  `__exit__` call it, so a caller can be deterministic. `close()` is
  idempotent and leaves the object unusable until it is rebuilt — state
  that, so okfgraph's key-change path (close, rebuild) is well-defined.
- Keep the module-level `bobine.ingest_document(...)` working unchanged
  (it builds, uses and drops one converter) — plus a docstring note that a
  caller converting several files should hold a `HybridConverter`.
- **Thread safety, stated:** one `HybridConverter` runs one document at a
  time; a concurrent call raises `RuntimeError` naming the contract
  (rather than corrupting state). Callers serialise — okfgraph takes a
  lock; the host below serialises per settings key.
- `describe()` from B3 rides this object.

### Change, part 2 — `python -m bobine.serve`

Standard library only (the shape embroider's `serve` established):

| Route | Body / answer |
|---|---|
| `POST /v1/convert` | `{path, output_dir, settings}` → NDJSON: one progress line per page, then the result `{md_path, image_dir, page_count, describe}` |
| `GET /v1/models` | `model_status()` (cache, unchanged) + the warm keys: `[{key, last_used, documents, describe}]` (finding 6) |
| `GET /health` | `{"status": "ok"}` |

- Args: `--host --port --cache-dir [--config key=value ...]
  [--idle-seconds N]`.
- **One `HybridConverter` per settings key**, kept warm; a new key
  constructs one; idle keys are evicted after `--idle-seconds`
  (default 300, `0` = release after every conversion). This is what
  removes Silk's per-call reload.
- **Concurrency (finding 4): serialise per key.** A second request for a
  busy key waits; it is not refused. No 409.
- Cancel: killing the process remains the hard stop; a per-job route is
  explicitly out of scope for 0.9. The NDJSON page lines are what let a
  client decide to kill it.
- Errors: 400 for a bad request/settings, 500 for a conversion failure
  with `{"error": "<bobine message>", "path": ...}` — the message text
  okfgraph's `CONVERSION_FAILED` carries. Never a 200 with a partial
  document.

### Tests

- Counting fake converter (Python): three files through one
  `HybridConverter` → one construction; `close()` releases once and is
  idempotent; a failing file in the middle still releases; `__exit__`
  releases; a use after `close()` raises.
- Concurrency: two threads against one converter serialise (the fake
  records no overlap); a second document is not started while the first
  runs.
- Host: `/v1/convert` over a **text file** (no models) returns NDJSON
  progress + a result; two calls with the same settings key reuse one
  converter (spy on the class); a different key builds a second; two
  concurrent calls on one key queue and both finish; idle eviction with a
  fake clock (`--idle-seconds 0` releases immediately); `/v1/models`
  answers offline and lists warm keys with `last_used`/`documents`;
  400/500 shapes.
- One live model-gated conversion through the host (skipped when the
  models are not cached).

### Done when (okfgraph §F acceptance)

- okfgraph can hold one converter for the files of a command, release it
  on exit, and rebuild it when `(routing_mode, extract_images, device,
  precision)` changes.
- A release actually frees the ONNX sessions (no growth across a
  build/close loop — assert with a repeated loop and a session-count or
  RSS proxy in the model-gated test).
- `should_continue` stops a conversion between pages.
- Silk can point its pool at the host and see warm keys with lifetimes.

---

## 5. B5. Test placement (summary)

| Area | Where |
|---|---|
| Locator without `onnxruntime` imported, `ORT_DYLIB_PATH` precedence | `python/tests/test_runtime_locator.py` (new) |
| Vocabulary table + precedence + refusals + exposed tuples | `tests/test_config_vocabulary.rs` (new, pure Rust) + a Python assert |
| Reuse / release / serialise / use-after-close | `python/tests/test_warm_converter.py` (new) |
| Host routes, warm keys, queueing, idle eviction | `python/tests/test_serve.py` (new) |
| Output unchanged | the existing golden suite, untouched |

---

## 6. B6. CLI and harness skill (deferred, after 0.9)

Not part of 0.9.0. When it lands (finding 10):

- `bobine convert FILE... [--output-dir] [--device] [--precision]
  [--routing-mode]` — many files in one process on **one** warm
  `HybridConverter` (the B4 API, which is why this waits).
- `bobine models` (status, offline), `bobine serve` (the B4 host).
- A harness-neutral `skills/bobine-cli/SKILL.md` shipped **inside the
  wheel** as package data — copy okfgraph's existing pattern
  (`okfgraph = ["skills/*/SKILL.md"]` in its `pyproject.toml`, plus
  `okf setup skills` to copy them into an agent's skill directory), not a
  new one. Weave's "okfgraph's skills are not yet in the wheel" is stale.

---

## 7. Release checklist (0.9.0)

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
7. **Publish before E/F can start.** okfgraph's `pdf` extra floor rises to
   `bobine>=0.9` only once the wheel is on PyPI; that bump is okfgraph's
   commit, not ours.
8. Report to Weave (S6 floors) and to okfgraph (E/F unblocked), and note
   the two corrections this review produced (findings 1 and 10) so the
   programme plan stops carrying them.

---

## 8. Open questions

1. ~~Which passes ship a q8 artifact?~~ **Answered**: none. `int8` is the
   only compact artifact (TexTeller). B3 refuses int8 on other passes by
   name. Revisit only if a q8 export appears.
2. **Host settings key.** One converter per settings key means the key
   must be canonical (sorted, defaults filled, no `None`). Agree the key
   string shape with okfgraph before implementing, so a future
   okfgraph-side host hashes the same thing.
3. **`--idle-seconds` default.** 300 matches okfgraph's
   `pdf_idle_seconds` default; keep both, and let the host's own default
   govern when it runs standalone.
4. **Does `ingest_document` need a `lint_fn` hook?** okfgraph lints with
   mordant before import; bobine's Rust pipeline has its own lint
   callback. For 0.9 keep bobine's internal lint and leave the hook
   unexposed — revisit if okfgraph wants one lint pass instead of two.
5. **Does `describe()` need a `space`?** embroider's identity carries a
   `space` (Weave §3.2) because vectors are compared across processes.
   bobine produces Markdown, not vectors, so there is nothing to pin —
   but say so explicitly in the docstring, or a future reader will add a
   `space` field that means nothing.
