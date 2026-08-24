# bobine_rs — Rust Implementation Plan

## Architecture

```
bobine_rs (Rust crate + PyO3 bindings)
│
├── src/
│   ├── lib.rs              # crate root + PyO3 module init
│   ├── config.rs           # ✅ ConverterConfig, RoutingMode, FormulaBackend, ModelPrecision
│   ├── error.rs            # ✅ BobineError
│   ├── tex_teller.rs       # ✅ TexTeller ONNX pipeline (hf-hub download + ort inference)
│   ├── engine.rs           # 🔧 OnnxEngine (TexTeller only; needs RapidOCR/Layout/Table)
│   ├── converter.rs        # ❌ HybridConverter core (20+ methods missing)
│   ├── tables.rs           # ❌ HTML→GFM pipe-table converter
│   ├── assets.rs           # ❌ okf-asset:// staging, image dedup
│   ├── documents.rs        # ❌ Document model, frontmatter
│   ├── pipeline.rs         # ❌ ingest_document, convert_directory
│   └── py_bindings.rs      # ❌ PyO3 #[pyclass] / #[pyfunction] exports
│
├── python/
│   └── bobine_rs/
│       ├── __init__.py     # Re-exports from native module
│       └── py.typed        # PEP 561 marker
│
├── Cargo.toml
├── pyproject.toml          # maturin build config
└── tests/
    └── test_bindings.py    # Python-side integration tests
```

## Phase 1 — Core Converter (pdf_oxide fast path + SURGICAL formula)

### 1.1 converter.rs — HybridConverter methods
- [x] `HybridConverter` struct
- [ ] `_fast_page_markdown(page) -> String`
- [ ] `_extract_page_images(doc, page, index, img_dir) -> Vec<PathBuf>`
- [ ] `image::DynamicImage` ↔ PIL equivalent helpers
- [ ] `_render_page_to_image(doc, page, index, dpi) -> Option<DynamicImage>`
- [ ] `_page_math_signal(page) -> (usize, usize)` — math char count
- [ ] `_is_scanned(page) -> bool`
- [ ] `_needs_onnx(page) -> bool` — routing decision
- [ ] `_math_boxes_from_chars(page) -> Vec<Rect>` — **line-aware merging** (complex)
- [ ] `_crop_image(img, box, page_h, dpi) -> Option<DynamicImage>` — coordinate flip
- [ ] `_region_text(doc, page, index, box) -> String`
- [ ] `_latex_wrap(latex, box, line_height) -> String` — inline vs display
- [ ] `_ws_replace(md, needle, block) -> Option<String>` — whitespace-tolerant
- [ ] `_splice(md, replacements) -> String`
- [ ] `_surgical_page_markdown(doc, page, index, work_dir) -> String`
- [ ] `_route_page(doc, page, index, work_dir) -> String`
- [ ] `convert_pdf(path, work_dir, progress_cb) -> String`
- [ ] `convert_office(path) -> String`
- [ ] `convert(path, work_dir) -> String` — dispatch by extension

### 1.2 engine.rs — RapidOCR/Layout/Table lazy loaders
- [ ] `ocr()` / `ocr_lines(img) -> Vec<(Rect, String, f32)>` — RapidOCR ONNX
- [ ] `layout()` / `layout_regions(img) -> Vec<(Rect, String, f32)>` — RapidLayout ONNX
- [ ] `table()` / `table_html(crop, ocr_results) -> Option<String>` — RapidTable ONNX
- [ ] `_full_structure_page_markdown(doc, page, index, work_dir) -> String`

### 1.3 tables.rs
- [ ] `html_tables_to_gfm(md: &str) -> String`
- [ ] Simple HTML table parser (no rowspan/colspan → bail to raw HTML)

### 1.4 code block detection
- [ ] `_wrap_code_blocks(page) -> Vec<String>` — monospace font detection

## Phase 2 — Python Bindings

### 2.1 py_bindings.rs
- [ ] `#[pyclass] ConverterConfig` — all fields with defaults
- [ ] `#[pyclass] RoutingMode`, `FormulaBackend`, `ModelPrecision` enums
- [ ] `#[pyclass] HybridConverter` — `convert()`, `convert_pdf()`, `recognize_formula()`
- [ ] `#[pyfunction] convert_to_markdown(path, config, work_dir) -> String`
- [ ] `#[pyfunction] ingest_document(path, output_dir, ...) -> ConvertedDocument`
- [ ] `#[pyclass] ConvertedDocument` — `md_path`, `md_text`, `image_count`, `page_count`

### 2.2 pyproject.toml (maturin)
- [ ] `[build-system]` with maturin
- [ ] `[project]` metadata
- [ ] `[tool.maturin]` — bindings = "pyo3", python-source = "python"

### 2.3 Build & test
- [ ] `maturin develop` — installs editable
- [ ] `import bobine_rs` from Python
- [ ] Verify pdf_oxide fast path round-trip
- [ ] Verify TexTeller formula recognition round-trip

## Phase 3 — Pipeline Layer

### 3.1 assets.rs
- [ ] `stage_images_as_okf_assets(md, image_dir, source_path, out_dir, concept_stem) -> (String, usize)`
- [ ] SHA-256 content-addressed asset IDs
- [ ] Image link regex rewrite: `![](local)` → `![](okf-asset://<id>)`

### 3.2 documents.rs
- [ ] `Document` struct with serde
- [ ] `parse_frontmatter(content) -> (body, metadata)`
- [ ] `load_markdown_document(path, ...) -> Document`
- [ ] `to_okf_markdown() -> String`

### 3.3 pipeline.rs
- [ ] `convert_to_markdown(path, config, work_dir, log) -> String`
- [ ] `ingest_document(path, output_dir, ...) -> ConvertedDocument`
- [ ] `stage_images(md, source_path, out_dir, stem) -> (String, usize)`
- [ ] `convert_directory(source_dir, output_dir, ...) -> Vec<ConvertedDocument>`

## Phase 4 — Polish

- [ ] Integration tests against test-PDF corpus
- [ ] Benchmarks (Rust hot loops vs Python)
- [ ] CI (cargo test + maturin build + pytest)
- [ ] Release workflow (maturin publish)
- [ ] Dual license headers

## Order of Implementation (this session)

1. `converter.rs` — `_fast_page_markdown`, `_page_math_signal`, `_is_scanned`, `_needs_onnx`, `_math_boxes_from_chars`, `_crop_image`, `_region_text`, `_latex_wrap`, `_splice`, `_ws_replace`, `_surgical_page_markdown`, `_route_page`, `convert_pdf`, `convert_office`
2. `tables.rs` — HTML → GFM
3. `py_bindings.rs` — PyO3 exports
4. `pyproject.toml` — maturin config
5. Build + smoke test
