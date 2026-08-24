# bobine_rs

Fast PDF/Office/text → Markdown ingestion engine. Rust core with Python bindings.

## Install

```bash
pip install bobine-rs
```

## Quick Start

```python
import bobine_rs

config = bobine_rs.ConverterConfig(
    routing_mode=bobine_rs.RoutingMode.Surgical,
)
converter = bobine_rs.HybridConverter(config, cache_dir="~/.cache/bobine")
md = converter.convert_pdf("paper.pdf", work_dir="/tmp/out")
print(md)
```
