# bobine

Fast PDF/Office/text → Markdown ingestion engine. Rust core with Python bindings.

## Install

```bash
pip install -e .
```

## Quick Start

```python
import bobine

config = bobine.ConverterConfig(
    routing_mode=bobine.RoutingMode.Surgical,
)
converter = bobine.HybridConverter(config, cache_dir="~/.cache/bobine")
md = converter.convert_pdf("paper.pdf", work_dir="/tmp/out")
print(md)
```
