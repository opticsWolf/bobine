"""Generate P-A differential fixtures: one .mml + .tex per <math> element.

Reads ikat-corpus/jats/*.xml (10 papers), extracts every MathML <math>
element in document order, writes tests/fixtures/mml/fNNNN.mml (raw
element XML) + fNNNN.tex (mml_reference output with the SPECIAL
corpus-hack shim neutralised, per proposal 2.A.3) + manifest.json.

Run from the repo root with the project venv:
  .venv/Scripts/python.exe scripts/gen_mml_fixtures.py
Re-run only when ikat-corpus changes; fixtures ship in git (crates.io
excludes tests/fixtures/ via Cargo.toml `exclude`).
"""

import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ikat-corpus"))

import mml_reference as m  # noqa: E402

MATHML_NS = "http://www.w3.org/1998/Math/MathML"
ET.register_namespace("mml", MATHML_NS)

# Proposal 2.A.3: neutralise the three PLOS-corpus glyph hacks on BOTH
# sides. The Rust port maps them to the passthrough branch ("?"); the
# Python side runs with an empty SPECIAL so outputs stay byte-identical.
m.SPECIAL = {}


def main() -> None:
    jats_dir = ROOT / "ikat-corpus" / "jats"
    out_dir = ROOT / "tests" / "fixtures" / "mml"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale fixtures (idempotent regeneration).
    for p in out_dir.glob("f*.mml"):
        p.unlink()
    for p in out_dir.glob("f*.tex"):
        p.unlink()

    manifest = []
    stats: dict = {}
    idx = 0
    for xml_path in sorted(jats_dir.glob("*.xml")):
        paper = xml_path.stem
        root = ET.parse(xml_path).getroot()
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] != "math":
                continue
            tex = m.mml_to_tex(el, stats)
            (out_dir / f"f{idx:04d}.mml").write_text(
                ET.tostring(el, encoding="unicode"), encoding="utf-8"
            )
            (out_dir / f"f{idx:04d}.tex").write_text(tex, encoding="utf-8")
            manifest.append({"id": idx, "paper": paper})
            idx += 1

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"count": idx, "special_shimmed": True, "fixtures": manifest}, indent=1
        ),
        encoding="utf-8",
    )
    print(f"wrote {idx} fixtures to {out_dir}")
    print("stats:", json.dumps(stats)[:400])


if __name__ == "__main__":
    main()
