"""Integration tests requiring real native backends (bobine[pdf-ingest]).

Everything here is gated behind the ``integration`` marker and an
``importorskip``, so the suite stays green on a bare install. To run:

    pip install -e ".[pdf-ingest]"        # pdf_oxide, office_oxide, RapidAI
    pytest -m integration
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _synthetic_pdf(path: Path) -> Path:
    """Minimal PDF with extractable text (the classic hand-rolled xref stub)."""
    path.write_bytes(
        b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World from PDF) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000359 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF
"""
    )
    return path


class TestPdfOxide:
    def test_convert_pdf_never_mode(self, tmp_path):
        pytest.importorskip("pdf_oxide")
        from bobine import ConverterConfig, RoutingMode, convert_to_markdown

        pdf = _synthetic_pdf(tmp_path / "hello.pdf")
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        md = convert_to_markdown(pdf, cfg, tmp_path / "work")
        assert isinstance(md, str) and len(md) > 0

    def test_ingest_document_reports_page_count(self, tmp_path):
        pytest.importorskip("pdf_oxide")
        from bobine import ConverterConfig, RoutingMode, ingest_document

        pdf = _synthetic_pdf(tmp_path / "hello.pdf")
        cfg = ConverterConfig(routing_mode=RoutingMode.NEVER, use_onnx=False)
        result = ingest_document(pdf, tmp_path / "out", config=cfg)
        assert result.md_path.exists()
        assert result.page_count == 1
        assert result.md_path.suffix == ".md"

    def test_ingest_document_missing_file(self, tmp_path):
        from bobine import ingest_document

        with pytest.raises(FileNotFoundError):
            ingest_document(tmp_path / "nope.pdf", tmp_path / "out")


class TestOfficeOxide:
    @staticmethod
    def _minimal_docx(path: Path) -> Path:
        """Build a minimal but valid .docx (zip of three XML parts)."""
        import zipfile

        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello from docx</w:t></w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", rels)
            z.writestr("word/document.xml", document)
        return path

    def test_convert_office(self, tmp_path):
        pytest.importorskip("office_oxide")
        from bobine import ConverterConfig, convert_to_markdown

        p = self._minimal_docx(tmp_path / "hello.docx")
        md = convert_to_markdown(p, ConverterConfig(), tmp_path / "work")
        assert "Hello from docx" in md
