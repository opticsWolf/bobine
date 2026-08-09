"""End-to-end pipeline tests (bobine.pipeline) — no heavy deps required.

Covers the text-document path fully (conversion, .md writing, image staging,
okf-asset:// rewriting, linting) plus the batch directory converter. PDF tests
skip when pdf_oxide is not installed.
"""

import base64

import pytest

from bobine import (
    ConverterConfig,
    convert_directory,
    convert_to_markdown,
    ingest_document,
    stage_images,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestConvertToMarkdown:
    def test_text_file_read_as_is(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("plain text content", encoding="utf-8")
        md = convert_to_markdown(p, ConverterConfig(), tmp_path / "work")
        assert md == "plain text content"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_to_markdown(tmp_path / "nope.txt", ConverterConfig(), tmp_path)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"xx")
        with pytest.raises(ValueError):
            convert_to_markdown(p, ConverterConfig(), tmp_path)


class TestStageImages:
    def test_moves_loose_images_and_rewrites_links(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "p0_img0.png").write_bytes(PNG)
        md = "# Doc\n\n![fig](p0_img0.png)\n"

        new_md, count = stage_images(md, tmp_path / "source.pdf", out_dir, "doc")

        assert count == 1
        assert "okf-asset://img_" in new_md
        # the loose file was moved into _assets and no longer sits in out_dir
        assert not (out_dir / "p0_img0.png").exists()
        staged = list((out_dir / "_assets").glob("img_*.png"))
        assert len(staged) == 1
        assert staged[0].read_bytes() == PNG

    def test_remote_links_untouched(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        md = "![r](https://x/y.png)"
        new_md, count = stage_images(md, tmp_path / "s.pdf", out_dir, "doc")
        assert count == 0
        assert "https://x/y.png" in new_md


class TestIngestDocumentText:
    def test_text_document_full_pipeline(self, tmp_path):
        src = tmp_path / "in" / "report.txt"
        src.parent.mkdir(parents=True)
        src.write_text("# Report\n\nSome body.\n", encoding="utf-8")
        out = tmp_path / "out"

        result = ingest_document(src, out)

        assert result.md_path == out / "report.md"
        assert result.md_path.exists()
        assert "# Report" in result.md_text
        assert result.image_count == 0
        assert result.page_count == 0
        assert result.lint["fixed"] is False

    def test_text_document_with_image_reference(self, tmp_path):
        src = tmp_path / "in" / "report.md"
        src.parent.mkdir(parents=True)
        src.write_text("# Report\n\n![diagram](fig.png)\n", encoding="utf-8")
        (src.parent / "fig.png").write_bytes(PNG)
        out = tmp_path / "out"

        result = ingest_document(src, out)

        assert result.image_count == 1
        assert "okf-asset://" in result.md_text
        staged = list((out / "_assets").glob("img_*.png"))
        assert len(staged) == 1
        assert staged[0].read_bytes() == PNG

    def test_extract_images_flag_gates_pdf_extraction(self, tmp_path):
        """extract_images controls PDF image *extraction*, not rewriting of
        image links already present in the markdown (matches OKFgraph semantics:
        stage_images runs unconditionally)."""
        src = tmp_path / "in" / "report.md"
        src.parent.mkdir(parents=True)
        src.write_text("![diagram](fig.png)\n", encoding="utf-8")
        (src.parent / "fig.png").write_bytes(PNG)
        out = tmp_path / "out"

        result = ingest_document(src, out, config=ConverterConfig(extract_images=True))
        assert result.image_count == 1

        # A referenced image is still staged even with extract_images=False
        result2 = ingest_document(src, out, config=ConverterConfig(extract_images=False))
        assert result2.image_count == 1

    def test_missing_document_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_document(tmp_path / "nope.txt", tmp_path / "out")


class TestConvertDirectory:
    def test_batch_converts_text_documents(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("doc a", encoding="utf-8")
        (src / "b.md").write_text("# B", encoding="utf-8")
        (src / "c.txt").write_text("doc c", encoding="utf-8")
        (src / "sub").mkdir()
        (src / "sub" / "d.txt").write_text("doc d", encoding="utf-8")
        (src / "logo.png").write_bytes(PNG)  # asset, not a document

        out = tmp_path / "bundle"
        results = convert_directory(src, out)

        assert len(results) == 4
        names = {r.md_path.name for r in results}
        assert names == {"a.md", "b.md", "c.md", "d.md"}
        assert (out / "logo.png").exists() is False  # images are not converted
        assert all(r.md_path.exists() for r in results)

    def test_batch_skips_unsupported(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("ok", encoding="utf-8")
        (src / "data.bin").write_bytes(b"\x00\x01")
        out = tmp_path / "bundle"

        results = convert_directory(src, out)
        assert len(results) == 1
        assert results[0].md_path.name == "a.md"

    def test_batch_continues_on_failure(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("ok", encoding="utf-8")
        bad = src / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 fake")  # will fail without pdf_oxide

        out = tmp_path / "bundle"
        results = convert_directory(src, out)
        # "ok" text doc converts; the PDF failure is logged, batch continues
        assert len(results) == 1
        assert results[0].md_path.name == "a.md"
