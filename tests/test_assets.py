"""Tests for okf-asset:// staging logic (ported from OKFgraph)."""

from bobine.assets import (
    asset_id,
    stage_images_as_okf_assets,
)


class TestAssetStaging:
    def test_asset_id_is_deterministic(self):
        id1 = asset_id("doc", 1, b"test data")
        id2 = asset_id("doc", 1, b"test data")
        assert id1 == id2
        assert id1.startswith("img_")

    def test_asset_id_differs_for_different_data(self):
        assert asset_id("doc", 1, b"data A") != asset_id("doc", 1, b"data B")

    def test_asset_id_differs_for_different_occurrence(self):
        assert asset_id("doc", 1, b"data") != asset_id("doc", 2, b"data")

    def test_stage_images_rewrites_local_links(self, tmp_path):
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        md = f"Text ![alt]({img_file.name})"
        new_md, count = stage_images_as_okf_assets(md, tmp_path, img_file, out_dir, "concept")
        assert count == 1
        assert "okf-asset://" in new_md
        assert img_file.name not in new_md

    def test_stage_images_skips_remote_links(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        md = "Text ![alt](https://example.com/img.png)"
        new_md, count = stage_images_as_okf_assets(
            md, tmp_path, tmp_path / "source.pdf", out_dir, "concept"
        )
        assert count == 0
        assert "https://example.com/img.png" in new_md

    def test_stage_images_skips_okf_asset_links(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        md = "Text ![alt](okf-asset://img_abc123)"
        new_md, count = stage_images_as_okf_assets(
            md, tmp_path, tmp_path / "source.pdf", out_dir, "concept"
        )
        assert count == 0
        assert "okf-asset://img_abc123" in new_md
