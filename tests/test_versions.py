"""Tests for RapidAI version pinning (ported from OKFgraph)."""

from bobine.versions import (
    _is_within_tolerance,
    _parse_version,
    check_rapid_versions,
)


class TestVersionChecking:
    def test_parse_version_basic(self):
        assert _parse_version("1.5.2") == (1, 5, 2)
        assert _parse_version("0.2.0") == (0, 2, 0)
        assert _parse_version("10.1.3") == (10, 1, 3)
        assert _parse_version("1") == (1, 0, 0)
        assert _parse_version("1.2") == (1, 2, 0)

    def test_is_within_tolerance_exact_match(self):
        assert _is_within_tolerance("1.5.2", "1.5.2") is True

    def test_is_within_tolerance_same_minor_different_patch(self):
        assert _is_within_tolerance("1.5.1", "1.5.2") is True
        assert _is_within_tolerance("1.5.3", "1.5.2") is True
        assert _is_within_tolerance("1.5.4", "1.5.2") is False

    def test_is_within_tolerance_different_minor(self):
        assert _is_within_tolerance("1.4.9", "1.5.2") is True
        assert _is_within_tolerance("1.6.0", "1.5.2") is True
        assert _is_within_tolerance("1.3.0", "1.5.2") is False
        assert _is_within_tolerance("1.7.0", "1.5.2") is False

    def test_is_within_tolerance_different_major(self):
        assert _is_within_tolerance("2.0.0", "1.5.2") is False
        assert _is_within_tolerance("0.5.2", "1.5.2") is False

    def test_check_rapid_versions_env_silence(self, monkeypatch):
        monkeypatch.setenv("BOBINE_INGEST_ALLOW_UNPINNED", "1")
        assert check_rapid_versions() == []

    def test_check_rapid_versions_legacy_env_silence(self, monkeypatch):
        """The legacy OKFGRAPH_INGEST_ALLOW_UNPINNED name is still honoured."""
        monkeypatch.setenv("OKFGRAPH_INGEST_ALLOW_UNPINNED", "1")
        assert check_rapid_versions() == []

    def test_check_rapid_versions_no_warn_flag(self, monkeypatch):
        monkeypatch.delenv("BOBINE_INGEST_ALLOW_UNPINNED", raising=False)
        monkeypatch.delenv("OKFGRAPH_INGEST_ALLOW_UNPINNED", raising=False)
        warnings = check_rapid_versions(warn=False)
        assert isinstance(warnings, list)

    def test_check_rapid_versions_returns_list(self, monkeypatch):
        monkeypatch.delenv("BOBINE_INGEST_ALLOW_UNPINNED", raising=False)
        monkeypatch.delenv("OKFGRAPH_INGEST_ALLOW_UNPINNED", raising=False)
        assert isinstance(check_rapid_versions(), list)
