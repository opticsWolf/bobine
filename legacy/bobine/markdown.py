"""Markdown linting for ingested documents (``bobine.markdown``).

Wraps the optional ``mordant`` linter so that:

- fixable formatting issues (MD009 trailing spaces, MD012 multiple blanks,
  MD047 missing final newline) are auto-corrected,
- structural errors (MD001, MD031, MD033) are reported but never block,
- everything else is surfaced as ``unfixable`` diagnostics.

When ``mordant`` is not installed the module degrades to a no-op passthrough
with a single "linter unavailable" diagnostic, so text ingestion still works
with zero dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:
    import mordant  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    mordant = None  # type: ignore[assignment]

# Rules that mordant can auto-fix in-place (whitespace / formatting).
_FIXABLE_RULES = {"MD009", "MD012", "MD047"}
# Rules treated as structural errors (logged, import continues).
_ERROR_RULES = {"MD001", "MD031", "MD033"}


def lint_markdown(content: str, *, auto_fix: bool = True) -> dict[str, Any]:
    """Lint markdown content in memory (no file I/O).

    Returns a dict with the same shape as :func:`lint_markdown_file`:

    - ``content``: the (possibly fixed) markdown content
    - ``fixed``: whether content was modified
    - ``fixed_count``: number of auto-fixed issues
    - ``unfixable``: list of unfixable diagnostics
    - ``errors``: list of error-level diagnostics
    """
    if mordant is None:
        return {
            "content": content,
            "fixed": False,
            "fixed_count": 0,
            "unfixable": [
                {"rule": "E999", "line": 0, "message": "mordant not installed; lint skipped"}
            ],
            "errors": [],
        }

    diagnostics = mordant.lint(content, gfm_opts=mordant.GfmOptions.all())
    if not diagnostics:
        return {
            "content": content,
            "fixed": False,
            "fixed_count": 0,
            "unfixable": [],
            "errors": [],
        }

    unfixable = [d for d in diagnostics if d.rule not in _FIXABLE_RULES]
    errors = [d for d in diagnostics if d.rule in _ERROR_RULES]

    fixed_content = content
    fixed_count = 0
    if auto_fix:
        fixable = [d for d in diagnostics if d.rule in _FIXABLE_RULES]
        if fixable:
            try:
                result = mordant.fix(content, gfm_opts=mordant.GfmOptions.all())
                if result.fixed:
                    fixed_content = result.output
                    fixed_count = len(result.fixed)
            except Exception as e:
                log.warning("mordant.fix failed: %s", e)

    return {
        "content": fixed_content,
        "fixed": fixed_count > 0,
        "fixed_count": fixed_count,
        "unfixable": [d for d in unfixable],
        "errors": [d for d in errors],
    }


def lint_markdown_file(md_path: str | Path, *, auto_fix: bool = True) -> dict[str, Any]:
    """Lint a markdown file and optionally auto-fix fixable issues in place.

    Returns the same dict shape as :func:`lint_markdown`, plus the content is
    written back to ``md_path`` when it was modified and ``auto_fix=True``.
    """
    md_path = Path(md_path)
    content = md_path.read_text(encoding="utf-8")
    result = lint_markdown(content, auto_fix=auto_fix)
    if result["fixed"]:
        md_path.write_text(result["content"], encoding="utf-8")
        log.info("auto-fixed %d issues in %s", result["fixed_count"], md_path.name)
    return result
