"""Text-type document model and frontmatter handling (``bobine.documents``).

Owns everything the graph layer needs to *describe* a plain-text / markdown
document without touching a database: frontmatter-aware loading, concept-id /
title / description / tag normalization, and the OKF-markdown wrapper for
raw reasoning text.

The only optional dependency is ``python-frontmatter`` (pure Python). When it
is missing, files are read as plain markdown with empty metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# -- optional: python-frontmatter for YAML frontmatter parsing --
try:
    import frontmatter as _frontmatter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _frontmatter = None  # type: ignore[assignment]


@dataclass
class Document:
    """A normalized, graph-ready document.

    This is the contract between the ingestion pipeline (``bobine``) and any
    graph/import layer (e.g. OKFgraph's ``ConceptModel``). ``bobine`` only
    produces it; it never writes it to a database.
    """

    id: str
    title: str
    description: str
    body: str
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    source_path: Path | None = None
    #: arbitrary extra frontmatter keys (e.g. ``thought_type``, ``topic``)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_okf_markdown(self) -> str:
        """Serialize back to frontmatter + body markdown.

        Only written metadata keys are emitted; the body is kept verbatim.
        """
        fm: dict[str, Any] = {"title": self.title, "type": self.type}
        if self.tags:
            fm["tags"] = self.tags
        if self.description:
            fm["description"] = self.description
        fm.update(self.metadata)
        if _frontmatter is not None:
            return _frontmatter.dumps(_frontmatter.Post(self.body, **fm))
        # Fallback: minimal YAML-free dump (blank separator). Keeps round-trip
        # readable; metadata is preserved as a comment-free plain block.
        lines = ["---"]
        for k, v in fm.items():
            lines.append(f"{k}: {v!r}")
        lines += ["---", "", self.body]
        return "\n".join(lines)


def _slugify(stem: str) -> str:
    return stem.replace(" ", "_").lower()


def parse_frontmatter(
    content: str,
    *,
    default_type: str = "note",
) -> tuple[str, dict[str, Any]]:
    """Split ``content`` into ``(body, metadata)``.

    Uses ``python-frontmatter`` when installed; otherwise returns the whole
    content as body with empty metadata.
    """
    if _frontmatter is not None:
        try:
            post = _frontmatter.loads(content)
            return post.content, dict(post.metadata)
        except Exception:
            pass
    return content, {}


def load_markdown_document(
    md_path: str | Path,
    *,
    concept_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    default_type: str = "note",
) -> Document:
    """Load a markdown file into a normalized :class:`Document`.

    Metadata precedence: explicit kwargs > frontmatter > filename-derived.

    - ``concept_id``: explicit, else the file stem (lowercased, spaces → ``_``).
    - ``title``: explicit, else frontmatter ``title``, else the file stem.
    - ``description``: explicit, else frontmatter ``description``/``summary``.
    - ``tags``: explicit plus frontmatter ``tags`` (deduplicated).
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    content = md_path.read_text(encoding="utf-8")
    body, fm = parse_frontmatter(content, default_type=default_type)

    cid = concept_id or _slugify(md_path.stem)
    t = title or fm.get("title") or md_path.stem
    desc = description or fm.get("description") or fm.get("summary") or ""
    file_tags = list(fm.get("tags") or [])
    all_tags = list(dict.fromkeys((tags or []) + file_tags))
    dtype = fm.get("type") or default_type

    return Document(
        id=cid,
        title=t,
        description=desc,
        body=body,
        type=dtype,
        tags=all_tags,
        source_path=md_path,
        metadata={
            k: v
            for k, v in fm.items()
            if k not in ("title", "type", "tags", "description", "summary")
        },
    )


def wrap_thoughts(
    thoughts: str,
    topic: str,
    *,
    concept_id: str | None = None,
    tags: list[str] | None = None,
) -> Document:
    """Wrap raw reasoning text in OKF-compliant markdown metadata.

    Produces a ``Document`` of type ``thought`` whose body carries the
    frontmatter block (title, type, thought_type, topic, tags, created).
    """
    if not concept_id:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        slug = topic.lower().replace(" ", "_")[:30]
        import uuid

        concept_id = f"thought_{slug}_{ts}_{str(uuid.uuid4())[:6]}"

    header_lines = [
        "---",
        f'title: "Thought: {topic}"',
        "type: thought",
        "thought_type: reasoning",
        f"topic: {topic}",
        f"tags: [thought, reasoning, {topic}]",
        f"created: {datetime.now().isoformat()}",
        "---",
        "",
        thoughts,
    ]
    markdown = "\n".join(header_lines)
    all_tags = list(dict.fromkeys(["thought", "reasoning", topic] + (tags or [])))

    return Document(
        id=concept_id,
        title=f"Thought: {topic}",
        description=f"Reasoning about {topic}",
        body=markdown,
        type="thought",
        tags=all_tags,
        metadata={"thought_type": "reasoning", "topic": topic},
    )
