//! Text-type document model and frontmatter handling.
//!
//! Owns everything the graph layer needs to *describe* a plain-text /
//! markdown document without touching a database: frontmatter-aware
//! loading, concept-id / title / description / tag normalization, and the
//! OKF-markdown wrapper for raw reasoning text.
//!
//! Frontmatter support is intentionally minimal (flat `key: value` lines,
//! quoted strings, `[a, b]` lists) — enough for the document contract
//! without pulling in a YAML implementation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{BobineError, Result};

/// A normalized, graph-ready document.
///
/// This is the contract between the ingestion pipeline ([`crate::pipeline`])
/// and any graph/import layer. bobine only produces it; it never writes it
/// to a database.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub description: String,
    pub body: String,
    pub doc_type: String,
    pub tags: Vec<String>,
    pub source_path: Option<PathBuf>,
    /// Arbitrary extra frontmatter keys (e.g. `thought_type`, `topic`).
    pub metadata: BTreeMap<String, String>,
}

impl Document {
    /// Serialize back to frontmatter + body markdown.
    ///
    /// Only written metadata keys are emitted; the body is kept verbatim.
    pub fn to_okf_markdown(&self) -> String {
        let mut fm: Vec<(String, String)> = vec![
            ("title".into(), format!("{:?}", self.title)),
            ("type".into(), self.doc_type.clone()),
        ];
        if !self.tags.is_empty() {
            let list = self
                .tags
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            fm.push(("tags".into(), format!("[{list}]")));
        }
        if !self.description.is_empty() {
            fm.push(("description".into(), format!("{:?}", self.description)));
        }
        for (k, v) in &self.metadata {
            fm.push((k.clone(), v.clone()));
        }
        let mut out = String::from("---\n");
        for (k, v) in fm {
            out.push_str(&k);
            out.push_str(": ");
            out.push_str(&v);
            out.push('\n');
        }
        out.push_str("---\n\n");
        out.push_str(&self.body);
        out
    }
}

/// Lowercase the stem and replace spaces with underscores (legacy slug rule).
pub fn slugify(stem: &str) -> String {
    stem.replace(' ', "_").to_lowercase()
}

/// Strip surrounding double quotes from a frontmatter value.
fn unquote(v: &str) -> String {
    let t = v.trim();
    if t.len() >= 2
        && ((t.starts_with('"') && t.ends_with('"')) || (t.starts_with('\'') && t.ends_with('\'')))
    {
        t[1..t.len() - 1].to_string()
    } else {
        t.to_string()
    }
}

/// Parse a `[a, b, c]` list value; `None` when not a list.
fn parse_list(v: &str) -> Option<Vec<String>> {
    let t = v.trim();
    if !(t.starts_with('[') && t.ends_with(']')) {
        return None;
    }
    Some(
        t[1..t.len() - 1]
            .split(',')
            .map(|s| unquote(s))
            .filter(|s| !s.is_empty())
            .collect(),
    )
}

/// Split `content` into `(body, metadata)`.
///
/// Recognizes a leading `---` fenced block with flat `key: value` lines;
/// anything else is returned verbatim as body with empty metadata.
pub fn parse_frontmatter(content: &str) -> (String, BTreeMap<String, String>) {
    let mut meta = BTreeMap::new();
    let trimmed = content.trim_start_matches('\u{feff}');
    let rest = match trimmed.strip_prefix("---") {
        Some(r) => r,
        None => return (content.to_string(), meta),
    };
    // Body starts after the closing "---" line.
    let close = match rest.find("\n---") {
        Some(i) => i,
        None => return (content.to_string(), meta),
    };
    let block = &rest[..close];
    let after_fence = rest[close + 1..]
        .strip_prefix("---")
        .and_then(|s| s.strip_prefix('\r').or(Some(s)))
        .and_then(|s| s.strip_prefix('\n'))
        .unwrap_or("");
    let body = after_fence.to_string();

    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            meta.insert(k.trim().to_string(), unquote(v));
        }
    }
    (body, meta)
}

/// Load a markdown file into a normalized [`Document`].
///
/// Metadata precedence: explicit args > frontmatter > filename-derived.
///
/// - `concept_id`: explicit, else the file stem (lowercased, spaces → `_`).
/// - `title`: explicit, else frontmatter `title`, else the file stem.
/// - `description`: explicit, else frontmatter `description`/`summary`.
/// - `extra_tags`: explicit plus frontmatter `tags` (deduplicated).
pub fn load_markdown_document(
    md_path: &Path,
    concept_id: Option<&str>,
    title: Option<&str>,
    description: Option<&str>,
    extra_tags: &[String],
    default_type: &str,
) -> Result<Document> {
    let content = std::fs::read_to_string(md_path).map_err(|e| {
        BobineError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", md_path.display()),
        ))
    })?;
    let (body, fm) = parse_frontmatter(&content);

    let cid = concept_id
        .map(|s| s.to_string())
        .unwrap_or_else(|| slugify(md_path.file_stem().and_then(|s| s.to_str()).unwrap_or("")));
    let t = title
        .map(|s| s.to_string())
        .or_else(|| fm.get("title").cloned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            md_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string()
        });
    let desc = description
        .map(|s| s.to_string())
        .or_else(|| fm.get("description").cloned())
        .filter(|s| !s.is_empty())
        .or_else(|| fm.get("summary").cloned())
        .unwrap_or_default();

    let mut all_tags: Vec<String> = extra_tags.to_vec();
    if let Some(list) = fm.get("tags").and_then(|v| parse_list(v)) {
        for t in list {
            if !all_tags.contains(&t) {
                all_tags.push(t);
            }
        }
    }

    let dtype = fm
        .get("type")
        .cloned()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_type.to_string());

    let metadata: BTreeMap<String, String> = fm
        .into_iter()
        .filter(|(k, _)| {
            !matches!(
                k.as_str(),
                "title" | "type" | "tags" | "description" | "summary"
            )
        })
        .collect();

    Ok(Document {
        id: cid,
        title: t,
        description: desc,
        body,
        doc_type: dtype,
        tags: all_tags,
        source_path: Some(md_path.to_path_buf()),
        metadata,
    })
}

/// Local timestamp formatted like legacy (`%Y%m%d%H%M%S`, UTC).
fn timestamp_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // days → civil date (Howard Hinnant's algorithm)
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era as i64 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}{mo:02}{d:02}{h:02}{m:02}{s:02}")
}

/// ISO-8601 UTC timestamp (`created:` field of thoughts).
fn iso_now() -> String {
    let ts = timestamp_now();
    format!(
        "{}-{}-{}T{}:{}:{}+00:00",
        &ts[0..4],
        &ts[4..6],
        &ts[6..8],
        &ts[8..10],
        &ts[10..12],
        &ts[12..14]
    )
}

/// Wrap raw reasoning text in OKF-compliant markdown metadata.
///
/// Produces a [`Document`] of type `thought` whose body carries the
/// frontmatter block (title, type, thought_type, topic, tags, created).
pub fn wrap_thoughts(
    thoughts: &str,
    topic: &str,
    concept_id: Option<&str>,
    tags: &[String],
) -> Document {
    let cid = concept_id.map(|s| s.to_string()).unwrap_or_else(|| {
        let ts = timestamp_now();
        let mut slug: String = topic
            .to_lowercase()
            .replace(' ', "_")
            .chars()
            .take(30)
            .collect();
        if slug.is_empty() {
            slug = "topic".into();
        }
        format!(
            "thought_{slug}_{ts}_{}",
            uuid::Uuid::new_v4().simple().to_string()[..6].to_string()
        )
    });

    let header_lines = [
        "---".to_string(),
        format!("title: \"Thought: {topic}\""),
        "type: thought".to_string(),
        "thought_type: reasoning".to_string(),
        format!("topic: {topic}"),
        format!("tags: [thought, reasoning, {topic}]"),
        format!("created: {}", iso_now()),
        "---".to_string(),
        String::new(),
    ];
    let markdown = format!("{}\n{}", header_lines.join("\n"), thoughts);

    let mut all_tags = vec![
        "thought".to_string(),
        "reasoning".to_string(),
        topic.to_string(),
    ];
    for t in tags {
        if !all_tags.contains(t) {
            all_tags.push(t.clone());
        }
    }

    Document {
        id: cid,
        title: format!("Thought: {topic}"),
        description: format!("Reasoning about {topic}"),
        body: markdown,
        doc_type: "thought".to_string(),
        tags: all_tags,
        source_path: None,
        metadata: BTreeMap::from([
            ("thought_type".to_string(), "reasoning".to_string()),
            ("topic".to_string(), topic.to_string()),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(slugify("My Great Doc"), "my_great_doc");
        assert_eq!(slugify("already_fine"), "already_fine");
    }

    #[test]
    fn test_parse_frontmatter_basic() {
        let src = "---\ntitle: \"Hello\"\ntype: note\ntags: [alpha, beta]\n---\n\nBody here.\n";
        let (body, fm) = parse_frontmatter(src);
        assert_eq!(fm.get("title").unwrap(), "Hello");
        assert_eq!(fm.get("type").unwrap(), "note");
        assert_eq!(
            parse_list(fm.get("tags").unwrap()).unwrap(),
            vec!["alpha", "beta"]
        );
        assert!(body.starts_with("\nBody here."));
    }

    #[test]
    fn test_parse_no_frontmatter() {
        let (body, fm) = parse_frontmatter("just text");
        assert_eq!(body, "just text");
        assert!(fm.is_empty());
    }

    #[test]
    fn test_load_precedence_and_dedup() {
        let tmp = std::env::temp_dir().join(format!("bobine_doc_{}.md", std::process::id()));
        std::fs::write(
            &tmp,
            "---\ntitle: FM Title\ndescription: FM desc\ntype: memo\ntags: [x, y]\ncustom: kept\n---\n\nBODY",
        )
        .unwrap();
        let doc = load_markdown_document(
            &tmp,
            None,
            None,
            Some("explicit"),
            &["y".into(), "z".into()],
            "note",
        )
        .unwrap();
        let expected_id = slugify(tmp.file_stem().unwrap().to_str().unwrap());
        assert_eq!(doc.id, expected_id);
        assert_eq!(doc.title, "FM Title");
        assert_eq!(doc.description, "explicit");
        assert_eq!(doc.doc_type, "memo");
        assert_eq!(doc.tags, vec!["y", "z", "x"]); // dedup, explicit first
        assert_eq!(doc.metadata.get("custom").map(String::as_str), Some("kept"));
        assert!(!doc.metadata.contains_key("title"));
        assert_eq!(doc.body, "\nBODY");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_to_okf_round_trip() {
        let doc = Document {
            id: "d".into(),
            title: "T\"q".into(),
            description: "desc".into(),
            body: "the body".into(),
            doc_type: "note".into(),
            tags: vec!["a".into(), "b".into()],
            source_path: None,
            metadata: BTreeMap::from([("k".into(), "v".into())]),
        };
        let md = doc.to_okf_markdown();
        assert!(md.starts_with("---\n"));
        let (body, fm) = parse_frontmatter(&md);
        assert_eq!(fm.get("type").unwrap(), "note");
        assert_eq!(fm.get("k").unwrap(), "v");
        assert_eq!(parse_list(fm.get("tags").unwrap()).unwrap(), vec!["a", "b"]);
        assert!(body.ends_with("the body"));
    }

    #[test]
    fn test_wrap_thoughts_shape() {
        let doc = wrap_thoughts("thinking...", "Rust Ports", None, &[]);
        assert!(doc.id.starts_with("thought_rust_ports_"));
        assert_eq!(doc.doc_type, "thought");
        assert_eq!(doc.tags, vec!["thought", "reasoning", "Rust Ports"]);
        assert_eq!(doc.title, "Thought: Rust Ports");
        assert!(doc.body.contains("thought_type: reasoning"));
        assert!(doc.body.contains("created: "));
        assert!(doc.body.ends_with("\nthinking..."));
    }
}
