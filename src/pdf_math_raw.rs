//! Narrow raw-object access for formula metadata omitted by pdf_oxide's
//! `StructElem` projection. Object loading, reference resolution, and stream
//! decoding remain owned by pdf_oxide.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use pdf_oxide::{
    document::PdfDocument,
    object::{Object, ObjectRef},
    structure::{McidScope, StructChild, StructElem, StructTreeRoot, StructType},
};
use serde_json::Value;

const MAX_EMBEDDED_NAME_TREE_NODES: usize = 10_000;
const MAX_EMBEDDED_NAME_TREE_TIME: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Default)]
pub(crate) struct FormulaMetadata {
    pub description: Option<String>,
    pub namespace: Option<Value>,
    pub attributes: Option<Value>,
    pub associated_files: Vec<AssociatedFileMetadata>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AssociatedFileMetadata {
    pub filename: Option<String>,
    pub relationship: Option<String>,
    pub mime_type: Option<String>,
}

#[derive(Debug)]
struct RawMcid {
    mcid: u32,
    page_object_id: Option<u32>,
    scope: RawMcidScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RawMcidScope {
    Page(Option<u32>),
    Form(ObjectRef),
    Pattern(ObjectRef),
}

#[derive(Debug, Default)]
struct RawFormula {
    metadata: FormulaMetadata,
    page_object_id: Option<u32>,
    page_anchor_seen: bool,
    mcids: Vec<RawMcid>,
}

#[derive(Debug, Default)]
struct ProjectedFormula {
    page: Option<u32>,
    mcids: Vec<(u32, McidScope, u32)>,
}

/// Read raw metadata from Formula dictionaries and align it to the parsed
/// projection. Any mismatch fails closed for the entire lookup.
pub(crate) fn aligned_formula_metadata(
    document: &PdfDocument,
    tree: &StructTreeRoot,
    max_nodes: usize,
) -> std::result::Result<Vec<FormulaMetadata>, String> {
    if max_nodes == 0 {
        return Ok(Vec::new());
    }
    if !document.is_authenticated() {
        return Err("PDF is encrypted and not authenticated".to_string());
    }

    let catalog = document
        .catalog()
        .map_err(|error| format!("catalog: {error}"))?;
    let catalog_dict = catalog
        .as_dict()
        .ok_or_else(|| "catalog is not a dictionary".to_string())?;
    let root = catalog_dict
        .get("StructTreeRoot")
        .ok_or_else(|| "catalog has no StructTreeRoot".to_string())?;
    let root = document
        .resolve_object(root)
        .map_err(|error| format!("resolve StructTreeRoot: {error}"))?;
    let root_dict = root
        .as_dict()
        .ok_or_else(|| "StructTreeRoot is not a dictionary".to_string())?;
    let raw_k = root_dict
        .get("K")
        .ok_or_else(|| "StructTreeRoot has no /K".to_string())?;
    let resolved_k = document
        .resolve_object(raw_k)
        .map_err(|error| format!("resolve StructTreeRoot /K: {error}"))?;

    let mut walker = RawFormulaWalker {
        document,
        role_map: &tree.role_map,
        max_nodes: max_nodes.min(10_000),
        started: Instant::now(),
        visited: HashSet::new(),
        active_formulas: Vec::new(),
        formulas: Vec::new(),
        node_count: 0,
        exceeded: false,
    };
    walker.walk_root(&resolved_k)?;
    if walker.exceeded || walker.started.elapsed() > Duration::from_secs(2) {
        return Err("raw Formula lookup reached its traversal budget".to_string());
    }

    let mut projected = Vec::new();
    let mut projected_nodes = 0usize;
    for elem in &tree.root_elements {
        collect_projected_formulas(
            elem,
            None,
            &mut projected_nodes,
            max_nodes.min(10_000),
            &mut projected,
        );
        if projected_nodes >= max_nodes.min(10_000) {
            break;
        }
    }
    if projected_nodes >= max_nodes.min(10_000) && projected.len() != walker.formulas.len() {
        return Err("parsed Formula lookup reached its traversal budget".to_string());
    }
    if projected.len() != walker.formulas.len() {
        return Err(format!(
            "raw and parsed Formula counts differ ({} raw, {} parsed)",
            walker.formulas.len(),
            projected.len()
        ));
    }

    let mut raw_page_to_page = HashMap::<u32, u32>::new();
    let mut page_to_raw_page = HashMap::<u32, u32>::new();
    for (index, (raw, parsed)) in walker.formulas.iter().zip(&projected).enumerate() {
        if raw.mcids.len() != parsed.mcids.len()
            || raw
                .mcids
                .iter()
                .zip(&parsed.mcids)
                .any(|(raw_mcid, (parsed_mcid, _, _))| raw_mcid.mcid != *parsed_mcid)
        {
            return Err(format!(
                "raw and parsed MCID order differs at Formula {index}"
            ));
        }
        if !align_page(
            raw.page_object_id,
            parsed.page,
            &mut raw_page_to_page,
            &mut page_to_raw_page,
        )? {
            return Err(format!(
                "raw and parsed page association differs at Formula {index}"
            ));
        }
        for (raw_mcid, (_, parsed_scope, parsed_page)) in raw.mcids.iter().zip(&parsed.mcids) {
            if !align_page(
                raw_mcid.page_object_id,
                Some(*parsed_page),
                &mut raw_page_to_page,
                &mut page_to_raw_page,
            )? {
                return Err(format!(
                    "raw and parsed MCID page association differs at Formula {index}"
                ));
            }
            match (&raw_mcid.scope, parsed_scope) {
                (RawMcidScope::Page(_), McidScope::Page(_)) => {}
                (RawMcidScope::Form(raw_ref), McidScope::Form(parsed_ref))
                    if raw_ref == parsed_ref => {}
                (RawMcidScope::Pattern(raw_ref), McidScope::Pattern(parsed_ref))
                    if raw_ref == parsed_ref => {}
                _ => {
                    return Err(format!(
                        "raw and parsed MCID scope differs at Formula {index}"
                    ));
                }
            }
        }
    }

    Ok(walker
        .formulas
        .into_iter()
        .map(|formula| formula.metadata)
        .collect())
}

fn align_page(
    raw_page_object_id: Option<u32>,
    parsed_page: Option<u32>,
    raw_page_to_page: &mut HashMap<u32, u32>,
    page_to_raw_page: &mut HashMap<u32, u32>,
) -> std::result::Result<bool, String> {
    match (raw_page_object_id, parsed_page) {
        (Some(object_id), Some(page)) => {
            if raw_page_to_page
                .get(&object_id)
                .is_some_and(|seen| *seen != page)
                || page_to_raw_page
                    .get(&page)
                    .is_some_and(|seen| *seen != object_id)
            {
                return Ok(false);
            }
            raw_page_to_page.insert(object_id, page);
            page_to_raw_page.insert(page, object_id);
            Ok(true)
        }
        (None, Some(page)) => Ok(page == 0),
        (Some(_), None) => Ok(false),
        (None, None) => Ok(true),
    }
}

fn collect_projected_formulas(
    elem: &StructElem,
    inherited_page: Option<u32>,
    visited: &mut usize,
    max_nodes: usize,
    out: &mut Vec<ProjectedFormula>,
) {
    if *visited >= max_nodes {
        return;
    }
    *visited += 1;
    let page = elem.page.or(inherited_page);
    if elem.struct_type == StructType::Formula {
        let mut formula = ProjectedFormula {
            page: page.or_else(|| first_marked_page(elem)),
            mcids: Vec::new(),
        };
        collect_projected_mcids(elem, &mut formula.mcids);
        out.push(formula);
    }
    for child in &elem.children {
        if *visited >= max_nodes {
            break;
        }
        if let StructChild::StructElem(child) = child {
            collect_projected_formulas(child, page, visited, max_nodes, out);
        }
    }
}

fn collect_projected_mcids(elem: &StructElem, out: &mut Vec<(u32, McidScope, u32)>) {
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { mcid, scope, page } => {
                out.push((*mcid, scope.clone(), *page))
            }
            StructChild::StructElem(child) => collect_projected_mcids(child, out),
            StructChild::ObjectRef(_, _) => {}
        }
    }
}

fn first_marked_page(elem: &StructElem) -> Option<u32> {
    for child in &elem.children {
        match child {
            StructChild::MarkedContentRef { page, .. } => return Some(*page),
            StructChild::StructElem(child) => {
                if let Some(page) = first_marked_page(child) {
                    return Some(page);
                }
            }
            StructChild::ObjectRef(_, _) => {}
        }
    }
    None
}

struct RawFormulaWalker<'a> {
    document: &'a PdfDocument,
    role_map: &'a HashMap<String, String>,
    max_nodes: usize,
    started: Instant,
    visited: HashSet<u32>,
    active_formulas: Vec<usize>,
    formulas: Vec<RawFormula>,
    node_count: usize,
    exceeded: bool,
}

impl RawFormulaWalker<'_> {
    fn walk_root(&mut self, k: &Object) -> std::result::Result<(), String> {
        if let Some(children) = k.as_array() {
            for child in children {
                if self.out_of_budget() {
                    break;
                }
                let resolved = if let Some(reference) = child.as_reference() {
                    if !self.visited.insert(reference.id) {
                        continue;
                    }
                    self.document
                        .load_object(reference)
                        .map_err(|error| format!("load root structure element: {error}"))?
                } else {
                    self.document
                        .resolve_object(child)
                        .map_err(|error| format!("resolve root structure element: {error}"))?
                };
                self.walk_struct_elem(&resolved, None)?;
            }
        } else {
            self.walk_struct_elem(k, None)?;
        }
        Ok(())
    }

    fn walk_struct_elem(
        &mut self,
        object: &Object,
        inherited_page: Option<ObjectRef>,
    ) -> std::result::Result<(), String> {
        if !self.count_node() {
            return Ok(());
        }
        let Some(dict) = object.as_dict() else {
            return Ok(());
        };
        if dict
            .get("Type")
            .and_then(Object::as_name)
            .is_some_and(|name| name != "StructElem")
        {
            return Ok(());
        }
        let Some(role) = dict.get("S").and_then(Object::as_name) else {
            return Ok(());
        };
        let mapped_role = self.role_map.get(role).map(String::as_str).unwrap_or(role);
        let is_formula = StructType::from_str(mapped_role) == StructType::Formula;
        let own_page = dict.get("Pg").and_then(Object::as_reference);
        let effective_page = own_page.or(inherited_page);
        let formula_index = if is_formula {
            let mut formula = RawFormula::default();
            formula.page_object_id = effective_page.map(|page| page.id);
            formula.page_anchor_seen = formula.page_object_id.is_some();
            formula.metadata = self.formula_metadata(dict);
            let index = self.formulas.len();
            self.formulas.push(formula);
            self.active_formulas.push(index);
            Some(index)
        } else {
            None
        };

        if let Some(k_raw) = dict.get("K") {
            let k = if let Some(reference) = k_raw.as_reference() {
                let resolved = self
                    .document
                    .load_object(reference)
                    .map_err(|error| format!("load structure /K: {error}"))?;
                if resolved.as_dict().is_some() && !self.visited.insert(reference.id) {
                    if formula_index.is_some() {
                        self.active_formulas.pop();
                    }
                    return Ok(());
                }
                resolved
            } else {
                self.document
                    .resolve_object(k_raw)
                    .map_err(|error| format!("resolve structure /K: {error}"))?
            };
            self.walk_k(&k, own_page)?;
        }

        if formula_index.is_some() {
            self.active_formulas.pop();
        }
        Ok(())
    }

    fn walk_k(
        &mut self,
        k: &Object,
        parent_direct_page: Option<ObjectRef>,
    ) -> std::result::Result<(), String> {
        if self.out_of_budget() {
            return Ok(());
        }
        match k {
            Object::Integer(mcid) => self.record_integer_mcid(*mcid as u32, parent_direct_page),
            Object::Array(children) => {
                for child in children {
                    if self.out_of_budget() {
                        break;
                    }
                    let resolved = if let Some(reference) = child.as_reference() {
                        if !self.visited.insert(reference.id) {
                            continue;
                        }
                        self.document
                            .load_object(reference)
                            .map_err(|error| format!("load structure child reference: {error}"))?
                    } else {
                        self.document
                            .resolve_object(child)
                            .map_err(|error| format!("resolve structure child: {error}"))?
                    };
                    self.walk_k_child(&resolved, parent_direct_page)?;
                }
            }
            Object::Dictionary(dict) | Object::Stream { dict, .. } => {
                if is_struct_elem_dict(dict) {
                    self.walk_struct_elem(k, parent_direct_page)?;
                } else {
                    self.count_node();
                    self.record_mcr(dict, None);
                }
            }
            Object::Reference(reference) => {
                if !self.visited.insert(reference.id) {
                    return Ok(());
                }
                let resolved = self
                    .document
                    .load_object(*reference)
                    .map_err(|error| format!("load indirect structure child: {error}"))?;
                self.walk_k_child(&resolved, parent_direct_page)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn walk_k_child(
        &mut self,
        child: &Object,
        parent_direct_page: Option<ObjectRef>,
    ) -> std::result::Result<(), String> {
        match child {
            Object::Dictionary(dict) | Object::Stream { dict, .. } if is_struct_elem_dict(dict) => {
                self.walk_struct_elem(child, parent_direct_page)
            }
            Object::Dictionary(dict) | Object::Stream { dict, .. } => {
                self.count_node();
                self.record_mcr(dict, None);
                Ok(())
            }
            Object::Integer(mcid) => {
                self.record_integer_mcid(*mcid as u32, parent_direct_page);
                Ok(())
            }
            Object::Reference(reference) => {
                if !self.visited.insert(reference.id) {
                    return Ok(());
                }
                let resolved = self
                    .document
                    .load_object(*reference)
                    .map_err(|error| format!("load nested structure child: {error}"))?;
                match &resolved {
                    Object::Dictionary(dict) | Object::Stream { dict, .. }
                        if is_struct_elem_dict(dict) =>
                    {
                        self.walk_struct_elem(&resolved, parent_direct_page)
                    }
                    Object::Dictionary(dict) | Object::Stream { dict, .. } => {
                        self.count_node();
                        self.record_mcr(dict, None);
                        Ok(())
                    }
                    Object::Integer(mcid) => {
                        self.record_integer_mcid(*mcid as u32, parent_direct_page);
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    fn record_mcr(
        &mut self,
        dict: &std::collections::HashMap<String, Object>,
        inherited: Option<ObjectRef>,
    ) {
        let Some(mcid) = dict.get("MCID").and_then(Object::as_integer) else {
            return;
        };
        if dict
            .get("Type")
            .and_then(Object::as_name)
            .is_some_and(|name| name != "MCR")
        {
            return;
        }
        let page = dict.get("Pg").and_then(Object::as_reference).or(inherited);
        let scope = self.mcr_scope(dict, page);
        self.record_mcid(mcid as u32, page, scope);
    }

    fn mcr_scope(
        &self,
        dict: &std::collections::HashMap<String, Object>,
        page: Option<ObjectRef>,
    ) -> RawMcidScope {
        let Some(stream_ref) = dict.get("Stm").and_then(Object::as_reference) else {
            return RawMcidScope::Page(page.map(|reference| reference.id));
        };
        let Ok(stream) = self.document.load_object(stream_ref) else {
            return RawMcidScope::Page(page.map(|reference| reference.id));
        };
        let Some(stream_dict) = stream.as_dict() else {
            return RawMcidScope::Page(page.map(|reference| reference.id));
        };
        if stream_dict
            .get("Subtype")
            .and_then(Object::as_name)
            .is_some_and(|subtype| subtype == "Form")
        {
            return RawMcidScope::Form(stream_ref);
        }
        if stream_dict
            .get("PatternType")
            .and_then(Object::as_integer)
            .is_some_and(|pattern_type| pattern_type == 1)
        {
            return RawMcidScope::Pattern(stream_ref);
        }
        RawMcidScope::Page(page.map(|reference| reference.id))
    }

    fn record_mcid(&mut self, mcid: u32, page: Option<ObjectRef>, scope: RawMcidScope) {
        for index in &self.active_formulas {
            if !self.formulas[*index].page_anchor_seen {
                self.formulas[*index].page_object_id = page.map(|reference| reference.id);
                self.formulas[*index].page_anchor_seen = true;
            }
            self.formulas[*index].mcids.push(RawMcid {
                mcid,
                page_object_id: page.map(|reference| reference.id),
                scope: scope.clone(),
            });
        }
    }

    fn record_integer_mcid(&mut self, mcid: u32, page: Option<ObjectRef>) {
        self.record_mcid(
            mcid,
            page,
            RawMcidScope::Page(page.map(|reference| reference.id)),
        );
    }

    fn formula_metadata(
        &self,
        dict: &std::collections::HashMap<String, Object>,
    ) -> FormulaMetadata {
        let mut warnings = Vec::new();
        let description = match dict.get("Alt") {
            Some(object) => match self.document.resolve_object(object) {
                Ok(object) => object
                    .as_string()
                    .map(decode_pdf_text_string)
                    .filter(|text| !text.is_empty()),
                Err(error) => {
                    warnings.push(format!("could not resolve formula /Alt: {error}"));
                    None
                }
            },
            None => None,
        };
        let namespace = raw_json_value(self.document, dict.get("NS"), "/NS", &mut warnings);
        let attributes = raw_json_value(self.document, dict.get("A"), "/A", &mut warnings);
        let associated_files = match dict.get("AF") {
            Some(af) => match self.associated_files(af) {
                Ok(files) if !files.is_empty() => files,
                Ok(_) => {
                    warnings.push("formula /AF has no readable filespec entries".to_string());
                    Vec::new()
                }
                Err(error) => {
                    warnings.push(error);
                    Vec::new()
                }
            },
            None => Vec::new(),
        };
        FormulaMetadata {
            description,
            namespace,
            attributes,
            associated_files,
            warnings,
        }
    }

    fn associated_files(
        &self,
        object: &Object,
    ) -> std::result::Result<Vec<AssociatedFileMetadata>, String> {
        let resolved = self
            .document
            .resolve_object(object)
            .map_err(|error| format!("resolve formula /AF: {error}"))?;
        let specs: Vec<&Object> = match &resolved {
            Object::Array(items) => items.iter().collect(),
            _ => vec![&resolved],
        };
        let mut files = Vec::new();
        for spec in specs {
            let spec = self
                .document
                .resolve_object(spec)
                .map_err(|error| format!("resolve formula /AF filespec: {error}"))?;
            let Some(dict) = spec.as_dict() else {
                continue;
            };
            let filename = dict
                .get("UF")
                .or_else(|| dict.get("F"))
                .and_then(|name| self.document.resolve_object(name).ok())
                .and_then(|name| name.as_string().map(decode_pdf_text_string));
            let relationship = dict
                .get("AFRelationship")
                .and_then(Object::as_name)
                .map(str::to_string);
            let mime_type = dict
                .get("EF")
                .and_then(|ef| self.document.resolve_object(ef).ok())
                .and_then(|ef| ef.as_dict().cloned())
                .and_then(|ef_dict| {
                    ef_dict
                        .get("F")
                        .or_else(|| ef_dict.get("UF"))
                        .and_then(|stream| self.document.resolve_object(stream).ok())
                })
                .and_then(|stream| stream.as_dict().cloned())
                .and_then(|stream_dict| {
                    stream_dict
                        .get("Subtype")
                        .and_then(|subtype| self.document.resolve_object(subtype).ok())
                })
                .and_then(|subtype| subtype.as_name().map(str::to_string));
            files.push(AssociatedFileMetadata {
                filename,
                relationship,
                mime_type,
            });
        }
        Ok(files)
    }

    fn out_of_budget(&mut self) -> bool {
        if self.started.elapsed() > Duration::from_secs(2) {
            self.exceeded = true;
        }
        self.exceeded
    }

    fn count_node(&mut self) -> bool {
        if self.out_of_budget() || self.node_count >= self.max_nodes {
            self.exceeded = true;
            return false;
        }
        self.node_count += 1;
        true
    }
}

fn resolve_all(document: &PdfDocument, object: &Object) -> pdf_oxide::error::Result<Object> {
    let mut current = object.clone();
    for _ in 0..32 {
        let Some(reference) = current.as_reference() else {
            return Ok(current);
        };
        current = document.load_object(reference)?;
    }
    Ok(current)
}

fn is_struct_elem_dict(dict: &std::collections::HashMap<String, Object>) -> bool {
    let has_struct_elem_type = dict
        .get("Type")
        .and_then(Object::as_name)
        .is_none_or(|name| name == "StructElem");
    has_struct_elem_type && dict.get("S").and_then(Object::as_name).is_some()
}

fn raw_json_value(
    document: &PdfDocument,
    object: Option<&Object>,
    key: &str,
    warnings: &mut Vec<String>,
) -> Option<Value> {
    let object = object?;
    match resolve_all(document, object) {
        Ok(object) => match object_to_json(document, &object, 0) {
            Ok(value) => Some(value),
            Err(error) => {
                warnings.push(format!("could not preserve formula {key}: {error}"));
                None
            }
        },
        Err(error) => {
            warnings.push(format!("could not resolve formula {key}: {error}"));
            None
        }
    }
}

fn object_to_json(
    document: &PdfDocument,
    object: &Object,
    depth: usize,
) -> std::result::Result<Value, String> {
    if depth >= 32 {
        return Err("attribute object nesting limit exceeded".to_string());
    }
    match object {
        Object::Null => Ok(Value::Null),
        Object::Boolean(value) => Ok(Value::Bool(*value)),
        Object::Integer(value) => Ok(Value::Number((*value).into())),
        Object::Real(value) => serde_json::Number::from_f64(*value)
            .map(Value::Number)
            .ok_or_else(|| "non-finite PDF attribute number".to_string()),
        Object::Name(value) => Ok(Value::String(value.clone())),
        Object::String(bytes) => Ok(Value::String(decode_pdf_text_string(bytes))),
        Object::Reference(reference) => {
            let resolved = document
                .load_object(*reference)
                .map_err(|error| format!("load PDF attribute reference: {error}"))?;
            object_to_json(document, &resolved, depth + 1)
        }
        Object::Array(items) => items
            .iter()
            .map(|item| object_to_json(document, item, depth + 1))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Value::Array),
        Object::Dictionary(dict) => {
            let mut result = serde_json::Map::new();
            for (key, value) in dict {
                result.insert(key.clone(), object_to_json(document, value, depth + 1)?);
            }
            Ok(Value::Object(result))
        }
        Object::Stream { dict, .. } => {
            let mut result = serde_json::Map::new();
            result.insert("$stream".to_string(), Value::Bool(true));
            for (key, value) in dict {
                result.insert(key.clone(), object_to_json(document, value, depth + 1)?);
            }
            Ok(Value::Object(result))
        }
    }
}

fn decode_pdf_text_string(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xFE, 0xFF]) {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).unwrap_or_else(|_| String::from_utf8_lossy(bytes).into_owned())
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).unwrap_or_else(|_| String::from_utf8_lossy(bytes).into_owned())
    } else {
        bytes
            .iter()
            .filter_map(|byte| pdf_oxide::fonts::font_dict::pdfdoc_encoding_lookup(*byte))
            .collect()
    }
}

/// `pdf_oxide 0.3.78::extract_embedded_files()` converts PDF string bytes
/// directly with `String::from_utf8_lossy`. Recover UTF-16BE filenames when
/// that loses the leading BOM bytes but leaves the UTF-16 code units intact.
/// Ordinary Unicode filenames pass through unchanged.
pub(crate) fn decode_embedded_filename_from_pdf_oxide(value: &str) -> String {
    let mut chars = value.chars();
    if chars.next() != Some('\u{FFFD}') || chars.next() != Some('\u{FFFD}') {
        return value.to_string();
    }
    let Some(rest): Option<Vec<u8>> = chars.map(|ch| u8::try_from(ch as u32).ok()).collect() else {
        return value.to_string();
    };
    if rest.is_empty() || rest.len() % 2 != 0 {
        return value.to_string();
    }
    let mut bytes = Vec::with_capacity(rest.len() + 2);
    bytes.extend_from_slice(&[0xFE, 0xFF]);
    bytes.extend(rest);
    decode_pdf_text_string(&bytes)
}

/// Map the lossy filenames returned by `pdf_oxide::extract_embedded_files()`
/// back to PDFDoc/UTF-16 filenames from the same EmbeddedFiles name tree.
/// Duplicate lossy labels with different originals are omitted so matching
/// fails closed instead of attaching bytes to the wrong formula.
pub(crate) fn embedded_filename_aliases(document: &PdfDocument) -> HashMap<String, String> {
    let Ok(catalog) = document.catalog() else {
        return HashMap::new();
    };
    let Some(names) = catalog
        .as_dict()
        .and_then(|dict| dict.get("Names"))
        .and_then(|names| document.resolve_object(names).ok())
    else {
        return HashMap::new();
    };
    let Some(embedded_files) = names
        .as_dict()
        .and_then(|dict| dict.get("EmbeddedFiles"))
        .and_then(|root| document.resolve_object(root).ok())
    else {
        return HashMap::new();
    };

    let mut aliases = HashMap::new();
    let mut nodes = 0;
    let started = Instant::now();
    let mut exceeded = false;
    collect_embedded_filename_aliases(
        document,
        &embedded_files,
        0,
        &mut aliases,
        &mut nodes,
        started,
        &mut exceeded,
    );
    if exceeded {
        return HashMap::new();
    }
    aliases
        .into_iter()
        .filter_map(|(lossy, decoded)| decoded.map(|decoded| (lossy, decoded)))
        .collect()
}

fn collect_embedded_filename_aliases(
    document: &PdfDocument,
    node: &Object,
    depth: u8,
    aliases: &mut HashMap<String, Option<String>>,
    nodes: &mut usize,
    started: Instant,
    exceeded: &mut bool,
) {
    // Match pdf_oxide 0.3.78's name-tree depth bound. The object traversal is
    // only for filenames; stream decoding remains in extract_embedded_files.
    if depth > 32
        || *nodes >= MAX_EMBEDDED_NAME_TREE_NODES
        || started.elapsed() > MAX_EMBEDDED_NAME_TREE_TIME
    {
        *exceeded = true;
        return;
    }
    *nodes += 1;
    let Ok(node) = document.resolve_object(node) else {
        return;
    };
    let Some(dict) = node.as_dict() else {
        return;
    };
    if let Some(names) = dict
        .get("Names")
        .and_then(|names| document.resolve_object(names).ok())
        .and_then(|names| names.as_array().cloned())
    {
        let mut index = 1;
        while index < names.len() {
            if let Ok(filespec) = document.resolve_object(&names[index]) {
                if let Some(filespec) = filespec.as_dict() {
                    let filename = filespec
                        .get("UF")
                        .or_else(|| filespec.get("F"))
                        .and_then(|name| document.resolve_object(name).ok())
                        .and_then(|name| name.as_string().map(|bytes| bytes.to_vec()));
                    if let Some(filename) = filename {
                        let lossy = String::from_utf8_lossy(&filename).into_owned();
                        let decoded = decode_pdf_text_string(&filename);
                        aliases
                            .entry(lossy)
                            .and_modify(|seen| {
                                if seen.as_deref() != Some(decoded.as_str()) {
                                    *seen = None;
                                }
                            })
                            .or_insert(Some(decoded));
                    }
                }
            }
            index += 2;
        }
    }
    if let Some(kids) = dict
        .get("Kids")
        .and_then(|kids| document.resolve_object(kids).ok())
        .and_then(|kids| kids.as_array().cloned())
    {
        for kid in kids {
            collect_embedded_filename_aliases(
                document,
                &kid,
                depth + 1,
                aliases,
                nodes,
                started,
                exceeded,
            );
            if *exceeded {
                return;
            }
        }
    }
}
