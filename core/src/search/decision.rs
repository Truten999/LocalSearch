//! DEV decision 2026-10-15. Independent bounded retrieval, evidence tiers.
//! No schema writes, learned weights, or dependence on evaluation labels.
use super::*;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

pub const PREFETCH: usize = 500;
pub const SHORTLIST: usize = 150;
pub const DEFAULT_ENABLED: bool = true;
thread_local! {
    static MODE: Cell<Option<bool>> = const { Cell::new(None) };
    static AUDIT: RefCell<CandidateAudit> = RefCell::new(CandidateAudit::default());
}
#[derive(Default, Clone, Serialize)]
pub struct CandidateAudit {
    pub exact: Vec<i64>,
    pub lexical: Vec<i64>,
    pub fallback: Vec<i64>,
    pub dense: Vec<i64>,
    pub visual: Vec<i64>,
    pub shortlist: Vec<i64>,
}
pub fn enabled() -> bool {
    MODE.with(|m| m.get().unwrap_or(DEFAULT_ENABLED))
}
/// Thread-scoped override: old/new share a binary without global environment races.
pub fn with_mode<T>(value: bool, f: impl FnOnce() -> T) -> T {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            MODE.with(|m| m.set(self.0));
        }
    }
    let _reset = Reset(MODE.with(|m| m.replace(Some(value))));
    f()
}
pub fn last_audit() -> CandidateAudit {
    AUDIT.with(|a| a.borrow().clone())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Route {
    Document,
    Visual,
    Mixed,
}
fn has_visual_word(text: &str) -> bool {
    step29_words(text).iter().any(|w| {
        matches!(
            w.as_str(),
            "фото"
                | "фотография"
                | "фотографию"
                | "изображение"
                | "картинка"
                | "видео"
                | "photo"
                | "image"
                | "picture"
                | "video"
        )
    })
}
fn exact_sensitive(text: &str) -> bool {
    text.chars()
        .any(|c| c.is_numeric() || c == '"' || c == '«' || c == '»')
}
/// Exact whole tokens, all terms, no stemming, no four-word truncation.
fn literal_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}
fn expression(text: &str, field: Option<&str>) -> Option<String> {
    let normalized_quotes = text.replace(['«', '»'], "\"");
    if normalized_quotes.contains('"') {
        if normalized_quotes.matches('"').count() % 2 != 0 {
            return None;
        }
        let mut clauses = Vec::new();
        for (i, part) in normalized_quotes.split('"').enumerate() {
            let words = literal_words(part);
            if i % 2 == 1 && !words.is_empty() {
                clauses.push(format!("\"{}\"", words.join(" ")));
            } else {
                clauses.extend(words.iter().map(|w| format!("\"{w}\"")));
            }
        }
        if clauses.is_empty() || clauses.len() > 32 {
            return None;
        }
        let expr = clauses.join(" AND ");
        return Some(if let Some(f) = field {
            format!("{f} : ({expr})")
        } else {
            expr
        });
    }
    let terms = literal_words(text);
    if terms.is_empty() || terms.len() > 32 {
        return None;
    }
    let mut seen = HashSet::new();
    let expr = terms
        .iter()
        .filter(|s| seen.insert(s.as_str()))
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(" AND ");
    Some(if let Some(f) = field {
        format!("{f} : ({expr})")
    } else {
        expr
    })
}
pub fn fts_ids(conn: &Connection, expr: &str, limit: usize) -> Result<Vec<i64>> {
    let mut s=conn.prepare_cached("SELECT rowid FROM files_fts WHERE files_fts MATCH ?1 AND rank MATCH 'bm25(8.0,2.0,1.0)' ORDER BY rank LIMIT ?2")?;
    let rows = s.query_map(rusqlite::params![expr, limit as i64], |r| r.get(0))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}
pub fn full_name_ids(conn: &Connection, text: &str, limit: usize) -> Result<Vec<i64>> {
    expression(text, Some("name"))
        .map(|e| fts_ids(conn, &e, limit))
        .unwrap_or_else(|| Ok(vec![]))
}
/// Restricted common-noun inflections only. Unknown/proper names, quotes and
/// identifiers are never broadened. No claim of a general Russian stemmer.
pub fn inflection_expression(text: &str) -> Option<String> {
    if exact_sensitive(text) || text.chars().any(char::is_uppercase) {
        return None;
    }
    const NOUNS: &[(&str, &str)] = &[
        ("договора", "договор"),
        ("договору", "договор"),
        ("договором", "договор"),
        ("договоры", "договор"),
        ("договоров", "договор"),
        ("документа", "документ"),
        ("документы", "документ"),
        ("документов", "документ"),
        ("отчета", "отчет"),
        ("отчетов", "отчет"),
        ("остатками", "остатк"),
        ("остатков", "остатк"),
        ("таблицы", "таблиц"),
        ("таблицу", "таблиц"),
        ("инструкции", "инструкц"),
    ];
    let terms = step29_words(text);
    let mut changed = false;
    let parts = terms
        .iter()
        .filter(|w| !matches!(w.as_str(), "с" | "со" | "о" | "об" | "по" | "для"))
        .map(|w| {
            if let Some((_, stem)) = NOUNS.iter().find(|(form, _)| *form == w) {
                changed = true;
                format!("\"{stem}\"*")
            } else {
                format!("\"{w}\"")
            }
        })
        .collect::<Vec<_>>();
    (changed && !parts.is_empty()).then(|| parts.join(" AND "))
}
pub fn route(conn: &Connection, query: &str, scope: LocalScope, scene_hint: bool) -> Result<Route> {
    if matches!(scope, LocalScope::Images | LocalScope::Videos) {
        return Ok(Route::Visual);
    }
    if scope == LocalScope::Text {
        return Ok(Route::Document);
    }
    let (filters, text) = crate::filters::parse(query);
    let visual = has_visual_word(&text);
    let office_filter = filters
        .exts
        .iter()
        .any(|e| document_priority(Some(e)) > 0.0);
    if office_filter {
        return Ok(Route::Document);
    }
    if !filters.exts.is_empty()
        && filters
            .exts
            .iter()
            .all(|e| crate::files::is_image_ext(Some(e)) || crate::videos::is_video_ext(Some(e)))
    {
        return Ok(Route::Visual);
    }
    let doc_words = step29_words(&text)
        .into_iter()
        .filter(|w| !has_visual_word(w))
        .collect::<Vec<_>>()
        .join(" ");
    let document =
        document_intent(&doc_words) || spoken_format(&text).is_some() || exact_sensitive(&text);
    if document {
        return Ok(if visual {
            Route::Mixed
        } else {
            Route::Document
        });
    }
    if visual {
        return Ok(Route::Visual);
    }
    if scene_hint {
        let ids = full_name_ids(conn, &normalize_local_query_text(&text), 50)?;
        let hits = ranking_file_hits(conn, &ids, &HashMap::new())?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;
        if hits.iter().any(|h| {
            document_priority(h.ext.as_deref()) > 0.0
                && filters.matches(h.ext.as_deref(), h.size, h.mtime, &h.path, now)
        }) {
            return Ok(Route::Document);
        }
        return Ok(Route::Mixed);
    }
    Ok(Route::Document)
}

pub fn search(
    conn: &Connection,
    store: &VectorStore,
    input: &str,
    qvec: Option<&[f32]>,
    iq: Option<&[f32]>,
    scope: LocalScope,
    limit: usize,
) -> Result<Vec<crate::files::FileHit>> {
    let (mut filters, text) = crate::filters::parse(input);
    let normalized = normalize_local_query_text(&text);
    let spoken = if filters.exts.is_empty() {
        spoken_format(&normalized)
    } else {
        None
    };
    let query = spoken
        .as_ref()
        .map(|s| s.text.as_str())
        .unwrap_or(&normalized);
    if let Some(s) = &spoken {
        filters.exts.extend(s.exts.iter().map(|s| s.to_string()));
    }
    if query.trim().is_empty() {
        return super::search_files_legacy(conn, store, input, qvec, iq, scope, limit);
    }
    if expression(query, None).is_none() {
        AUDIT.with(|a| *a.borrow_mut() = CandidateAudit::default());
        return Ok(vec![]);
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs() as i64;
    let address = AddressIntent::parse(query);
    let joint = if let Some(a) = &address {
        fts_ids(conn, &a.fts_expression(), 100)?
    } else {
        vec![]
    };
    let mut lexical = fts_ids(conn, &expression(query, None).unwrap(), PREFETCH)?;
    let strict_ids: HashSet<_> = lexical.iter().copied().collect();
    if address.is_some() {
        // Keep the existing validated topic+address evidence and partial
        // fallbacks. Do not stem arbitrary identifiers or broaden house numbers.
        let partial = step29_words(query)
            .iter()
            .map(|w| format!("\"{w}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        for id in joint.iter().copied().chain(fts_ids(conn, &partial, 100)?) {
            if lexical.len() >= PREFETCH {
                break;
            }
            if !lexical.contains(&id) {
                lexical.push(id);
            }
        }
    }
    let exact = full_name_ids(conn, query, 100)?;
    let fallback = if lexical.is_empty() {
        if let Some(expr) = inflection_expression(query) {
            fts_ids(conn, &expr, PREFETCH)?
        } else {
            vec![]
        }
    } else {
        vec![]
    };
    let dense = qvec
        .map(|v| top_similar_grouped(&store.file_chunks, v, 100))
        .unwrap_or_default();
    let visual = iq
        .map(|v| top_similar(&store.images, v, 100))
        .unwrap_or_default();
    let mut audit = CandidateAudit {
        exact: exact.clone(),
        lexical: lexical.clone(),
        fallback: fallback.clone(),
        dense: dense.iter().map(|x| x.0).collect(),
        visual: visual.iter().map(|x| x.0).collect(),
        shortlist: vec![],
    };
    let lexical_rank: HashMap<_, _> = lexical.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let exact_rank: HashMap<_, _> = exact.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let fallback_rank: HashMap<_, _> = fallback
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();
    let lists = vec![
        lexical.clone(),
        fallback.clone(),
        audit.dense.clone(),
        audit.visual.clone(),
        exact.clone(),
    ];
    let scores: HashMap<_, _> = rrf_fuse(&lists).into_iter().collect();
    let ids: Vec<_> = scores.keys().copied().collect();
    let mut hits = ranking_file_hits(conn, &ids, &scores)?;
    hits.retain(|h| {
        filters.matches(h.ext.as_deref(), h.size, h.mtime, &h.path, now)
            && match scope {
                LocalScope::All => true,
                LocalScope::Text => !crate::files::is_image_ext(h.ext.as_deref()),
                LocalScope::Images => crate::files::is_image_ext(h.ext.as_deref()),
                LocalScope::Videos => crate::videos::is_video_ext(h.ext.as_deref()),
            }
    });
    // Explicit evidence tiers, not fitted coefficients: whole filename, all
    // requested terms, conservative inflection rescue, then semantic evidence.
    let key = |id: &i64| {
        if let Some(r) = exact_rank.get(id) {
            (0, *r)
        } else if let Some(r) = lexical_rank.get(id) {
            (1, *r)
        } else if let Some(r) = fallback_rank.get(id) {
            (2, *r)
        } else {
            (3, 0)
        }
    };
    hits.sort_by(|a, b| {
        joint
            .contains(&b.id)
            .cmp(&joint.contains(&a.id))
            .then_with(|| key(&a.id).cmp(&key(&b.id)))
            .then_with(|| b.score.total_cmp(&a.score))
            .then(a.id.cmp(&b.id))
    });
    // Reserve independent dense evidence while bounding expensive text work.
    let ordered_ids: Vec<_> = hits.iter().map(|h| h.id).collect();
    let mut chosen: Vec<_> = ordered_ids.iter().take(100).copied().collect();
    for id in audit
        .dense
        .iter()
        .chain(audit.visual.iter())
        .chain(ordered_ids.iter())
    {
        if chosen.len() >= SHORTLIST {
            break;
        }
        if !chosen.contains(id) && hits.iter().any(|h| h.id == *id) {
            chosen.push(*id);
        }
    }
    let chosen_set: HashSet<_> = chosen.iter().copied().collect();
    hits.retain(|h| chosen_set.contains(&h.id));
    audit.shortlist = hits.iter().map(|h| h.id).collect();
    AUDIT.with(|a| *a.borrow_mut() = audit);
    let samples = ranking_text_samples(conn, &chosen)?;
    let prepared = crate::document_rank::PreparedTopicQuery::new(query);
    for h in &mut hits {
        if let Some(sample) = samples.get(&h.id) {
            let (profile, _) = crate::document_rank::cached_topic_profile(h.id, &h.name, sample);
            h.score += prepared.boost(&profile);
            h.snippet = structured_passage(query, h.ext.as_deref(), sample);
        }
        h.match_kind = Some(
            if address.as_ref().is_some_and(|a| {
                samples
                    .get(&h.id)
                    .is_some_and(|sample| a.has_joint_document_evidence(&h.name, sample))
            }) {
                "Тема + адрес"
            } else if exact_rank.contains_key(&h.id) {
                "Полное имя файла"
            } else if strict_ids.contains(&h.id) {
                "Все слова запроса"
            } else if lexical_rank.contains_key(&h.id) {
                "Частичное совпадение"
            } else if fallback_rank.contains_key(&h.id) {
                "Словоформа"
            } else {
                "По смыслу"
            }
            .into(),
        );
    }
    // Expensive evidence breaks ties only; it cannot demote exact lexical
    // matches behind vector-only results with a different similarity scale.
    hits.sort_by(|a, b| {
        joint
            .contains(&b.id)
            .cmp(&joint.contains(&a.id))
            .then_with(|| key(&a.id).cmp(&key(&b.id)))
            .then_with(|| b.score.total_cmp(&a.score))
            .then(a.id.cmp(&b.id))
    });
    if address.is_some() {
        sort_joint_address_first(&mut hits);
    }
    hits.truncate(limit);
    attach_final_thumbnails(conn, &mut hits)?;
    Ok(hits)
}
/// Complete indexed text, never a 64K prefix; blank text is not equivalent.
pub fn normalized_full_text(text: &str) -> Option<String> {
    let s = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    (!s.is_empty()).then_some(s)
}
#[cfg(test)]
mod decision_tests {
    use super::*;
    fn db() -> Connection {
        let c = crate::db::open_in_memory().unwrap();
        c.execute("INSERT INTO folders(path) VALUES('/fixture')", [])
            .unwrap();
        c
    }
    fn add(c: &Connection, id: i64, name: &str, text: &str) {
        c.execute("INSERT INTO files(id,folder_id,path,name,ext,size,mtime,content) VALUES(?1,1,?2,?3,'docx',100,1,?4)",rusqlite::params![id,format!("/fixture/{name}"),name,text]).unwrap();
    }
    #[test]
    fn decision_inflection_is_conservative() {
        assert!(inflection_expression("условия договора").is_some());
        for q in [
            "договор №123",
            "договору Елкина",
            "Иванова",
            "АБ-302",
            "\"договора\"",
            "2026-10-15",
        ] {
            assert!(inflection_expression(q).is_none(), "{q}");
        }
    }
    #[test]
    fn decision_full_text_not_prefix_or_blank() {
        let a = "x".repeat(65536);
        assert_ne!(
            normalized_full_text(&(a.clone() + " one")),
            normalized_full_text(&(a + " two"))
        );
        assert_eq!(
            normalized_full_text("  АБВ\nтекст "),
            normalized_full_text("абв текст")
        );
        assert_eq!(normalized_full_text(" \n"), None);
    }
    #[test]
    fn decision_exact_and_inflected_without_reindex() {
        let c = db();
        add(&c, 1, "контракт.docx", "условия договор аренда");
        add(&c, 2, "другое.docx", "условия договора ремонт");
        let store = VectorStore::empty();
        let h = search(
            &c,
            &store,
            "условия договора",
            None,
            None,
            LocalScope::All,
            10,
        )
        .unwrap();
        assert_eq!(h[0].id, 2);
        assert!(last_audit().fallback.is_empty());
        let h = search(
            &c,
            &store,
            "аренда договору",
            None,
            None,
            LocalScope::All,
            10,
        )
        .unwrap();
        assert_eq!(h[0].id, 1);
        assert!(search(
            &c,
            &store,
            "аренда договору ext:pdf",
            None,
            None,
            LocalScope::All,
            10
        )
        .unwrap()
        .is_empty());
    }
    #[test]
    fn decision_exact_title_survives_generic_pool_and_shortlist_is_bounded() {
        let c = db();
        for n in 1..=520 {
            add(&c, n, &format!("unrelated-{n}.docx"), "зеленый стул");
        }
        add(&c, 600, "зеленый стул.docx", "описание товара");
        let h = search(
            &c,
            &VectorStore::empty(),
            "зеленый стул",
            None,
            None,
            LocalScope::All,
            10,
        )
        .unwrap();
        assert_eq!(h[0].id, 600);
        assert!(last_audit().shortlist.len() <= 150);
        assert_eq!(last_audit().lexical.len(), 500);
        assert_eq!(
            route(&c, "зеленый стул", LocalScope::All, true).unwrap(),
            Route::Document
        );
        assert_eq!(
            route(&c, "фото зеленого стула", LocalScope::All, true).unwrap(),
            Route::Visual
        );
        assert_eq!(
            route(&c, "фото договора", LocalScope::All, true).unwrap(),
            Route::Mixed
        );
        assert_eq!(
            route(&c, "фото стула ext:docx", LocalScope::All, true).unwrap(),
            Route::Document
        );
        assert_eq!(
            route(&c, "договор 123", LocalScope::Images, false).unwrap(),
            Route::Visual
        );
    }
    #[test]
    fn decision_quotes_identifiers_and_ocr_are_exact() {
        let c = db();
        add(&c, 1, "акт.docx", "условия договора АБ-123");
        add(&c, 2, "лишнее.docx", "условия другого договора АБ-1234");
        let store = VectorStore::empty();
        let h = search(
            &c,
            &store,
            "\"условия договора\"",
            None,
            None,
            LocalScope::All,
            10,
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].id, 1);
        let h = search(&c, &store, "АБ-123", None, None, LocalScope::All, 10).unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].id, 1);
        c.execute("UPDATE files SET ext='png' WHERE id=1", [])
            .unwrap();
        assert_eq!(
            search(&c, &store, "АБ-123", None, None, LocalScope::All, 10).unwrap()[0].id,
            1
        );
    }
    #[test]
    fn decision_proper_name_keeps_yo_literal() {
        let c = db();
        add(&c, 1, "первый.docx", "Ёлкин");
        add(&c, 2, "второй.docx", "Елкин");
        let h = search(
            &c,
            &VectorStore::empty(),
            "Ёлкин",
            None,
            None,
            LocalScope::All,
            10,
        )
        .unwrap();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].id, 1);
    }
    #[test]
    fn decision_mode_restores_after_scope() {
        let old = enabled();
        with_mode(!old, || assert_eq!(enabled(), !old));
        assert_eq!(enabled(), old);
    }
}
