//! Document-aware routing and one-vote visual merge for DEV.
use super::*;
use magpie_core::rusqlite::OptionalExtension;
use magpie_core::search::decision::{self, Route};
fn merge_once(
    primary: Vec<serde_json::Value>,
    secondary: Vec<serde_json::Value>,
    limit: usize,
) -> Vec<serde_json::Value> {
    use std::collections::HashMap;
    let mut pool = HashMap::<String, (serde_json::Value, usize, usize)>::new();
    for (source, list) in [primary, secondary].into_iter().enumerate() {
        for (rank, hit) in list.into_iter().enumerate() {
            let Some(path) = hit["path"].as_str() else {
                continue;
            };
            let e = pool
                .entry(path.to_owned())
                .or_insert((hit.clone(), rank, source));
            if (rank, source) < (e.1, e.2) {
                *e = (hit.clone(), rank, source);
            }
            if hit["end_ms"].as_i64().unwrap_or(0) > hit["start_ms"].as_i64().unwrap_or(0) {
                e.0 = hit;
            }
        }
    }
    let mut out: Vec<_> = pool.into_values().collect();
    out.sort_by(|a, b| {
        (a.1, a.2)
            .cmp(&(b.1, b.2))
            .then_with(|| a.0["path"].as_str().cmp(&b.0["path"].as_str()))
    });
    out.into_iter().take(limit).map(|(v, _, _)| v).collect()
}
fn annotate_copies(
    conn: &magpie_core::rusqlite::Connection,
    hits: &mut [serde_json::Value],
) -> Result<()> {
    use std::collections::HashMap;
    let mut groups = HashMap::<String, Vec<usize>>::new();
    for (i, h) in hits.iter().enumerate() {
        let Some(path) = h["path"].as_str() else {
            continue;
        };
        // Refuse equivalence for oversized text rather than hash a prefix.
        let text:Option<String>=conn.query_row("SELECT CASE WHEN length(content)<=2000000 THEN content ELSE NULL END FROM files WHERE path=?1",[path],|r|r.get(0)).optional()?.flatten();
        if let Some(t) = text.and_then(|t| decision::normalized_full_text(&t)) {
            groups.entry(t).or_default().push(i);
        }
    }
    for indices in groups.values().filter(|v| v.len() > 1) {
        let paths: Vec<_> = indices
            .iter()
            .filter_map(|&i| hits[i]["path"].as_str().map(str::to_owned))
            .collect();
        for &i in indices {
            hits[i]["equivalent_paths"] = json!(paths);
            hits[i]["equivalence_basis"] = json!("complete_normalized_indexed_text");
        }
    }
    Ok(())
}
pub(super) fn rank(
    conn: &magpie_core::rusqlite::Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    iq: Option<&[f32]>,
    pivot: Option<(&str, &[f32])>,
    scope: search::LocalScope,
    limit: usize,
) -> Result<Vec<serde_json::Value>> {
    // Preserve explicit visual scopes, their filters, video timestamps and anchors.
    if matches!(
        scope,
        search::LocalScope::Images | search::LocalScope::Videos
    ) {
        return rank_local_bilingual_legacy(conn, store, query, qvec, iq, pivot, scope, limit);
    }
    let route = decision::route(conn, query, scope, visual_query_intent(query))?;
    let visual_vector = if route == Route::Document { None } else { iq };
    let files = search::search_files(conn, store, query, qvec, visual_vector, scope, limit)?;
    let primary = tag_hits(files, vec![], false);
    let mut out = if route == Route::Document {
        primary
    } else {
        let native = if let Some(v) = iq {
            visual_expansion_hits(conn, store, query, v, scope, limit)?
        } else {
            vec![]
        };
        let translated = if let Some((_, v)) = pivot {
            visual_expansion_hits(conn, store, query, v, scope, limit)?
        } else {
            vec![]
        };
        let visual = merge_once(native, translated, limit);
        if route == Route::Visual {
            merge_once(visual, primary, limit)
        } else {
            merge_once(primary, visual, limit)
        }
    };
    annotate_copies(conn, &mut out)?;
    Ok(out)
}
#[cfg(test)]
mod decision_tests {
    use super::*;
    use magpie_core::rusqlite::OptionalExtension;
    #[test]
    fn decision_visual_duplicate_has_one_vote_and_preserves_shot() {
        let a = vec![json!({"path":"/doc.docx"}), json!({"path":"/photo.jpg"})];
        let b = vec![json!({"path":"/photo.jpg"}), json!({"path":"/photo.jpg"})];
        let out = merge_once(a, b, 10);
        assert_eq!(out[0]["path"], "/doc.docx");
        assert_eq!(out.len(), 2);
        let out = merge_once(
            vec![json!({"path":"/clip.mov"})],
            vec![json!({"path":"/clip.mov","start_ms":1,"end_ms":10})],
            10,
        );
        assert_eq!(out[0]["end_ms"], 10);
    }
    #[test]
    fn decision_mixed_keeps_document_and_visual() {
        let out = merge_once(
            vec![json!({"path":"/report.pdf"})],
            vec![json!({"path":"/scene.jpg"})],
            10,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["path"], "/report.pdf");
    }
    #[test]
    fn decision_document_route_and_complete_copy_groups() {
        let c = magpie_core::db::open_in_memory().unwrap();
        c.execute("INSERT INTO folders(path) VALUES('/fixture')", [])
            .unwrap();
        for (id, name, ext, text) in [
            (1, "зеленый стул.docx", "docx", "полный текст"),
            (2, "other.png", "png", "полный текст"),
            (3, "different.docx", "docx", "полный текст другой"),
        ] {
            c.execute("INSERT INTO files(id,folder_id,path,name,ext,size,mtime,content) VALUES(?1,1,?2,?3,?4,100,1,?5)",magpie_core::rusqlite::params![id,format!("/fixture/{name}"),name,ext,text]).unwrap();
        }
        let mut store = VectorStore::empty();
        store.images = vec![(2, vec![1., 0.])];
        let hits = rank(
            &c,
            &store,
            "зеленый стул",
            None,
            Some(&[1., 0.]),
            Some(("a green chair", &[1., 0.])),
            search::LocalScope::All,
            30,
        )
        .unwrap();
        assert_eq!(hits[0]["id"], 1);
        let hits = rank(
            &c,
            &store,
            "фото зеленый стул",
            None,
            Some(&[1., 0.]),
            None,
            search::LocalScope::All,
            30,
        )
        .unwrap();
        assert!(hits.iter().any(|h| h["id"] == 2));
        let mut copies = vec![
            json!({"path":"/fixture/зеленый стул.docx"}),
            json!({"path":"/fixture/other.png"}),
            json!({"path":"/fixture/different.docx"}),
        ];
        annotate_copies(&c, &mut copies).unwrap();
        assert_eq!(copies[0]["equivalent_paths"].as_array().unwrap().len(), 2);
        assert!(copies[2].get("equivalent_paths").is_none());
        assert_eq!(copies.len(), 3);
    }
}
