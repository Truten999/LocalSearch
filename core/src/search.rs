//! Hybrid retrieval: FTS5 BM25 + vector cosine, fused with Reciprocal Rank Fusion.

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

use crate::db::{self, Repo};
use crate::embed::dot;

const RRF_K: f32 = 60.0;
const CANDIDATES_PER_LIST: usize = 50;
/// Cosine similarity contributes a small additive term on top of RRF: rank
/// structure dominates, similarity magnitude breaks ties and nudges close calls.
const VEC_SIM_WEIGHT: f32 = 0.005;


/// Strip spoken *command* prefixes for local-file retrieval only. Never change
/// the visible search field. A query that is itself a filename (e.g.
/// "Найди.docx") must not lose its first word.
///
/// A trailing compound spoken number is converted to digits when it is likely
/// an identifier/address. This helps a voice query match indexed "28" without
/// rewriting a whole document or changing any persistent index.
pub fn normalize_local_query_text(input: &str) -> String {
    let mut words: Vec<String> = input.split_whitespace().map(str::to_string).collect();
    if words.is_empty() { return String::new(); }
    let first = words[0].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    if ["найди", "найти", "найдите", "покажи", "покажите", "отыщи", "поищи"]
        .contains(&first.as_str()) && words.len() > 1
    {
        words.remove(0);
        while words.len() > 1 {
            let lead = words[0].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if ["мне", "пожалуйста", "файл", "файлы", "документ", "документы"].contains(&lead.as_str()) {
                words.remove(0);
            } else { break; }
        }
    }
    if words.len() >= 3 {
        let n = words.len();
        let tens = words[n-2].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let ones = words[n-1].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        let base: Option<usize> = match tens.as_str() {
            "двадцать" => Some(20), "тридцать" => Some(30), "сорок" => Some(40),
            "пятьдесят" => Some(50), "шестьдесят" => Some(60),
            "семьдесят" => Some(70), "восемьдесят" => Some(80),
            "девяносто" => Some(90), _ => None,
        };
        let unit: Option<usize> = match ones.as_str() {
            "один" => Some(1), "два" => Some(2), "три" => Some(3),
            "четыре" => Some(4), "пять" => Some(5), "шесть" => Some(6),
            "семь" => Some(7), "восемь" => Some(8), "девять" => Some(9),
            _ => None,
        };
        if let (Some(t), Some(u)) = (base, unit) {
            words.truncate(n - 2);
            words.push((t + u).to_string());
        }
    }
    words.join(" ")
}


/// A natural-language file type mention is a constraint, not a relevance keyword.
/// Keep explicitly typed `ext:` filters authoritative; never mistake a suffix
/// inside a file name ("report.xlsx") for a spoken type instruction.
#[derive(Debug, Clone)]
struct SpokenFormat {
    exts: &'static [&'static str],
    text: String,
}

fn spoken_format(query: &str) -> Option<SpokenFormat> {
    let mut exts: Option<&'static [&'static str]> = None;
    let mut text = Vec::new();
    for token in query.split_whitespace() {
        let key = token.trim_matches(|c: char| matches!(c, ',' | ':' | ';' | '!' | '?' | '(' | ')' | '«' | '»'))
            .to_lowercase().replace('ё', "е");
        let candidate: Option<&'static [&'static str]> = match key.as_str() {
            "excel" | "эксель" | "xlsx" | "xls" => Some(&["xlsx", "xls"]),
            "word" | "ворд" | "docx" | "doc" => Some(&["docx", "doc"]),
            "pdf" | "пдф" => Some(&["pdf"]),
            "csv" => Some(&["csv"]),
            _ => None,
        };
        if let Some(found) = candidate {
            // Multiple format mentions are ambiguous ("Word или PDF").
            // Avoid silently enforcing only one of them.
            if exts.is_some() && exts != Some(found) { return None; }
            exts = Some(found);
        } else {
            text.push(token);
        }
    }
    exts.map(|exts| SpokenFormat { exts, text: text.join(" ") })
}

/// Extract meaningful inflection-tolerant text tokens. Numeric identifiers
/// remain exact. This is used only as a *second candidate source* when the
/// user has specifically requested a file format, not as a rewrite of every
/// FTS query.
fn format_topic_expression(text: &str) -> Option<String> {
    const SKIP: &[&str] = &[
        "в", "на", "с", "со", "по", "за", "для", "из", "у", "и", "или", "о", "об", "от",
        "найди", "найти", "мне", "покажи", "файл", "файлы", "файла", "документ",
        "документы", "документа", "где", "котором", "который", "нужен", "нужна",
        "таблица", "таблицу", "таблицы", "список", "перечень",
        "данные", "данными", "сведения", "сведениями",
    ];
    let terms: Vec<String> = step29_words(text).into_iter()
        .filter(|s| !SKIP.contains(&s.as_str()))
        .filter(|s| s.chars().count() >= 4 || s.chars().all(|c| c.is_ascii_digit()))
        .take(3)
        .map(|s| {
            if s.chars().all(|c| c.is_ascii_digit()) { format!("\"{s}\"") }
            else { format!("\"{}\"*", step29_stem(&s)) }
        })
        .collect();
    if terms.is_empty() { None } else { Some(terms.join(" AND ")) }
}

/// Retrieve matches *within* the requested format before generic search can
/// use up its top 50 on unrelated file types. Only uses existing FTS5 and
/// `files.ext`, without a new table or index migration.
fn format_fts_candidates(
    conn: &Connection, query: &str, extensions: &[&str], limit: usize,
) -> Result<Vec<(i64, String)>> {
    let Some(expression) = format_topic_expression(query) else { return Ok(Vec::new()); };
    let mut stmt = conn.prepare(
        "SELECT files_fts.rowid, snippet(files_fts, 2, char(1), char(2), '…', 12)
         FROM files_fts JOIN files f ON f.id = files_fts.rowid
         WHERE files_fts MATCH ?1 AND lower(f.ext) = ?2
         ORDER BY bm25(files_fts, 8.0, 2.0, 1.0) LIMIT ?3"
    )?;
    let mut out: Vec<(i64, String)> = Vec::new();
    for ext in extensions {
        if out.len() >= limit { break; } // Other extensions cannot add results.
        let items = stmt.query_map(rusqlite::params![expression, ext, limit as i64], |r| {
            let snippet: Option<String> = r.get(1)?;
            Ok((r.get::<_, i64>(0)?, snippet.unwrap_or_default()))
        })?.collect::<rusqlite::Result<Vec<_>>>()?;
        for item in items {
            if out.len() >= limit { break; }
            if !out.iter().any(|x| x.0 == item.0) { out.push(item); }
        }
    }
    Ok(out)
}

/// Score only explicit *document* intent. "Скан", "фото", "код", etc.
/// must not be demoted as if they were unintended results.
fn document_intent(query: &str) -> bool {
    let words = step29_words(query);
    const DOC: &[&str] = &[
        "документ", "документы", "документа", "договор", "договора", "договоры",
        "письмо", "письма", "претензия", "претензии", "претензию", "акт", "акта",
        "отчет", "отчета", "отчеты", "отчетом", "инструкция", "инструкцию",
        "регламент", "заявление", "заявления", "накладная", "счет", "счета",
        "таблица", "таблицу", "таблицы", "ведомость", "расчет", "расчета",
    ];
    const NOT_DOC: &[&str] = &[
        "фото", "фотография", "фотографию", "изображение", "изображения", "картинка",
        "картинку", "скриншот", "снимок", "скан", "сканы", "скана",
        "код", "скрипт", "функция", "python", "javascript", "исходники",
    ];
    words.iter().any(|w| DOC.contains(&w.as_str()))
        && !words.iter().any(|w| NOT_DOC.contains(&w.as_str()))
}

fn document_priority(ext: Option<&str>) -> f32 {
    match ext.unwrap_or("").to_ascii_lowercase().as_str() {
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "csv" | "txt"
        | "md" | "rtf" | "odt" | "ods" | "ppt" | "pptx" | "epub" => 0.006,
        _ => 0.0,
    }
}

/// Stronger filename evidence requires at least two independent query terms.
/// A single incidental word in the title must not beat a relevant passage.
fn filename_evidence(query: &str, filename: &str) -> f32 {
    const SKIP: &[&str] = &[
        "найди", "найти", "покажи", "мне", "файл", "файла", "файлы", "документ",
        "документы", "где", "который", "котором", "что", "с", "со", "в", "на",
        "по", "и", "из", "для", "это", "тот", "та", "как", "нужно",
    ];
    let wanted: Vec<String> = step29_words(query).into_iter()
        .filter(|w| !SKIP.contains(&w.as_str()))
        .map(|w| step29_stem(&w))
        .filter(|w| w.chars().count() >= 3)
        .collect();
    if wanted.len() < 2 { return 0.0; }
    let name = filename.rsplit_once('.').map(|x| x.0).unwrap_or(filename);
    let have: Vec<String> = step29_words(name).iter().map(|w| step29_stem(w)).collect();
    let mut unique = std::collections::HashSet::new();
    let distinct: Vec<&String> = wanted.iter().filter(|w| unique.insert(w.as_str())).collect();
    if distinct.len() < 2 { return 0.0; }
    let count = distinct.iter().filter(|w| have.iter().any(|n| n.as_str() == w.as_str())).count();
    if count < 2 { 0.0 } else { 0.008 * count as f32 / distinct.len() as f32 }
}

/// Filename FTS candidates are retrieved independently of the main BM25 pool.
/// Without this branch a matching title may be below the 50-100 row cap if
/// numerous large file contents repeat the same terms.
fn filename_fts_candidates(conn: &Connection, query: &str, limit: usize) -> Result<Vec<i64>> {
    const SKIP: &[&str] = &["найди", "найти", "покажи", "мне", "файл", "файла",
        "файлы", "документ", "документы", "документа", "где", "котором", "который",
        "на", "с", "в", "про", "по", "из", "и", "или", "для", "тот", "та",
        "данные", "данными", "сведения", "сведениями"];
    let mut terms = Vec::new();
    for word in step29_words(query) {
        if word.chars().count() < 3 || SKIP.contains(&word.as_str()) { continue; }
        let term = step29_stem(&word);
        if !terms.contains(&term) { terms.push(term); }
        if terms.len() >= 4 { break; }
    }
    if terms.len() < 2 || limit == 0 { return Ok(Vec::new()); }
    let expr = format!("name : ({})", terms.iter()
        .map(|s| format!("\"{s}\"*"))
        .collect::<Vec<_>>().join(" AND "));
    let mut stmt = conn.prepare(
        "SELECT rowid FROM files_fts WHERE files_fts MATCH ?1
         AND rank MATCH 'bm25(8.0,2.0,1.0)' ORDER BY rank LIMIT ?2"
    )?;
    let mapped = stmt.query_map(rusqlite::params![expr, limit as i64], |r| r.get::<_, i64>(0))?;
    Ok(mapped.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Address-constrained compound query: at least two subject words followed by
/// a street-like name and an exact house number. Conservative by design:
/// ordinary dates, invoice numbers and short queries use the original ranking.
/// The words are extracted once per query; the user-visible query is untouched.
#[derive(Debug, Clone)]
pub(crate) struct AddressIntent {
    topic: [String; 2],
    street: String,
    house: String,
}

fn step29_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    for ch in input.chars().chain(std::iter::once(' ')) {
        if ch.is_alphanumeric() {
            for c in ch.to_lowercase() { word.push(if c == 'ё' { 'е' } else { c }); }
        } else if !word.is_empty() {
            words.push(std::mem::take(&mut word));
        }
    }
    words
}

/// Very limited inflection tolerance for Russian long words. Not a stemmer.
/// Consonant-ending identifiers remain intact; exact house numbers stay exact.
fn step29_stem(word: &str) -> String {
    let ends = ["иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "иях",
        "овой", "ого", "ями", "иях", "ами", "ой", "ей", "ый", "ий", "ая",
        "ое", "ые", "ых", "их", "ам", "ям", "ом", "ем", "ов", "ев", "ах", "ях",
        "а", "я", "ы", "и", "е", "у", "ю"];
    if word.chars().count() >= 6 && word.chars().all(|c| ('а'..='я').contains(&c)) {
        for ending in ends {
            if word.ends_with(ending) && word.chars().count() >= ending.chars().count() + 4 {
                return word[..word.len() - ending.len()].to_owned();
            }
        }
    }
    word.to_owned()
}

impl AddressIntent {
    pub(crate) fn parse(query: &str) -> Option<Self> {
        let words = step29_words(query);
        if words.len() < 4 || words.len() > 15 { return None; }
        let house = words.last()?.clone();
        if house.parse::<u32>().ok().filter(|n| *n > 0 && *n <= 999).is_none() { return None; }
        // Spoken or typed 'улица Учебная дом 28' is an address, not 'дом 28'.
        const ADDRESS_FILLERS: &[&str] = &["дом", "д", "дома", "номер", "№", "корпус"];
        const TOPIC_FILLERS: &[&str] = &["найди", "найти", "покажи", "мне", "документ",
            "документы", "файл", "файлы", "на", "в", "по", "из", "и", "о", "об", "с",
            "улица", "улице", "ул", "адрес", "адресу", "город", "г", "пожалуйста"];
        let street_pos = (0..words.len()-1).rev().find(|&i| !ADDRESS_FILLERS.contains(&words[i].as_str()))?;
        if street_pos + 4 < words.len() || street_pos == 0 { return None; }
        let street = step29_stem(&words[street_pos]);
        // Reject common date/identifier contexts without a street name.
        const NOT_STREET: &[&str] = &["январ", "феврал", "март", "апрел", "ма", "июн", "июл",
            "август", "сентябр", "октябр", "ноябр", "декабр", "номер", "артикул",
            "счет", "страниц", "пункт", "стат", "раздел", "года", "год", "числ"];
        if street.chars().count() < 4 || !street.chars().all(|c| ('а'..='я').contains(&c))
            || NOT_STREET.iter().any(|w| street.starts_with(w)) { return None; }
        let topic: Vec<String> = words[..street_pos].iter()
            .filter(|w| !TOPIC_FILLERS.contains(&w.as_str()))
            .map(|w| step29_stem(w))
            .filter(|w| w.chars().count() >= 4)
            .take(2).collect();
        if topic.len() != 2 || topic[0] == topic[1] { return None; }
        Some(Self { topic: [topic[0].clone(), topic[1].clone()], street, house })
    }

    /// FTS5 selects compound candidates *before* the generic pair fallback
    /// truncates a high-frequency subject like 'договор аренды'. A token's
    /// prefix is used only for Russian text; the house number is exact.
    pub(crate) fn fts_expression(&self) -> String {
        format!("\"{}\"* AND \"{}\"* AND NEAR(\"{}\"* \"{}\", 4)",
            self.topic[0], self.topic[1], self.street, self.house)
    }

    fn has_joint_document_evidence(&self, filename: &str, body_sample: &str) -> bool {
        let title = step29_words(filename);
        let body = step29_words(body_sample);
        let topic_present = self.topic.iter().all(|term| {
            title.iter().chain(body.iter()).any(|w| w.starts_with(term))
        });
        if !topic_present { return false; }
        [title.as_slice(), body.as_slice()].iter().any(|words| {
            words.iter().enumerate().any(|(i, word)| {
                if !word.starts_with(&self.street) { return false; }
                words.iter().skip(i+1).take(4).any(|candidate| candidate == &self.house)
                    && !words.iter().skip(i+1).take_while(|w| *w != &self.house)
                        .any(|w| w.chars().all(|c| c.is_ascii_digit()))
            })
        })
    }
}

/// Final frecency sorting happens in the Tauri adapter. Keep evidence-led
/// results ahead after *all* score changes; for other queries this does nothing.
pub fn sort_joint_address_first(hits: &mut [crate::files::FileHit]) {
    if !hits.iter().any(|hit| hit.match_kind.as_deref() == Some("Тема + адрес")) {
        return;
    }
    hits.sort_by(|a, b| {
        (b.match_kind.as_deref() == Some("Тема + адрес"))
            .cmp(&(a.match_kind.as_deref() == Some("Тема + адрес")))
            .then_with(|| b.score.total_cmp(&a.score))
            .then_with(|| a.path.cmp(&b.path))
    });
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    #[serde(flatten)]
    pub repo: Repo,
    pub score: f32,
}

/// Reciprocal Rank Fusion over ranked id lists. Input lists are best-first.
pub fn rrf_fuse(lists: &[Vec<i64>]) -> Vec<(i64, f32)> {
    use std::collections::HashMap;
    let mut scores: HashMap<i64, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(i64, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// All embedding vectors held resident in memory. Reload after any embed or
/// sync pass; per-query loading from SQLite does not survive chunk-scale
/// corpora (tens of MB per keystroke).
pub struct VectorStore {
    /// (repo_id, chunk vector); repo_id repeats across a repo's README chunks.
    pub repos: Vec<(i64, Vec<f32>)>,
    /// (file_id, chunk vector); file_id repeats across a file's chunks.
    pub file_chunks: Vec<(i64, Vec<f32>)>,
    pub images: Vec<(i64, Vec<f32>)>,
    pub bookmarks: Vec<(i64, Vec<f32>)>,
    pub history: Vec<(i64, Vec<f32>)>,
    pub clips: Vec<(i64, Vec<f32>)>,
    /// (shot_id, SigLIP vector) — one entry per representative video frame.
    pub video_shots: Vec<(i64, Vec<f32>)>,
    /// (clip_id, SigLIP vector) — image clips, searchable by description.
    pub clip_images: Vec<(i64, Vec<f32>)>,
}

impl VectorStore {
    pub fn load(conn: &Connection) -> Result<Self> {
        Ok(Self {
            repos: db::all_repo_chunk_embeddings(conn)?,
            file_chunks: crate::files::all_file_chunk_embeddings(conn)?,
            images: crate::files::all_image_embeddings(conn)?,
            bookmarks: crate::bookmarks::all_bookmark_embeddings(conn)?,
            history: crate::history::all_history_embeddings(conn)?,
            clips: crate::clips::all_clip_embeddings(conn)?,
            video_shots: crate::videos::all_shot_embeddings(conn)?,
            clip_images: crate::clips::all_clip_image_embeddings(conn)?,
        })
    }

    pub fn empty() -> Self {
        Self {
            repos: Vec::new(),
            file_chunks: Vec::new(),
            images: Vec::new(),
            bookmarks: Vec::new(),
            history: Vec::new(),
            clips: Vec::new(),
            video_shots: Vec::new(),
            clip_images: Vec::new(),
        }
    }
}

/// Exact, bounded-memory top-k. Unlike collecting and sorting EVERY vector,
/// the heap holds at most `limit` winners. Ranking semantics (score desc,
/// file_id asc on ties) are unchanged. There is no approximation/ANN recall loss.
#[derive(Clone, Copy, Debug)]
struct VectorWinner { id: i64, score: f32 }
impl PartialEq for VectorWinner {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.total_cmp(&other.score).is_eq()
    }
}
impl Eq for VectorWinner {}
impl Ord for VectorWinner {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score).then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for VectorWinner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
fn bounded_vector_topk(
    matches: impl IntoIterator<Item=(i64, f32)>, limit: usize,
) -> Vec<(i64, f32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    if limit == 0 { return Vec::new(); }
    let mut heap: BinaryHeap<Reverse<VectorWinner>> = BinaryHeap::with_capacity(limit + 1);
    for (id, score) in matches {
        let candidate = VectorWinner { id, score };
        if heap.len() < limit {
            heap.push(Reverse(candidate));
        } else if heap.peek().is_some_and(|Reverse(worst)| candidate > *worst) {
            heap.pop();
            heap.push(Reverse(candidate));
        }
    }
    let mut out: Vec<(i64, f32)> = heap.into_iter()
        .map(|Reverse(winner)| (winner.id, winner.score)).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Exact vector retrieval without sorting the entire vector collection.
pub fn top_similar(all: &[(i64, Vec<f32>)], qvec: &[f32], limit: usize) -> Vec<(i64, f32)> {
    bounded_vector_topk(
        all.iter().filter(|(_, v)| v.len() == qvec.len())
            .map(|(id, v)| (*id, dot(qvec, v))), limit,
    )
}

/// Document-first exact vector retrieval: each file receives the score of
/// its best chunk. Only the final unique-file top-k uses a heap.
pub fn top_similar_grouped(
    all: &[(i64, Vec<f32>)], qvec: &[f32], limit: usize,
) -> Vec<(i64, f32)> {
    let mut best: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    for (id, v) in all {
        if v.len() != qvec.len() { continue; }
        let sim = dot(qvec, v);
        let entry = best.entry(*id).or_insert(f32::NEG_INFINITY);
        if sim > *entry { *entry = sim; }
    }
    bounded_vector_topk(best, limit)
}

/// Fuse an FTS ranking with any number of vector candidate lists: RRF over
/// ranks plus a small summed similarity term. Returns (id, score), best first.
pub fn rank_hybrid(fts: Vec<i64>, vecs: Vec<Vec<(i64, f32)>>) -> Vec<(i64, f32)> {
    let mut lists = vec![fts];
    let mut sims: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    for vs in vecs {
        for (id, sim) in &vs {
            *sims.entry(*id).or_default() += sim;
        }
        lists.push(vs.into_iter().map(|(id, _)| id).collect());
    }
    let mut fused = rrf_fuse(&lists);
    for (id, score) in fused.iter_mut() {
        *score += VEC_SIM_WEIGHT * sims.get(id).copied().unwrap_or(0.0);
    }
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

/// Ordering applied to matched repos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoSort {
    /// Hybrid retrieval score (default).
    Relevance,
    /// Most recently starred first.
    Starred,
    /// Highest star count first.
    Stars,
}

/// Hybrid search. `qvec` is None when the embedding model is not ready — FTS-only
/// then. Non-relevance sorts reorder the matched candidate set.
pub fn search(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    sort: RepoSort,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    // sorting only applies to search matches; the empty-query browse view is
    // always the recently-starred list
    if query.trim().is_empty() {
        return Ok(db::recent_repos(conn, limit)?
            .into_iter()
            .map(|repo| SearchResult { repo, score: 0.0 })
            .collect());
    }

    let fts = db::fts_search(conn, query, CANDIDATES_PER_LIST)?;
    let vecs = match qvec {
        // a repo ranks by its best README chunk
        Some(qvec) => vec![top_similar_grouped(&store.repos, qvec, CANDIDATES_PER_LIST)],
        None => vec![],
    };
    let fused = rank_hybrid(fts, vecs);
    let scores: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
    // alternate sorts reorder exactly the top-`limit` relevant matches — never
    // a wider pool, so a famous-but-barely-related repo cannot crowd in
    let ids: Vec<i64> = fused.iter().take(limit).map(|(id, _)| *id).collect();
    let mut results: Vec<SearchResult> = db::repos_by_ids(conn, &ids)?
        .into_iter()
        .map(|repo| {
            let score = scores.get(&repo.id).copied().unwrap_or(0.0);
            SearchResult { repo, score }
        })
        .collect();
    results.truncate(limit);
    match sort {
        RepoSort::Relevance => {}
        RepoSort::Starred => results.sort_by(|a, b| {
            b.repo
                .starred_at
                .cmp(&a.repo.starred_at)
                .then(a.repo.id.cmp(&b.repo.id))
        }),
        RepoSort::Stars => results.sort_by(|a, b| {
            b.repo
                .stars
                .cmp(&a.repo.stars)
                .then(a.repo.id.cmp(&b.repo.id))
        }),
    }
    Ok(results)
}

/// What a local search may return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LocalScope {
    /// Text files and images mixed (default).
    All,
    /// Text files only.
    Text,
    /// Images only.
    Images,
    /// Videos only (filename matches + semantic shot search — handled by
    /// [`search_videos_scope`], not [`search_files`]).
    Videos,
}

/// Bulk fetch metadata for all candidates. Preserve the original fused order;
/// SQL IN clause row order is undefined. BLOB thumbnails are deferred until
/// after final ranking (normally at most 50 rows).
fn ranking_file_hits(
    conn: &Connection, ids: &[i64], scores: &std::collections::HashMap<i64, f32>,
) -> Result<Vec<crate::files::FileHit>> {
    if ids.is_empty() { return Ok(Vec::new()); }
    let marks = (0..ids.len()).map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT id, path, name, ext, size, mtime FROM files WHERE id IN ({marks})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
        Ok(crate::files::FileHit {
            id: row.get(0)?,
            path: row.get(1)?,
            name: row.get(2)?,
            ext: row.get(3)?,
            size: row.get(4)?,
            mtime: row.get(5)?,
            score: 0.0,
            thumb: None,
            snippet: None,
            match_kind: None,
        })
    })?;
    let mut by_id = std::collections::HashMap::new();
    for row in rows { let hit = row?; by_id.insert(hit.id, hit); }
    let mut hits = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(mut hit) = by_id.remove(id) {
            hit.score = scores.get(id).copied().unwrap_or(0.0);
            hits.push(hit);
        }
    }
    Ok(hits)
}

/// Read the same bounded source samples as the previous per-row query, but in
/// one prepared SQLite statement. Topic score computation is unchanged.
fn ranking_text_samples(
    conn: &Connection, ids: &[i64],
) -> Result<std::collections::HashMap<i64, String>> {
    if ids.is_empty() { return Ok(std::collections::HashMap::new()); }
    let marks = (0..ids.len()).map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT id, COALESCE(substr(content, 1, 65536), '') FROM files WHERE id IN ({marks})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(ids.iter()), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut out = std::collections::HashMap::with_capacity(ids.len());
    for row in rows { let (id, sample) = row?; out.insert(id, sample); }
    Ok(out)
}

/// Preserve thumbnail bytes in the final FileHit objects, fetching blobs only
/// for survivors rather than for hundreds of unreturned candidates.
fn attach_final_thumbnails(conn: &Connection, hits: &mut [crate::files::FileHit]) -> Result<()> {
    use base64::Engine;
    if hits.is_empty() { return Ok(()); }
    let marks = (0..hits.len()).map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!("SELECT id, thumb FROM files WHERE id IN ({marks})");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(
        rusqlite::params_from_iter(hits.iter().map(|h| h.id)),
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
    )?;
    let mut thumbs = std::collections::HashMap::new();
    for row in rows { let (id, data) = row?; thumbs.insert(id, data); }
    for hit in hits {
        hit.thumb = thumbs.remove(&hit.id).flatten().map(|data|
            base64::engine::general_purpose::STANDARD.encode(data)
        );
    }
    Ok(())
}

/// Only page numbers read from the PDF parser may be exposed as provenance.
/// This syntax is reserved for our indexed PDF text; old indexes never
/// receive an inferred page number.
fn indexed_pdf_page(line: &str) -> Option<usize> {
    let value = line.strip_prefix("[Страница PDF ")?.strip_suffix(']')?;
    let number = value.parse::<usize>().ok()?;
    (number > 0).then_some(number)
}

/// Verify the exact highlighted text occurs once in the bounded indexed
/// sample before using its preceding page marker. Ambiguity means no page
/// claim, not a guessed page. This is deliberately conservative.
fn pdf_fts_location(snippet: &str, indexed: &str) -> Option<String> {
    let needle = snippet.split('\u{1}').nth(1)?.split('\u{2}').next()?;
    if needle.chars().count() < 4 { return None; }
    let mut matches = indexed.match_indices(needle);
    let (offset, _) = matches.next()?;
    if matches.next().is_some() { return None; }
    let prefix = &indexed[..offset];
    let number = prefix.lines().filter_map(indexed_pdf_page).last()?;
    Some(format!("Страница {number}"))
}

/// A search result is a *document*, not a bare chunk. Use the indexed text
/// structure to show a useful passage and, for XLSX, the actual worksheet and
/// cell coordinate. This is intentionally conservative: if there is no lexical
/// evidence, return None instead of fabricating a location for a vector hit.
/// No files are read from disk and existing FTS-highlighted snippets are kept.
fn structured_passage(query: &str, ext: Option<&str>, indexed: &str) -> Option<String> {
    const NOISE: &[&str] = &["найди", "найти", "покажи", "пожалуйста", "где", "документ",
        "документы", "файл", "файла", "файлы", "таблица", "таблицу", "данные",
        "который", "которая", "котором", "мне", "про", "как", "для"];
    let words: Vec<String> = step29_words(query).into_iter()
        .filter(|w| !NOISE.contains(&w.as_str()))
        .filter(|w| w.chars().count() >= 4 || w.chars().all(|c| c.is_ascii_digit()))
        .take(12).map(|w| step29_stem(&w)).collect();
    if words.is_empty() { return None; }
    let is_xlsx = matches!(ext, Some("xlsx" | "xls"));
    let is_pdf = matches!(ext, Some("pdf"));
    let mut worksheet = "";
    let mut pdf_page: Option<usize> = None;
    let mut heading = "";
    let mut best: Option<(usize, String)> = None;
    for line in indexed.lines().take(12_000) {
        let line = line.trim();
        if line.is_empty() { continue; }
        if is_xlsx && line.starts_with("Лист: ") {
            worksheet = line.trim_start_matches("Лист: ").trim();
            continue;
        }
        if is_pdf {
            if let Some(number) = indexed_pdf_page(line) {
                pdf_page = Some(number);
                continue;
            }
        }
        if line.starts_with('#') && line.len() < 160 {
            heading = line.trim_start_matches('#').trim();
            continue;
        }
        if line.chars().count() > 1400 { continue; }
        let normalized: std::collections::HashSet<String> = step29_words(line)
            .into_iter().map(|w| step29_stem(&w)).collect();
        let score = words.iter().filter(|w| normalized.contains(*w)).count();
        if score == 0 { continue; }
        if best.as_ref().is_some_and(|(best_score, _)| score <= *best_score) { continue; }
        let place = if is_xlsx {
            let coord = line.split_once(':').map(|(l, _)| l.trim()).unwrap_or("");
            let valid = !coord.is_empty() && coord.len() <= 10
                && coord.chars().any(|c| c.is_ascii_digit())
                && coord.chars().all(|c| c.is_ascii_alphanumeric());
            if !worksheet.is_empty() && valid { format!("Лист «{worksheet}», {coord}: ") }
            else if !worksheet.is_empty() { format!("Лист «{worksheet}»: ") }
            else { String::new() }
        } else if is_pdf {
            pdf_page.map(|n| format!("Страница {n}: ")).unwrap_or_default()
        } else if !heading.is_empty() { format!("Раздел «{heading}»: ") }
        else { String::new() };
        let excerpt: String = line.chars().take(300).collect();
        best = Some((score, format!("{place}{excerpt}")));
        if score == words.len() { break; }
    }
    best.map(|(_, passage)| passage)
}

/// Recover worksheet/cell metadata only if the highlighted text actually
/// occurs in the indexed XLSX body. Never manufacture a worksheet coordinate.
fn xlsx_fts_location(snippet: &str, indexed: &str) -> Option<String> {
    let needle = snippet.split('\u{1}').nth(1)?.split('\u{2}').next()?;
    if needle.chars().count() < 3 { return None; }
    let pos = indexed.find(needle)?;
    let context = &indexed[..pos];
    let sheet = context.lines().rev()
        .find_map(|l| l.trim().strip_prefix("Лист: "))?;
    let line = indexed[pos..].lines().next().unwrap_or("");
    let before_line = context.rsplit('\n').next().unwrap_or("");
    let cell_line = if before_line.contains(':') { before_line } else { line };
    let coord = cell_line.split_once(':').map(|(x, _)| x.trim()).unwrap_or("");
    let valid = !coord.is_empty() && coord.len() <= 10
        && coord.chars().any(|c| c.is_ascii_digit())
        && coord.chars().all(|c| c.is_ascii_alphanumeric());
    if valid { Some(format!("Лист «{sheet}», {coord}")) }
    else { Some(format!("Лист «{sheet}»")) }
}

/// Hybrid search over local files: filename/content FTS + e5 text vectors +
/// SigLIP image vectors, all fused by rank. The two vector spaces cover
/// disjoint id sets (text files vs images), so their similarities never mix.
/// `scope` restricts which kinds of files come back.
pub mod decision;
pub fn search_files(conn: &Connection, store: &VectorStore, query: &str, qvec: Option<&[f32]>, image_qvec: Option<&[f32]>, scope: LocalScope, limit: usize) -> Result<Vec<crate::files::FileHit>> {
    if decision::enabled() && matches!(scope,LocalScope::All|LocalScope::Text) { decision::search(conn,store,query,qvec,image_qvec,scope,limit) }
    else { search_files_legacy(conn,store,query,qvec,image_qvec,scope,limit) }
}

pub fn search_files_legacy(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    image_qvec: Option<&[f32]>,
    scope: LocalScope,
    limit: usize,
) -> Result<Vec<crate::files::FileHit>> {
    use crate::files;
    // `ext:pdf`, `>10mb`, `7d`, `in:docs` come out of the text first; the
    // caller has already embedded the cleaned text (see search_local)
    let (mut filters, text) = crate::filters::parse(query);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    if text.trim().is_empty() {
        return if filters.is_empty() {
            files::recent_files(conn, limit)
        } else {
            files::files_matching(conn, &filters, now, limit)
        };
    }
    let normalized = normalize_local_query_text(&text);
    let spoken = if filters.exts.is_empty() { spoken_format(&normalized) } else { None };
    let cleaned = spoken.as_ref().map(|s| s.text.as_str()).unwrap_or(&normalized);
    if let Some(format) = &spoken {
        filters.exts.extend(format.exts.iter().map(|e| (*e).to_owned()));
    }
    // "найди Excel" means browse Excel files, not an empty keyword search.
    if cleaned.trim().is_empty() && spoken.is_some() {
        return files::files_matching(conn, &filters, now, limit);
    }
    let query = cleaned;
    let doc_intent = document_intent(&normalized);
    let address_intent = AddressIntent::parse(query);
    // Performance diagnostics: measure each existing stage without changing results.
    let perf_start = std::time::Instant::now();
    // filters sieve the candidates after ranking, so widen the net when any
    // are set or a narrow filter could empty a full page of good matches
    let per_list = if filters.is_empty() { CANDIDATES_PER_LIST * 2 } else { CANDIDATES_PER_LIST * 4 };
    let format_start = std::time::Instant::now();
    let mut fts_hits = if let Some(s) = &spoken {
        format_fts_candidates(conn, query, s.exts, per_list)?
    } else { Vec::new() };
    let format_ms = format_start.elapsed().as_millis();
    let general_start = std::time::Instant::now();
    let general = files::files_fts_search(conn, query, per_list)?;
    let general_ms = general_start.elapsed().as_millis();
    for item in general {
        if fts_hits.len() >= per_list { break; }
        if !fts_hits.iter().any(|x| x.0 == item.0) { fts_hits.push(item); }
    }
    let title_start = std::time::Instant::now();
    let title_hits = filename_fts_candidates(conn, query, CANDIDATES_PER_LIST)?;
    let title_ms = title_start.elapsed().as_millis();
    if perf_start.elapsed().as_millis() >= 40 {
        // Numeric-only: never log the user query, filename or document path.
        log::info!("local_fts_detail: format={}ms general={}ms title={}ms spoken={} candidates={}",
            format_ms, general_ms, title_ms, spoken.is_some(), fts_hits.len());
    }
    let title_ids: std::collections::HashSet<i64> = title_hits.iter().copied().collect();
    let perf_fts = perf_start.elapsed();
    let snippets: std::collections::HashMap<i64, String> = fts_hits
        .iter()
        .filter(|(_, s)| s.contains('\u{1}'))
        .map(|(id, s)| (*id, s.clone()))
        .collect();
    let fts: Vec<i64> = fts_hits.into_iter().map(|(id, _)| id).collect();
    let fts_ids: std::collections::HashSet<i64> = fts.iter().copied().collect();
    let trace = std::env::var("LOCALSEARCH_SEARCH_TRACE").ok().as_deref() == Some("1");
    let fts_ranks: std::collections::HashMap<i64, usize> = if trace {
        fts.iter().enumerate().map(|(i, id)| (*id, i + 1)).collect()
    } else { std::collections::HashMap::new() };
    let mut text_ranks = std::collections::HashMap::new();
    let mut image_ranks = std::collections::HashMap::new();
    let mut text_vec_ids = std::collections::HashSet::new();
    let mut image_vec_ids = std::collections::HashSet::new();
    let mut vecs = Vec::new();
    if let Some(qvec) = qvec {
        // a file ranks by its best chunk. This used to be skipped for the
        // Images scope (images had no text), but OCR text now lives in the
        // same chunk space — images without text simply aren't in it, so
        // there is nothing to exclude anymore.
        let candidates = top_similar_grouped(&store.file_chunks, qvec, per_list);
        if trace { text_ranks.extend(candidates.iter().enumerate().map(|(i, (id, _))| (*id, i + 1))); }
        text_vec_ids.extend(candidates.iter().map(|(id, _)| *id));
        vecs.push(candidates);
    }
    if scope != LocalScope::Text {
        if let Some(image_qvec) = image_qvec {
            let candidates = top_similar(&store.images, image_qvec, per_list);
            if trace { image_ranks.extend(candidates.iter().enumerate().map(|(i, (id, _))| (*id, i + 1))); }
            image_vec_ids.extend(candidates.iter().map(|(id, _)| *id));
            vecs.push(candidates);
        }
    }
    let perf_vectors = perf_start.elapsed();
    let mut fused = rank_hybrid(fts, vecs);
    // A separate lexical evidence list, added with RRF-scale weights. A file
    // matching the entire title enters the pool even if the generic FTS list
    // was swamped by long contents. No model or index change is needed.
    {
        let mut combined: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
        for (position, id) in title_hits.iter().enumerate() {
            *combined.entry(*id).or_default() += 0.60 / (RRF_K + position as f32 + 1.0);
        }
        fused = combined.into_iter().collect();
        fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    let perf_fused = perf_start.elapsed();
    let scores: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
    // fetch the full candidate set, filter by scope, then truncate
    let ids: Vec<i64> = fused.iter().map(|(id, _)| *id).collect();
    let mut hits = ranking_file_hits(conn, &ids, &scores)?;
    let perf_fetched = perf_start.elapsed();
    let candidate_count = hits.len();
    // Scope and explicit filters remove candidates regardless of topicality.
    // Exclude them before expensive 64K content reads. Remaining scores and
    // relative order are unchanged: no rejected candidate can re-enter later.
    if !filters.is_empty() {
        hits.retain(|h| filters.matches(h.ext.as_deref(), h.size, h.mtime, &h.path, now));
    }
    match scope {
        LocalScope::All => {}
        LocalScope::Text => hits.retain(|h| !files::is_image_ext(h.ext.as_deref())),
        LocalScope::Images => hits.retain(|h| files::is_image_ext(h.ext.as_deref())),
        LocalScope::Videos => hits.retain(|h| crate::videos::is_video_ext(h.ext.as_deref())),
    }
    // Same topicality scores, with query terms parsed once and bounded verified-by-value
    // document profiles reused across queries. Candidate set and 64K limit unchanged.
    let mut topic_cache_hits = 0usize;
    let mut topic_cache_misses = 0usize;
    let mut joint_address_ids = std::collections::HashSet::new();
    let mut ranked_samples = std::collections::HashMap::new();
    {
        let prepared = crate::document_rank::PreparedTopicQuery::new(query);
        if !prepared.is_empty() {
            let ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
            let samples = ranking_text_samples(conn, &ids)?;
            for hit in &mut hits {
                let sample = samples.get(&hit.id)
                    .ok_or(rusqlite::Error::QueryReturnedNoRows)?;
                let (profile, cached) = crate::document_rank::cached_topic_profile(
                    hit.id, &hit.name, sample,
                );
                if cached { topic_cache_hits += 1; } else { topic_cache_misses += 1; }
                hit.score += prepared.boost(&profile);
                hit.score += filename_evidence(query, &hit.name);
                if doc_intent && scope != LocalScope::Images {
                    hit.score += document_priority(hit.ext.as_deref());
                }
                if let Some(intent) = &address_intent {
                    if intent.has_joint_document_evidence(&hit.name, sample) {
                        joint_address_ids.insert(hit.id);
                    }
                }
            }
            ranked_samples = samples;
        }
        // Materialized as a separate priority tier after filters; no magic score
        // multiplier is mixed into RRF or the topicality model.
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.path.cmp(&b.path)));
    }
    let perf_topic = perf_start.elapsed();
    // Exact conjunction beats partial/semantic hits; retain ordinary score
    // order within each tier and preserve every partial match as a fallback.
    if address_intent.is_some() {
        hits.sort_by(|a, b| joint_address_ids.contains(&b.id)
            .cmp(&joint_address_ids.contains(&a.id))
            .then_with(|| b.score.total_cmp(&a.score)).then_with(|| a.path.cmp(&b.path)));
    }
    hits.truncate(limit);
    attach_final_thumbnails(conn, &mut hits)?;
    if trace {
        // Numeric evidence only: do not log a document's name, path or query.
        for (i, hit) in hits.iter().take(10).enumerate() {
            log::info!(
                "local_rank_trace: pos={} id={} fts_rank={:?} name_match={} text_vector_rank={:?} image_vector_rank={:?} fused_score={:.5} final_score={:.5} spoken_format={} document_intent={}",
                i + 1, hit.id, fts_ranks.get(&hit.id), title_ids.contains(&hit.id),
                text_ranks.get(&hit.id), image_ranks.get(&hit.id),
                scores.get(&hit.id).copied().unwrap_or(0.0),
                hit.score, spoken.is_some(), doc_intent,
            );
        }
    }
    for h in &mut hits {
        h.snippet = snippets.get(&h.id).cloned();
        if let (Some(snippet), Some(sample)) = (h.snippet.as_ref(), ranked_samples.get(&h.id)) {
            let location = match h.ext.as_deref() {
                Some("xlsx") => xlsx_fts_location(snippet, sample),
                Some("pdf") => pdf_fts_location(snippet, sample),
                _ => None,
            };
            if let Some(where_found) = location {
                h.snippet = Some(format!("{where_found}: {snippet}"));
            }
        }
        // Preserve FTS highlights; add a verified XLSX location where possible.
        // Fill only missing snippets with evidence from already loaded text.
        if h.snippet.as_deref().is_none_or(str::is_empty) {
            if let Some(sample) = ranked_samples.get(&h.id) {
                h.snippet = structured_passage(query, h.ext.as_deref(), sample);
            }
        }
        let keyword = fts_ids.contains(&h.id);
        let named = title_ids.contains(&h.id);
        let semantic = text_vec_ids.contains(&h.id);
        let visual = image_vec_ids.contains(&h.id);
        h.match_kind = Some(if joint_address_ids.contains(&h.id) {
            "Тема + адрес"
        } else if named && keyword && semantic {
            "Имя + текст + смысл"
        } else if named && keyword {
            "Имя + текст"
        } else if named {
            "Имя файла"
        } else { match (keyword, semantic, visual) {
            (true, true, _) => "Текст + смысл",
            (true, false, _) => "Совпадение текста",
            (false, true, _) => "По смыслу",
            (false, false, true) => "По изображению",
            _ => "Совпадение",
        }}.to_string());
    }
    let perf_done = perf_start.elapsed();
    if perf_done.as_millis() >= 100 {
        log::info!(
            "local_search_phases: total={}ms fts={}ms vectors={}ms fusion={}ms file_fetch={}ms topic={}ms rest={}ms candidates={} vec_count={} topic_cache_hits={} topic_cache_misses={}",
            perf_done.as_millis(),
            perf_fts.as_millis(),
            (perf_vectors - perf_fts).as_millis(),
            (perf_fused - perf_vectors).as_millis(),
            (perf_fetched - perf_fused).as_millis(),
            (perf_topic - perf_fetched).as_millis(),
            (perf_done - perf_topic).as_millis(),
            candidate_count,
            store.file_chunks.len(),
            topic_cache_hits,
            topic_cache_misses,
        );
    }
    Ok(hits)
}

/// Hybrid search over browser bookmarks: title/url/folder FTS + one e5
/// vector per bookmark.
pub fn search_bookmarks(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<crate::bookmarks::BookmarkHit>> {
    use crate::bookmarks;
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let fts = bookmarks::bookmarks_fts_search(conn, query, CANDIDATES_PER_LIST)?;
    let keyword: std::collections::HashSet<i64> = fts.iter().copied().collect();
    let vecs = match qvec {
        Some(qvec) => vec![top_similar(&store.bookmarks, qvec, CANDIDATES_PER_LIST)],
        None => vec![],
    };
    let fused = rank_hybrid(fts, vecs);
    let scores: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
    // over-fetch, then collapse the same URL bookmarked in several browsers
    // (Edge + Edge SxS…) into its best-ranked row
    let ids: Vec<i64> = fused.iter().take(limit * 2).map(|(id, _)| *id).collect();
    let mut hits = bookmarks::bookmarks_by_ids(conn, &ids, &scores)?;
    let keyword_urls: std::collections::HashSet<String> =
        hits.iter().filter(|h| keyword.contains(&h.id)).map(|h| h.url.clone()).collect();
    let mut seen = std::collections::HashSet::new();
    hits.retain(|h| seen.insert(h.url.clone()));
    hits.truncate(limit);
    let urls: Vec<String> = hits.iter().map(|h| h.url.clone()).collect();
    let mut by_url = browsers_by_url(conn, "bookmarks", &urls)?;
    for h in &mut hits {
        h.fuzzy = !keyword_urls.contains(&h.url);
        h.browsers = by_url.remove(&h.url).unwrap_or_else(|| vec![h.browser.clone()]);
    }
    Ok(hits)
}

/// Hybrid search over browser history: title/url FTS + one e5 vector each,
/// with a visit-count boost so frequently-visited pages float up.
pub fn search_history(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<crate::history::HistoryHit>> {
    use crate::history;
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let fts = history::history_fts_search(conn, query, CANDIDATES_PER_LIST)?;
    let keyword: std::collections::HashSet<i64> = fts.iter().copied().collect();
    let vecs = match qvec {
        Some(qvec) => vec![top_similar(&store.history, qvec, CANDIDATES_PER_LIST)],
        None => vec![],
    };
    let fused = rank_hybrid(fts, vecs);
    let scores: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
    // over-fetch so the cross-browser URL dedup below can't starve the limit
    let ids: Vec<i64> = fused.iter().take(limit * 2).map(|(id, _)| *id).collect();
    let mut hits = history::history_by_ids(conn, &ids, &scores)?;
    // light visit-count boost: log-scaled so a popular page nudges ahead of a
    // once-visited one at similar relevance, without dominating the ranking
    for h in &mut hits {
        h.score += 0.001 * ((h.visit_count.max(1) as f32).ln());
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    let keyword_urls: std::collections::HashSet<String> =
        hits.iter().filter(|h| keyword.contains(&h.id)).map(|h| h.url.clone()).collect();
    // the same page visited in several browsers (Edge + Edge SxS…) collapses
    // into its best-scoring row
    let mut seen = std::collections::HashSet::new();
    hits.retain(|h| seen.insert(h.url.clone()));
    hits.truncate(limit);
    let urls: Vec<String> = hits.iter().map(|h| h.url.clone()).collect();
    let mut by_url = browsers_by_url(conn, "history", &urls)?;
    for h in &mut hits {
        h.fuzzy = !keyword_urls.contains(&h.url);
        h.browsers = by_url.remove(&h.url).unwrap_or_else(|| vec![h.browser.clone()]);
    }
    Ok(hits)
}

/// Every browser holding each of `urls` in `table` ("bookmarks" or
/// "history"), sorted, for the rows that collapse one URL kept in several.
fn browsers_by_url(
    conn: &Connection,
    table: &str,
    urls: &[String],
) -> Result<std::collections::HashMap<String, Vec<String>>> {
    debug_assert!(table == "bookmarks" || table == "history");
    let mut out: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    if urls.is_empty() {
        return Ok(out);
    }
    let marks = vec!["?"; urls.len()].join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT DISTINCT url, browser FROM {table} WHERE url IN ({marks}) ORDER BY browser"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(urls.iter()), |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (url, browser) = row?;
        out.entry(url).or_default().push(browser);
    }
    Ok(out)
}

/// Hybrid search over clipboard history; empty query shows most recent clips.
/// Text clips answer via FTS + e5; image clips via SigLIP text→image
/// (`image_qvec`) — rank fusion keeps the two score spaces apart.
pub fn search_clips(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    qvec: Option<&[f32]>,
    image_qvec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<crate::clips::ClipHit>> {
    use crate::clips;
    if query.trim().is_empty() {
        return clips::recent_clips(conn, limit);
    }
    let fts = clips::clips_fts_search(conn, query, CANDIDATES_PER_LIST)?;
    let mut vecs = Vec::new();
    if let Some(qvec) = qvec {
        vecs.push(top_similar(&store.clips, qvec, CANDIDATES_PER_LIST));
    }
    if let Some(iq) = image_qvec {
        vecs.push(top_similar(&store.clip_images, iq, CANDIDATES_PER_LIST));
    }
    let fused = rank_hybrid(fts, vecs);
    let scores: std::collections::HashMap<i64, f32> = fused.iter().copied().collect();
    let ids: Vec<i64> = fused.iter().take(limit).map(|(id, _)| *id).collect();
    clips::clips_by_ids(conn, &ids, &scores)
}

/// Image-to-image search: a query image's SigLIP vector against all indexed
/// image vectors. Scores are raw cosine similarities.
pub fn search_images(
    conn: &Connection,
    store: &VectorStore,
    image_qvec: &[f32],
    limit: usize,
) -> Result<Vec<crate::files::FileHit>> {
    search_images_filtered(conn, store, image_qvec, limit, None)
}

/// Filter candidate IDs before top-k and loading thumbnail payloads.
pub fn search_images_filtered(
    conn: &Connection,
    store: &VectorStore,
    image_qvec: &[f32],
    limit: usize,
    allowed: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<crate::files::FileHit>> {
    let scored = bounded_vector_topk(store.images.iter()
        .filter(|(id, v)| v.len() == image_qvec.len() && allowed.is_none_or(|ids| ids.contains(id)))
        .map(|(id, v)| (*id, dot(image_qvec, v))), limit);
    let scores: std::collections::HashMap<i64, f32> = scored.iter().copied().collect();
    let ids: Vec<i64> = scored.into_iter().map(|(id, _)| id).collect();
    crate::files::files_by_ids(conn, &ids, &scores)
}

/// Best shot per video for a SigLIP query vector: (file_id, shot_id, sim),
/// ranked best-first.
fn best_shot_per_video(
    conn: &Connection,
    store: &VectorStore,
    qvec: &[f32],
) -> Result<Vec<(i64, i64, f32)>> {
    if store.video_shots.is_empty() {
        return Ok(Vec::new());
    }
    let owners = crate::videos::shot_owners(conn)?;
    let scored = top_similar(&store.video_shots, qvec, store.video_shots.len());
    let mut best: std::collections::HashMap<i64, (i64, f32)> = std::collections::HashMap::new();
    for (shot_id, sim) in scored {
        let Some(&file_id) = owners.get(&shot_id) else { continue };
        let e = best.entry(file_id).or_insert((shot_id, f32::NEG_INFINITY));
        if sim > e.1 {
            *e = (shot_id, sim);
        }
    }
    let mut winners: Vec<(i64, i64, f32)> =
        best.into_iter().map(|(fid, (sid, sim))| (fid, sid, sim)).collect();
    winners.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.1.cmp(&b.1)));
    Ok(winners)
}

/// Video-shot search in the same SigLIP space (image or text query vector).
/// Shots are grouped per video — a video answers with its best shot only.
pub fn search_video_shots(
    conn: &Connection,
    store: &VectorStore,
    qvec: &[f32],
    limit: usize,
) -> Result<Vec<crate::videos::VideoHit>> {
    search_video_shots_filtered(conn, store, qvec, limit, None)
}

/// Filter video owners before loading the winning shot thumbnails.
pub fn search_video_shots_filtered(
    conn: &Connection,
    store: &VectorStore,
    qvec: &[f32],
    limit: usize,
    allowed: Option<&std::collections::HashSet<i64>>,
) -> Result<Vec<crate::videos::VideoHit>> {
    let winners: Vec<_> = best_shot_per_video(conn, store, qvec)?.into_iter()
        .filter(|(fid, _, _)| allowed.is_none_or(|ids| ids.contains(fid)))
        .take(limit).collect();
    let scores: std::collections::HashMap<i64, f32> =
        winners.iter().map(|(_, sid, sim)| (*sid, *sim)).collect();
    let ids: Vec<i64> = winners.into_iter().take(limit).map(|(_, sid, _)| sid).collect();
    crate::videos::hits_by_shot_ids(conn, &ids, &scores)
}

/// The dedicated "videos" scope: filename matches and semantic shot matches,
/// rank-fused (RRF — the two scores live on different scales). A video that
/// matches both ways answers once, preferring its shot representation (it
/// carries the time range and scene thumbnail).
pub fn search_videos_scope(
    conn: &Connection,
    store: &VectorStore,
    query: &str,
    image_qvec: Option<&[f32]>,
    limit: usize,
) -> Result<Vec<crate::videos::VideoHit>> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let name_ids = crate::videos::video_name_search(conn, query, CANDIDATES_PER_LIST)?;
    let shot_winners = match image_qvec {
        Some(qv) => best_shot_per_video(conn, store, qv)?,
        None => Vec::new(),
    };
    let sem_ids: Vec<i64> = shot_winners.iter().take(CANDIDATES_PER_LIST).map(|(f, _, _)| *f).collect();
    let mut shot_of: std::collections::HashMap<i64, (i64, f32)> =
        shot_winners.into_iter().map(|(f, s, sim)| (f, (s, sim))).collect();
    // third list: shots whose OCR text contains the query. An exact text hit
    // pins the shot too — it wins over the SigLIP pick, so Enter jumps to
    // where the text actually appears.
    let ocr_hits = crate::videos::video_ocr_search(conn, query, CANDIDATES_PER_LIST)?;
    let ocr_ids: Vec<i64> = ocr_hits.iter().map(|(f, _)| *f).collect();
    for (file_id, shot_id) in ocr_hits {
        shot_of.insert(file_id, (shot_id, 1.0));
    }
    let fused = rrf_fuse(&[name_ids, sem_ids, ocr_ids]);
    let mut out = Vec::new();
    for (file_id, score) in fused.into_iter().take(limit) {
        match shot_of.get(&file_id) {
            Some((shot_id, _)) => {
                let scores: std::collections::HashMap<i64, f32> =
                    [(*shot_id, score)].into_iter().collect();
                out.extend(crate::videos::hits_by_shot_ids(conn, &[*shot_id], &scores)?);
            }
            None => {
                if let Some(h) = crate::videos::file_level_hit(conn, file_id, score)? {
                    out.push(h);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_vector_ranking_matches_full_sort() {
        // Includes duplicates, ties, dimensions and empty top-k.
        let all: Vec<(i64, Vec<f32>)> = (0..500)
            .map(|i| (i as i64, vec![(i % 17) as f32, (i % 13) as f32])).collect();
        for k in [0, 1, 3, 10, 100, 1000] {
            let mut full: Vec<(i64, f32)> = all.iter()
                .map(|(id, v)| (*id, dot(&[0.4, 0.6], v)))
                .collect();
            full.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            full.truncate(k);
            assert_eq!(top_similar(&all, &[0.4, 0.6], k), full);
        }
        let repeated = vec![(1, vec![1.0]), (1, vec![3.0]),
            (2, vec![2.0]), (3, vec![3.0]), (3, vec![1.0])];
        assert_eq!(top_similar_grouped(&repeated, &[1.0], 2), vec![(1, 3.0), (3, 3.0)]);
    }

    #[test]
    fn xlsx_highlight_location_is_grounded() {
        let source = "Лист: Продажи\nA3: Цена дивана\nB3: 1000\nЛист: Склад\nC12: Остаток товара 20\n";
        assert_eq!(xlsx_fts_location("…\u{1}Остаток\u{2} товара…", source).as_deref(),
            Some("Лист «Склад», C12"));
        assert!(xlsx_fts_location("…\u{1}Несуществующее\u{2}…", source).is_none());
    }

    #[test]
    fn pdf_snippet_uses_confirmed_pages_not_guesses() {
        let indexed = "[Страница PDF 1]\nДоговор аренды\n[Страница PDF 3]\nСогласована поставка стульев\n";
        assert_eq!(pdf_fts_location("…\u{1}поставка стульев\u{2}…", indexed).as_deref(), Some("Страница 3"));
        assert_eq!(pdf_fts_location("…\u{1}Договор аренды\u{2}…", indexed).as_deref(), Some("Страница 1"));
        assert!(pdf_fts_location("…\u{1}неизвестный текст\u{2}…", indexed).is_none());
        assert!(pdf_fts_location("…\u{1}поставка стульев\u{2}…", "текст без маркера").is_none());
        let duplicate = "[Страница PDF 1]\nСумма договора 200\n[Страница PDF 2]\nСумма договора 300";
        assert!(pdf_fts_location("…\u{1}Сумма договора\u{2}…", duplicate).is_none());
        assert_eq!(structured_passage("поставка стульев", Some("pdf"), indexed).as_deref(),
            Some("Страница 3: Согласована поставка стульев"));
        assert!(structured_passage("нет такого", Some("pdf"), indexed).is_none());
    }

    #[test]
    fn structured_snippet_has_real_sheet_and_cell() {
        let indexed = "Лист: Продажи\nA7: Цена стула\nB7: 5000\nЛист: Склад\nC12: Остаток товара 20\n";
        let passage = structured_passage("остаток товара", Some("xlsx"), indexed).unwrap();
        assert!(passage.starts_with("Лист «Склад», C12:"));
        assert!(passage.contains("Остаток товара"));
        assert!(structured_passage("вакцина", Some("xlsx"), indexed).is_none());
    }


    #[test]
    fn rrf_prefers_items_in_both_lists() {
        // 2 appears in both lists (ranks 1 and 0) → must beat 1 and 3.
        let fused = rrf_fuse(&[vec![1, 2], vec![2, 3]]);
        assert_eq!(fused[0].0, 2);
        let ids: Vec<i64> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 3);
        // deterministic tie-break: 1 (rank 0) vs 3 (rank 1) → 1 scores higher
        assert_eq!(ids, vec![2, 1, 3]);
    }

    #[test]
    fn web_search_finds_mid_token_substrings_and_dedupes_browsers() {
        // repro of issue #1: "sms" must reach "longsms" (FTS tokenizes it as
        // one token, so only the LIKE supplement can), and the same URL from
        // Edge + Edge SxS must collapse to one row
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO bookmarks (url, title, folder, browser, added_at) VALUES
               ('https://longsms.example/app/shop', '选购号码 · longsms', '公网', 'edge', 1),
               ('https://longsms.example/app/shop', '选购号码 · longsms', '公网', 'edge sxs', 1);
             INSERT INTO history (url, title, browser, visit_count, last_visit) VALUES
               ('https://hero-sms.example/cn', '接码平台 – HeroSMS', 'edge', 5, 1),
               ('https://longsms.example/app/shop', 'Buy numbers · longsms', 'edge', 11, 1),
               ('https://longsms.example/app/shop', 'Buy numbers · longsms', 'edge sxs', 11, 1);",
        )
        .unwrap();
        let store = VectorStore::empty();

        // raw FTS really cannot see it (guards the root-cause diagnosis)
        assert!(!crate::db::build_fts_query("sms").is_empty());
        let raw: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM bookmarks_fts WHERE bookmarks_fts MATCH 'sms*'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(raw, 0, "unicode61 FTS matches token prefixes only");

        let hits = search_bookmarks(&conn, &store, "sms", None, 10).unwrap();
        assert_eq!(hits.len(), 1, "substring supplement + cross-browser dedup");
        assert!(hits[0].url.contains("longsms"));

        let hits = search_history(&conn, &store, "sms", None, 10).unwrap();
        assert_eq!(hits.len(), 2, "hero-sms (FTS) + longsms (substring), deduped");
        let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
        assert!(urls.contains(&"https://hero-sms.example/cn"));
        assert!(urls.contains(&"https://longsms.example/app/shop"));

        // LIKE wildcards in user input stay literal
        assert!(search_bookmarks(&conn, &store, "%", None, 10).unwrap().is_empty());
    }

    /// issue #5: one row per URL lists every browser holding it, and a hit
    /// found only by its vector is marked fuzzy.
    #[test]
    fn web_hits_list_all_browsers_and_mark_meaning_only_matches() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO bookmarks (id, url, title, folder, browser, added_at) VALUES
               (1, 'https://www.bilibili.com/', '哔哩哔哩', 'fun', 'librewolf', 1),
               (2, 'https://www.bilibili.com/', 'bilibili', 'video', 'helium', 1),
               (3, 'https://techcrunch.com/', 'TechCrunch', 'news', 'helium', 1);
             INSERT INTO history (id, url, title, browser, visit_count, last_visit) VALUES
               (1, 'https://www.bilibili.com/', 'bilibili', 'librewolf', 9, 1),
               (2, 'https://www.bilibili.com/', 'bilibili', 'chrome', 2, 1),
               (3, 'https://techcrunch.com/', 'TechCrunch', 'helium', 1, 1);",
        )
        .unwrap();
        // every vector points the same way: each bookmark is a "semantic" hit
        let mut store = VectorStore::empty();
        store.bookmarks = (1..=3).map(|id| (id, vec![1.0, 0.0])).collect();
        store.history = (1..=3).map(|id| (id, vec![1.0, 0.0])).collect();
        let q = [1.0, 0.0];

        let hits = search_bookmarks(&conn, &store, "bili", Some(&q), 10).unwrap();
        let bili = hits.iter().find(|h| h.url.contains("bilibili")).unwrap();
        assert_eq!(bili.browsers, vec!["helium", "librewolf"], "one row, both browsers, sorted");
        assert!(!bili.fuzzy, "matched by keyword");
        let tc = hits.iter().find(|h| h.url.contains("techcrunch")).unwrap();
        assert!(tc.fuzzy, "only the vector matched");
        assert_eq!(tc.browsers, vec!["helium"]);
        assert_eq!(hits.len(), 2);

        let hits = search_history(&conn, &store, "bili", Some(&q), 10).unwrap();
        let bili = hits.iter().find(|h| h.url.contains("bilibili")).unwrap();
        assert_eq!(bili.browsers, vec!["chrome", "librewolf"]);
        assert!(!bili.fuzzy);
        assert!(hits.iter().find(|h| h.url.contains("techcrunch")).unwrap().fuzzy);

        // no vector: only keyword hits, none fuzzy
        let hits = search_bookmarks(&conn, &store, "bili", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].fuzzy);
    }

    #[test]
    fn images_scope_ranks_by_ocr_text_semantics() {
        // an image whose only searchable signal is its OCR text's e5 chunk
        // vector must surface in the Images scope (the old code skipped the
        // chunk-vector list there, which predates OCR)
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO folders (path) VALUES ('/p');
             INSERT INTO files (folder_id, path, name, ext, size, mtime, content) VALUES
               (1, '/p/receipt.png', 'receipt.png', 'png', 1, 1, 'refund policy screenshot'),
               (1, '/p/beach.png', 'beach.png', 'png', 1, 1, NULL);",
        )
        .unwrap();
        let store = VectorStore {
            file_chunks: vec![(1, vec![1.0, 0.0])],
            ..VectorStore::empty()
        };
        // query vector aligned with the OCR chunk; query text matches no name
        let hits = search_files(
            &conn,
            &store,
            "退款截图",
            Some(&[1.0, 0.0]),
            None,
            LocalScope::Images,
            10,
        )
        .unwrap();
        assert_eq!(hits.len(), 1, "OCR-text semantic hit must surface in Images scope");
        assert_eq!(hits[0].name, "receipt.png");
    }

    #[test]
    fn videos_scope_fuses_name_and_shot_hits() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO folders (path) VALUES ('/v');
             INSERT INTO files (folder_id, path, name, ext, size, mtime) VALUES
               (1, '/v/sunset-drone.mp4', 'sunset-drone.mp4', 'mp4', 1, 1),
               (1, '/v/longsms-tutorial.mp4', 'longsms-tutorial.mp4', 'mp4', 1, 1),
               (1, '/v/notes.md', 'notes.md', 'md', 1, 1);
             INSERT INTO video_index (file_id, mtime, duration_ms, shot_count) VALUES (1, 1, 9000, 1);",
        )
        .unwrap();
        // one embedded shot for sunset-drone.mp4: vector [1, 0]
        conn.execute(
            "INSERT INTO video_shots (file_id, start_ms, end_ms, ts_ms, thumb, embedding, model)
             VALUES (1, 3000, 6000, 4500, 'x', ?1, 'm')",
            rusqlite::params![[0u8, 0, 128, 63, 0, 0, 0, 0].as_slice()],
        )
        .unwrap();
        let store = VectorStore::load(&conn).unwrap();
        assert_eq!(store.video_shots.len(), 1);

        // filename-only match (mid-token!): whole-file hit, no time range
        let hits = search_videos_scope(&conn, &store, "sms", None, 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "longsms-tutorial.mp4");
        assert_eq!(hits[0].end_ms, 0, "file-level representation");

        // semantic-only match: shot hit carries its range
        let hits = search_videos_scope(&conn, &store, "sunset scene", Some(&[1.0, 0.0]), 10).unwrap();
        assert!(hits.iter().any(|h| h.name == "sunset-drone.mp4" && h.start_ms == 3000 && h.end_ms == 6000));

        // both ways at once: one row, shot representation preferred
        let hits = search_videos_scope(&conn, &store, "sunset", Some(&[1.0, 0.0]), 10).unwrap();
        let drone: Vec<_> = hits.iter().filter(|h| h.name == "sunset-drone.mp4").collect();
        assert_eq!(drone.len(), 1, "no duplicate for name+shot double match");
        assert_eq!(drone[0].start_ms, 3000, "shot representation wins");

        // non-video files never leak into the scope
        assert!(search_videos_scope(&conn, &store, "notes", None, 10).unwrap().is_empty());
    }

    #[test]
    fn videos_scope_matches_shot_ocr_text_and_pins_the_shot() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO folders (path) VALUES ('/v');
             INSERT INTO files (folder_id, path, name, ext, size, mtime) VALUES
               (1, '/v/meeting.mp4', 'meeting.mp4', 'mp4', 1, 1);
             INSERT INTO video_index (file_id, mtime, duration_ms, shot_count) VALUES (1, 1, 60000, 3);
             INSERT INTO video_shots (file_id, start_ms, end_ms, ts_ms, thumb, ocr_text) VALUES
               (1, 0, 10000, 5000, 'x', ''),
               (1, 10000, 20000, 15000, 'x', '第三季度退款政策说明'),
               (1, 20000, 30000, 25000, 'x', '退款政策附录');",
        )
        .unwrap();
        let store = VectorStore::empty();
        // CJK substring of the OCR text, no name match, no vectors: the hit
        // must land on the EARLIEST matching shot's time range
        let hits = search_videos_scope(&conn, &store, "退款政策", None, 10).unwrap();
        assert_eq!(hits.len(), 1, "one row per video");
        assert_eq!(hits[0].start_ms, 10000, "earliest matching shot pinned");
        assert_eq!(hits[0].end_ms, 20000);
        // no OCR match, nothing else to go on: empty
        assert!(search_videos_scope(&conn, &store, "预算", None, 10).unwrap().is_empty());
    }

    #[test]
    fn step29_address_intent_is_conservative_and_exact() {
        let intent = super::AddressIntent::parse("договор аренды Учебная 28").unwrap();
        assert_eq!(intent.fts_expression(),
            "\"договор\"* AND \"аренд\"* AND NEAR(\"учебн\"* \"28\", 4)");
        assert!(intent.has_joint_document_evidence(
            "Договор аренды.docx", "Россия, Примерск, ул. Учебная, д. 28"));
        assert!(intent.has_joint_document_evidence(
            "Договор аренды.docx", "по адресу Учебная 28 — договор аренды"));
        assert!(!intent.has_joint_document_evidence(
            "Договор аренды.docx", "Учебная 280, договор аренды"));
        assert!(!intent.has_joint_document_evidence(
            "Договор аренды.docx", "Учебная 180, комната 28, договор аренды"));
        assert!(!intent.has_joint_document_evidence("Уведомление.pdf", "ул Учебная 28"));
        assert!(super::AddressIntent::parse("договор аренды октябрь 28").is_none());
        assert!(super::AddressIntent::parse("договор аренды 2024").is_none());
        assert!(super::AddressIntent::parse("Учебная 28").is_none());
    }

    #[test]
    fn step29_joint_address_survives_final_reranking() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO folders(path) VALUES ('/q');
             INSERT INTO files(folder_id,path,name,ext,size,mtime,content) VALUES
              (1,'/q/general.docx','Договор аренды.docx','docx',100,1,'Договор аренды других помещений.'),
              (1,'/q/address.pdf','Адрес Учебная 28.pdf','pdf',100,2,'Адрес: улица Учебная, дом 28.'),
              (1,'/q/joint.pdf','Справка.pdf','pdf',100,3,'Договор аренды помещения. Город Примерск, улица Учебная, д. 28.'),
              (1,'/q/false.pdf','Договор аренды.pdf','pdf',100,4,'Учебная 180, кабинет 28.');"
        ).unwrap();
        let store = super::VectorStore::empty();
        let mut hits = super::search_files(&conn, &store, "найди договор аренды Учебная 28",
            None, None, super::LocalScope::All, 10).unwrap();
        assert_eq!(hits[0].path, "/q/joint.pdf", "complete evidence must lead");
        assert_eq!(hits[0].match_kind.as_deref(), Some("Тема + адрес"));
        assert!(hits.iter().any(|h| h.path == "/q/general.docx"), "keep partial topic hits");
        assert!(hits.iter().any(|h| h.path == "/q/address.pdf"), "keep partial address hits");
        hits[1].score = 1000.0; // emulate frecency
        super::sort_joint_address_first(&mut hits);
        assert_eq!(hits[0].path, "/q/joint.pdf", "frecency cannot defeat complete evidence");
        assert!(!hits.iter().any(|h| h.path == "/q/false.pdf" &&
            h.match_kind.as_deref() == Some("Тема + адрес")), "number proximity is required");
    }

    #[test]
    fn step27_voice_query_cleanup_is_conservative() {
        assert_eq!(normalize_local_query_text("Найди договор аренды"), "договор аренды");
        assert_eq!(normalize_local_query_text("Покажи мне документ договор аренды"), "договор аренды");
        assert_eq!(normalize_local_query_text("договор аренды"), "договор аренды");
        assert_eq!(normalize_local_query_text("найди"), "найди");
        assert_eq!(normalize_local_query_text("найдидоговор.txt"), "найдидоговор.txt");
        assert_eq!(normalize_local_query_text("Найди договор аренды на улице Учебная двадцать восемь."),
                   "договор аренды на улице Учебная 28");
        assert_eq!(normalize_local_query_text("двадцать восемь панфиловцев"),
                   "двадцать восемь панфиловцев");
    }

    #[test]
    fn step40_spoken_format_is_unambiguous() {
        let fmt = super::spoken_format("Excel с остатками").unwrap();
        assert_eq!(fmt.exts, &["xlsx", "xls"]);
        assert_eq!(fmt.text, "с остатками");
        assert!(super::spoken_format("Отчет.xlsx о продажах").is_none());
        assert!(super::spoken_format("Word или PDF").is_none());
        assert_eq!(super::format_topic_expression("с остатками").unwrap(), "\"остатк\"*");
        assert_eq!(super::filename_evidence("акт осмотра заказа", "Акт осмотра 123.docx") > 0.0, true);
        assert_eq!(super::filename_evidence("акт договора", "Договор.docx"), 0.0);
    }

    #[test]
    fn step40_excel_search_enforces_extension_and_russian_inflection() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute_batch(
            "INSERT INTO folders(path) VALUES ('/q');
             INSERT INTO files(folder_id,path,name,ext,size,mtime,content) VALUES
              (1,'/q/text.pdf','Остатками.pdf','pdf',100,1,'Список остатков и наличие'),
              (1,'/q/stock.xlsx','Склад.xlsx','xlsx',100,2,'Остатки на складе: 10 стульев'),
              (1,'/q/stock.json','Остатки.json','json',100,3,'Остатки товаров');"
        ).unwrap();
        let store = super::VectorStore::empty();
        let hits = super::search_files(&conn, &store, "найди Excel с остатками",
            None, None, super::LocalScope::All, 10).unwrap();
        assert_eq!(hits.len(), 1, "only requested format can be returned");
        assert_eq!(hits[0].path, "/q/stock.xlsx");
        let no_filter = super::search_files(&conn, &store, "Остатки на складе",
            None, None, super::LocalScope::All, 10).unwrap();
        assert!(no_filter.iter().any(|h| h.path == "/q/stock.xlsx"));
    }

    #[test]
    fn step40_title_rescue_works_when_content_competes() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute("INSERT INTO folders(path) VALUES ('/docs')", []).unwrap();
        for i in 0..75 {
            conn.execute(
                "INSERT INTO files(folder_id,path,name,ext,size,mtime,content)
                 VALUES(1,?1,?2,'pdf',10,1,'добрин пушкина')",
                rusqlite::params![format!("/docs/{i}.pdf"), format!("Документ {i}.pdf")],
            ).unwrap();
        }
        conn.execute(
            "INSERT INTO files(folder_id,path,name,ext,size,mtime,content)
             VALUES(1,'/docs/target.xlsx','Добрин Пушкина.xlsx','xlsx',10,1,'прочий текст')",
            [],
        ).unwrap();
        let hits = super::filename_fts_candidates(&conn, "добрин Пушкина", 50).unwrap();
        assert_eq!(hits.len(), 1);
        let name: String = conn.query_row("SELECT name FROM files WHERE id=?1", [hits[0]], |r| r.get(0)).unwrap();
        assert_eq!(name, "Добрин Пушкина.xlsx");
    }

    #[test]
    fn step40_document_boost_does_not_override_image_requests() {
        assert!(super::document_intent("найди договор поставки"));
        assert!(!super::document_intent("найди скан договора"));
        assert!(!super::document_intent("найди фото документа"));
        assert!(super::document_priority(Some("docx")) > super::document_priority(Some("jpg")));
        assert_eq!(super::document_priority(Some("py")), 0.0);
    }

    #[test]
    fn rrf_empty() {
        assert!(rrf_fuse(&[]).is_empty());
        assert!(rrf_fuse(&[vec![], vec![]]).is_empty());
    }

    #[test]
    fn vector_search_orders_by_similarity() {
        let conn = crate::db::open_in_memory().unwrap();
        for (id, name) in [(1, "a/a"), (2, "b/b"), (3, "c/c")] {
            crate::db::upsert_repo(
                &conn,
                &crate::db::Repo {
                    id,
                    full_name: name.into(),
                    description: None,
                    language: None,
                    topics: vec![],
                    stars: 0,
                    html_url: String::new(),
                    homepage: None,
                    archived: false,
                    fork: false,
                    starred_at: None,
                    pushed_at: None,
                },
            )
            .unwrap();
        }
        crate::db::put_repo_chunks(&conn, 1, "h", &[vec![1.0, 0.0]]).unwrap();
        crate::db::put_repo_chunks(&conn, 2, "h", &[vec![0.0, 1.0]]).unwrap();
        crate::db::put_repo_chunks(&conn, 3, "h", &[vec![0.7, 0.7]]).unwrap();
        let store = VectorStore::load(&conn).unwrap();
        let hits: Vec<i64> = top_similar(&store.repos, &[1.0, 0.0], 3)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(hits, vec![1, 3, 2]);
        // dimension-mismatched query vectors match nothing rather than panicking
        assert!(top_similar(&store.repos, &[1.0, 0.0, 0.0], 3).is_empty());

        // grouped: a file scores as its best chunk
        let chunks = vec![
            (10i64, vec![0.2, 0.0]),
            (10i64, vec![0.9, 0.0]),
            (11i64, vec![0.5, 0.0]),
        ];
        let grouped = top_similar_grouped(&chunks, &[1.0, 0.0], 5);
        assert_eq!(grouped[0], (10, 0.9));
        assert_eq!(grouped[1], (11, 0.5));
    }

    #[test]
    fn vector_input_changes_hybrid_ranking() {
        let conn = crate::db::open_in_memory().unwrap();
        for (id, name, desc) in [(1, "a/cli", "terminal tool"), (2, "b/gui", "desktop tool")] {
            crate::db::upsert_repo(
                &conn,
                &crate::db::Repo {
                    id,
                    full_name: name.into(),
                    description: Some(desc.into()),
                    language: None,
                    topics: vec![],
                    stars: 0,
                    html_url: String::new(),
                    homepage: None,
                    archived: false,
                    fork: false,
                    starred_at: Some(format!("2026-01-0{id}T00:00:00Z")),
                    pushed_at: None,
                },
            )
            .unwrap();
        }
        crate::db::put_repo_chunks(&conn, 1, "h", &[vec![1.0, 0.0]]).unwrap();
        crate::db::put_repo_chunks(&conn, 2, "h", &[vec![0.0, 1.0]]).unwrap();

        let store = VectorStore::load(&conn).unwrap();
        // both match FTS equally on "tool"; the query vector must break the tie
        let towards_1 =
            search(&conn, &store, "tool", Some(&[1.0, 0.0]), RepoSort::Relevance, 10).unwrap();
        let towards_2 =
            search(&conn, &store, "tool", Some(&[0.0, 1.0]), RepoSort::Relevance, 10).unwrap();
        assert_eq!(towards_1.len(), 2);
        assert_eq!(towards_1[0].repo.id, 1, "qvec [1,0] must rank repo 1 first");
        assert_eq!(towards_2[0].repo.id, 2, "qvec [0,1] must rank repo 2 first");
        // scores align with the repo they belong to (regression: zip misalignment)
        assert!(towards_1[0].score > towards_1[1].score);

        // FTS-only fallback still returns both
        let fts_only = search(&conn, &store, "tool", None, RepoSort::Relevance, 10).unwrap();
        assert_eq!(fts_only.len(), 2);

        // alternate sorts reorder matches: repo 2 starred later, repo 1 has more stars
        conn.execute("UPDATE repos SET stars = 500 WHERE id = 1", []).unwrap();
        let by_starred = search(&conn, &store, "tool", None, RepoSort::Starred, 10).unwrap();
        assert_eq!(by_starred[0].repo.id, 2, "starred_at 2026-01-02 wins");
        let by_stars = search(&conn, &store, "tool", None, RepoSort::Stars, 10).unwrap();
        assert_eq!(by_stars[0].repo.id, 1, "500 stars wins");
    }

    #[test]
    fn empty_query_returns_recent() {
        let conn = crate::db::open_in_memory().unwrap();
        let store = VectorStore::empty();
        let results = search(&conn, &store, "   ", None, RepoSort::Relevance, 10).unwrap();
        assert!(results.is_empty());
    }
}
