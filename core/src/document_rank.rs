//! Document-level topicality booster for Magpie's hybrid retrieval.
//! This is deliberately a bounded, transparent heuristic — NOT an LLM reranker.
//! Score changes are tuned only after a real labelled archive test.
use std::collections::HashSet;

const STOP: &[&str] = &[
    "найди", "найти", "покажи", "документ", "документы", "файл", "файлы", "где",
    "котором", "который", "которая", "которые", "обсуждалась", "обсуждали",
    "обсуждается", "про", "по", "что", "это", "этот", "этом", "для", "или",
    "как", "был", "была", "были", "есть", "мне", "нужно", "содержится",
    "в", "на", "о", "об", "от", "до", "и", "с", "из", "к", "у", "а",
];

/// Does a cheap *approximate* normalization of Russian noun/adjective endings.
/// This is not a full Russian stemmer and must be assessed by the relevance suite.
fn normalize_word(word: &str) -> String {
    let w = word.to_lowercase().replace('ё', "е");
    // Order is important: longest suffix first. Stopwords are filtered before stemming.
    const ENDINGS: &[&str] = &[
        "иями", "ями", "ами", "ения", "ению", "ениям", "ениях", "ование", "овать",
        "ого", "ему", "ыми", "ими", "ами", "иях", "иях", "овую", "овой",
        "ами", "ях", "ах", "ям", "ам", "ом", "ем", "ов", "ев", "ой",
        "ей", "ый", "ий", "ая", "ое", "ые", "ых", "их", "ую", "ию",
        "а", "я", "ы", "и", "е", "у", "ю", "о",
    ];
    for suffix in ENDINGS {
        if w.ends_with(suffix) && w.chars().count() > suffix.chars().count() + 3 {
            return w[..w.len() - suffix.len()].to_string();
        }
    }
    w
}

fn tokens(input: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut word = String::new();
    for c in input.chars().chain(std::iter::once(' ')) {
        if c.is_alphanumeric() {
            word.extend(c.to_lowercase());
        } else if !word.is_empty() {
            if !STOP.contains(&word.as_str()) && word.chars().count() >= 3 {
                result.push(normalize_word(&word));
            }
            word.clear();
        }
    }
    result
}

fn terms(input: &str) -> HashSet<String> {
    tokens(input).into_iter().collect()
}

fn coverage(q: &HashSet<String>, t: &HashSet<String>) -> f32 {
    if q.is_empty() { return 0.0; }
    q.iter().filter(|w| t.contains(*w)).count() as f32 / q.len() as f32
}

/// Add at most ~0.027 to Magpie's RRF score (~0.01–0.04 typical).
/// Body sample is limited to 64K characters by the caller, to cap query costs.
/// Boost balances filename/lead topic against sustained evidence in many paragraphs.
pub fn topic_boost(query: &str, filename: &str, body_sample: &str) -> f32 {
    let q = terms(query);
    if q.is_empty() { return 0.0; }

    let filename = filename.rsplit_once('.').map(|(s, _)| s).unwrap_or(filename);
    let title_cov = coverage(&q, &terms(filename));
    let lead: String = body_sample.chars().take(1800).collect();
    let lead_cov = coverage(&q, &terms(&lead));

    // Use at most 100 non-empty logical passages in the bounded sample.
    // A document with one relevant aside among many paragraphs gets a smaller bonus.
    let passages: Vec<_> = body_sample
        .split('\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .take(100)
        .collect();
    let sufficient = ((q.len() * 2 + 2) / 3).max(1);
    let relevant = passages.iter().filter(|p| {
        let pterms = terms(p);
        q.iter().filter(|t| pterms.contains(*t)).count() >= sufficient
    }).count();
    let focus = if passages.is_empty() { 0.0 } else {
        relevant as f32 / passages.len() as f32
    };
    let body_cov = coverage(&q, &terms(body_sample));

    0.010 * title_cov + 0.005 * lead_cov + 0.010 * focus + 0.002 * body_cov
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_forms_normalize() {
        assert_eq!(normalize_word("редукторе"), "редуктор");
        assert_eq!(normalize_word("редуктора"), "редуктор");
        assert_eq!(normalize_word("масла"), normalize_word("масло"));
    }

    #[test]
    fn focused_document_outweighs_incidental_mention() {
        let q = "Найди документ, где обсуждалась замена масла в редукторе";
        let primary = "Регламент замены масла в редукторе";
        let main_body = (0..12).map(|_| "Замена масла в редукторе: порядок обслуживания и контроль уровня.").collect::<Vec<_>>().join("\n");
        let note = (0..60).map(|i| if i == 29 { "Разово упомянули замену масла в редукторе." } else { "Обсуждали даты, билеты и гостиницу." }).collect::<Vec<_>>().join("\n");
        assert!(topic_boost(q, primary, &main_body) > topic_boost(q, "Путевые заметки.txt", &note));
    }

    #[test]
    fn empty_or_stopword_only_query_has_no_boost() {
        assert_eq!(topic_boost("где и в", "редуктор", "масло"), 0.0);
    }

    #[test]
    fn title_weight_is_positive() {
        let q = "инструкция сборка стула";
        assert!(topic_boost(q, "Инструкция сборки стула.pdf", "") > topic_boost(q, "note.pdf", ""));
    }
}


/// Query-independent terms of the same 64K preview used by `topic_boost`.
/// Cache entries keep the exact source text so updates/OCR cannot serve stale scores.
#[derive(Debug)]
pub struct TopicProfile {
    title: HashSet<String>,
    lead: HashSet<String>,
    passages: Vec<HashSet<String>>,
    body: HashSet<String>,
}

impl TopicProfile {
    pub fn new(filename: &str, body_sample: &str) -> Self {
        let stem = filename.rsplit_once('.').map(|(s, _)| s).unwrap_or(filename);
        let title = terms(stem);
        let lead: String = body_sample.chars().take(1800).collect();
        let lead = terms(&lead);
        let passages = body_sample
            .split('\n')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .take(100)
            .map(terms)
            .collect();
        let body = terms(body_sample);
        Self { title, lead, passages, body }
    }
}

/// Query is parsed once for all candidates, rather than 100 times.
pub struct PreparedTopicQuery {
    query_terms: HashSet<String>,
    sufficient: usize,
}

impl PreparedTopicQuery {
    pub fn new(query: &str) -> Self {
        let query_terms = terms(query);
        let sufficient = ((query_terms.len() * 2 + 2) / 3).max(1);
        Self { query_terms, sufficient }
    }
    pub fn is_empty(&self) -> bool { self.query_terms.is_empty() }
    pub fn boost(&self, profile: &TopicProfile) -> f32 {
        let q = &self.query_terms;
        if q.is_empty() { return 0.0; }
        let title_cov = coverage(q, &profile.title);
        let lead_cov = coverage(q, &profile.lead);
        let relevant = profile.passages.iter().filter(|pterms| {
            q.iter().filter(|term| pterms.contains(*term)).count() >= self.sufficient
        }).count();
        let focus = if profile.passages.is_empty() { 0.0 } else {
            relevant as f32 / profile.passages.len() as f32
        };
        let body_cov = coverage(q, &profile.body);
        0.010 * title_cov + 0.005 * lead_cov + 0.010 * focus + 0.002 * body_cov
    }
}

const TOPIC_PROFILE_CACHE_LIMIT: usize = 128;
struct ProfileEntry {
    id: i64,
    name: String,
    sample: String,
    profile: std::sync::Arc<TopicProfile>,
}

/// Verified-by-value small in-memory LRU; no persistent state and no stale file versions.
/// Holding at most 128 previews bounds memory; misses compute exactly the original score.
pub fn cached_topic_profile(
    id: i64, filename: &str, body_sample: &str,
) -> (std::sync::Arc<TopicProfile>, bool) {
    use std::collections::VecDeque;
    use std::sync::{Mutex, OnceLock, Arc};
    static CACHE: OnceLock<Mutex<VecDeque<ProfileEntry>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    {
        let mut entries = cache.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(pos) = entries.iter().position(|e| {
            e.id == id && e.name == filename && e.sample == body_sample
        }) {
            let entry = entries.remove(pos).expect("known cache position");
            let profile = Arc::clone(&entry.profile);
            entries.push_back(entry);
            return (profile, true);
        }
    }
    let profile = Arc::new(TopicProfile::new(filename, body_sample));
    let mut entries = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    entries.retain(|entry| entry.id != id); // old file revisions never retained
    entries.push_back(ProfileEntry {
        id, name: filename.to_string(), sample: body_sample.to_string(),
        profile: Arc::clone(&profile),
    });
    while entries.len() > TOPIC_PROFILE_CACHE_LIMIT { entries.pop_front(); }
    (profile, false)
}

#[cfg(test)]
mod topic_profile_equivalence_tests {
    use super::*;
    #[test]
    fn prepared_profile_is_identical_to_existing_reranker() {
        let long_body = "Продукт Орехи EXAMPLE\n".repeat(300);
        let bodies = [
            "", "Оформление договора поставки мебели.\nЦена 15000 руб.",
            "срок поставки 15 дней\n\nдоговор стульев и столов",
            long_body.as_str(),
        ];
        for query in [
            "где и в", "договор поставки мебели", "масло редуктора",
            "СТУЛ цена", "Example орехи 15", "квартире", "ОФОРМЛЕНИЕ",
        ] {
            let prepared = PreparedTopicQuery::new(query);
            for filename in ["Договор 2026.docx", "Поездка Минск.xlsx", "notes.txt"] {
                for body in bodies {
                    let original = topic_boost(query, filename, body);
                    let fresh = prepared.boost(&TopicProfile::new(filename, body));
                    assert_eq!(original.to_bits(), fresh.to_bits(), "{query:?} {filename:?}");
                }
            }
        }
    }
    #[test]
    fn cache_invalidates_exactly_when_content_changes() {
        let (a, hit) = cached_topic_profile(1257, "memo.txt", "старый текст");
        assert!(!hit);
        let (b, hit) = cached_topic_profile(1257, "memo.txt", "старый текст");
        assert!(hit);
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        let (c, hit) = cached_topic_profile(1257, "memo.txt", "новый текст");
        assert!(!hit);
        assert!(!std::sync::Arc::ptr_eq(&b, &c));
    }
}
