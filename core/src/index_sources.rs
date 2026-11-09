//! Read-only inventory of files already present in the local index.
//! No filesystem walk, schema migration, writes or changes to embeddings.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rusqlite::Connection;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct SourceCounts {
    pub files: u64,
    pub with_text: u64,
    pub with_vectors: u64,
}

impl SourceCounts {
    fn add(&mut self, has_text: bool, has_vectors: bool) {
        self.files += 1;
        self.with_text += u64::from(has_text);
        self.with_vectors += u64::from(has_vectors);
    }
    fn merge(&mut self, other: Self) {
        self.files += other.files;
        self.with_text += other.with_text;
        self.with_vectors += other.with_vectors;
    }
}

#[derive(Debug, Serialize)]
pub struct RegisteredFolder {
    pub id: i64,
    pub path: String,
    pub available: bool,
    pub files: u64,
}

#[derive(Debug, Serialize)]
pub struct SourceNode {
    pub name: String,
    pub path: String,
    pub counts: SourceCounts,
    pub direct_files: u64,
    pub children: Vec<SourceNode>,
}

#[derive(Debug, Serialize)]
pub struct SourceGroup {
    pub id: &'static str,
    pub counts: SourceCounts,
    pub roots: Vec<SourceNode>,
}

#[derive(Debug, Serialize)]
pub struct IndexSourcesReport {
    pub counts: SourceCounts,
    pub registered: Vec<RegisteredFolder>,
    pub groups: Vec<SourceGroup>,
    /// Unexpected file records, if their recorded folder no longer matches the path.
    pub inconsistent_paths: u64,
}

#[derive(Default)]
struct Branch {
    counts: SourceCounts,
    direct_files: u64,
    children: BTreeMap<String, Branch>,
}
impl Branch {
    fn insert(&mut self, parts: &[String], has_text: bool, has_vectors: bool) {
        self.counts.add(has_text, has_vectors);
        if parts.is_empty() {
            self.direct_files += 1;
        } else {
            self.children.entry(parts[0].clone()).or_default().insert(&parts[1..], has_text, has_vectors);
        }
    }
    fn into_node(self, name: String, path: PathBuf) -> SourceNode {
        let children = self.children.into_iter().map(|(child_name, child)| {
            let child_path = path.join(&child_name);
            child.into_node(child_name, child_path)
        }).collect();
        SourceNode { name, path: path.to_string_lossy().into_owned(), counts: self.counts, direct_files: self.direct_files, children }
    }
}

const GROUP_IDS: [&str; 7] = ["local", "icloud", "dropbox", "google", "yandex", "other_cloud", "external"];

// Detection by well-known *anchored* macOS mount/sync paths, not by an
// arbitrary folder called "Dropbox" somewhere in Documents. Classification
// describes the path only; it does not claim the account is signed in or synced.
fn source_for(path: &Path, registered_root: &Path, home: &Path) -> (&'static str, PathBuf) {
    if let Ok(relative) = path.strip_prefix(home) {
        let parts: Vec<_> = relative.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
        if parts.len() >= 4 && parts[0] == "Library" && parts[1] == "CloudStorage" {
            let provider = parts[2].to_lowercase();
            let kind = if provider.starts_with("googledrive") || provider.starts_with("google drive") {
                "google"
            } else if provider.starts_with("dropbox") {
                "dropbox"
            } else if provider.starts_with("yandex") || provider.starts_with("яндекс") {
                "yandex"
            } else if provider.starts_with("icloud") {
                "icloud"
            } else { "other_cloud" };
            return (kind, home.join("Library").join("CloudStorage").join(&parts[2]));
        }
        if parts.len() >= 4 && parts[0] == "Library" && parts[1] == "Mobile Documents" {
            return ("icloud", home.join("Library").join("Mobile Documents").join(&parts[2]));
        }
        // Historical provider locations: only direct children of $HOME.
        if parts.len() >= 2 {
            let first = parts[0].to_lowercase();
            let provider = if first == "dropbox" { Some("dropbox") }
                else if first == "google drive" || first == "googledrive" { Some("google") }
                else if first == "yandex.disk" || first == "yandexdisk" || first == "яндекс.диск" { Some("yandex") }
                else { None };
            if let Some(provider) = provider { return (provider, home.join(&parts[0])); }
        }
    }
    // External volumes are distinguished from the internal local disk.
    if let Ok(relative) = path.strip_prefix("/Volumes") {
        let mut comps = relative.components();
        if let Some(disk) = comps.next() {
            if comps.next().is_some() {
                let mount = disk.as_os_str().to_string_lossy().to_lowercase();
                let kind = if mount.starts_with("google drive") || mount.starts_with("googledrive") { "google" }
                    else if mount.starts_with("dropbox") { "dropbox" }
                    else if mount.starts_with("yandex") || mount.starts_with("яндекс") { "yandex" }
                    else { "external" };
                return (kind, Path::new("/Volumes").join(disk.as_os_str()));
            }
        }
    }
    ("local", registered_root.to_path_buf())
}

pub fn report(conn: &Connection, home: &Path) -> Result<IndexSourcesReport> {
    let mut registered = Vec::<RegisteredFolder>::new();
    let mut roots_by_id = BTreeMap::<i64, PathBuf>::new();
    {
        let mut stmt = conn.prepare("SELECT id, path FROM folders ORDER BY path")?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))?;
        for item in rows {
            let (id, path) = item?;
            roots_by_id.insert(id, PathBuf::from(&path));
            registered.push(RegisteredFolder { id, available: Path::new(&path).is_dir(), path, files: 0 });
        }
    }
    let mut folder_positions = BTreeMap::new();
    for (i, f) in registered.iter().enumerate() { folder_positions.insert(f.id, i); }

    let mut tree = BTreeMap::<&'static str, BTreeMap<PathBuf, Branch>>::new();
    let mut counts = SourceCounts::default();
    let mut inconsistent_paths = 0;
    // The EXISTS uses file_chunks(file_id, chunk_idx) primary key; no blobs are read.
    let mut stmt = conn.prepare(
        "SELECT f.folder_id, f.path, (f.content IS NOT NULL AND f.content != ''),\n                EXISTS(SELECT 1 FROM file_chunks fc WHERE fc.file_id = f.id)\n         FROM files f ORDER BY f.folder_id, f.path"
    )?;
    let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, bool>(2)?, row.get::<_, bool>(3)?)))?;
    for item in rows {
        let (folder_id, path, has_text, has_vectors) = item?;
        let Some(root) = roots_by_id.get(&folder_id) else { inconsistent_paths += 1; continue; };
        let file_path = Path::new(&path);
        if !file_path.starts_with(root) || file_path.file_name().is_none() {
            inconsistent_paths += 1;
            continue;
        }
        counts.add(has_text, has_vectors);
        if let Some(&i) = folder_positions.get(&folder_id) { registered[i].files += 1; }
        let (kind, anchor) = source_for(file_path, root, home);
        let parent = file_path.parent().unwrap_or(file_path);
        let Ok(relative) = parent.strip_prefix(&anchor) else {
            inconsistent_paths += 1;
            continue;
        };
        let parts: Vec<_> = relative.components().map(|part| part.as_os_str().to_string_lossy().to_string()).collect();
        tree.entry(kind).or_default().entry(anchor).or_default().insert(&parts, has_text, has_vectors);
    }
    let groups = GROUP_IDS.iter().map(|&id| {
        let mut group_counts = SourceCounts::default();
        let roots = tree.remove(id).unwrap_or_default().into_iter().map(|(root, branch)| {
            group_counts.merge(branch.counts);
            let name = root.file_name().unwrap_or(root.as_os_str()).to_string_lossy().into_owned();
            branch.into_node(name, root)
        }).collect();
        SourceGroup { id, counts: group_counts, roots }
    }).collect();
    Ok(IndexSourcesReport { counts, registered, groups, inconsistent_paths })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchored_cloud_classification_avoids_false_positives() {
        let home = Path::new("/Users/test");
        let root = home;
        assert_eq!(source_for(Path::new("/Users/test/Documents/Dropbox notes/a.pdf"), root, home).0, "local");
        assert_eq!(source_for(Path::new("/Users/test/Library/CloudStorage/GoogleDrive-account/My Drive/a.pdf"), root, home).0, "google");
        assert_eq!(source_for(Path::new("/Users/test/Library/Mobile Documents/com~apple~CloudDocs/one.pdf"), root, home).0, "icloud");
        assert_eq!(source_for(Path::new("/Users/test/Dropbox/a.pdf"), root, home).0, "dropbox");
        assert_eq!(source_for(Path::new("/Users/test/Yandex.Disk/a.pdf"), root, home).0, "yandex");
        assert_eq!(source_for(Path::new("/Volumes/Archive/docs/a.pdf"), root, home).0, "external");
    }

    #[test]
    fn read_only_tree_counts_vectors_and_empty_roots() {
        let conn = crate::db::open_in_memory().unwrap();
        conn.execute("INSERT INTO folders(id,path) VALUES (1,'/Users/test'), (2,'/nowhere/no-index')", []).unwrap();
        conn.execute("INSERT INTO files(id,folder_id,path,name,size,mtime,content) VALUES\n            (1,1,'/Users/test/Documents/Документы/договор.pdf','договор.pdf',1,1,'запись'),\n            (2,1,'/Users/test/Library/CloudStorage/GoogleDrive-account/My Drive/table.xlsx','table.xlsx',1,1,'таблица'),\n            (3,1,'/Users/test/Dropbox/photo.png','photo.png',1,1,NULL)", []).unwrap();
        conn.execute("INSERT INTO file_chunks(file_id,chunk_idx,doc_hash,dim,vec) VALUES (1,0,'h',1,x'01020304')", []).unwrap();
        let report = report(&conn, Path::new("/Users/test")).unwrap();
        assert_eq!(report.counts, SourceCounts { files: 3, with_text: 2, with_vectors: 1 });
        assert_eq!(report.registered[1].files, 0);
        assert_eq!(report.inconsistent_paths, 0);
        assert_eq!(report.groups.iter().find(|g| g.id == "local").unwrap().counts.files, 1);
        assert_eq!(report.groups.iter().find(|g| g.id == "google").unwrap().counts.files, 1);
        assert_eq!(report.groups.iter().find(|g| g.id == "dropbox").unwrap().counts.files, 1);
        let local = report.groups.iter().find(|g| g.id == "local").unwrap();
        let doc = &local.roots[0].children[0].children[0];
        assert_eq!(doc.name, "Документы");
        assert_eq!(doc.counts.files, 1);
    }
}
