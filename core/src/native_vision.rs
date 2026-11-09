//! Small, opt-in macOS Vision OCR bridge for local-search.
//! No model, network, or personal folder scan is required by this module.
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::fs::OpenOptions;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct VisionBatch {
    pub attempted: usize,
    pub indexed: usize,
    pub failed: usize,
}

/// Run one bounded OCR subprocess. No shell expansion; paths may contain spaces.
pub fn recognize(helper: &Path, image: &Path, timeout: Duration) -> Result<String> {
    if !helper.is_file() {
        bail!("Vision helper unavailable: {}", helper.display());
    }
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let capture = std::env::temp_dir().join(format!(
        "local-search-ocr-{}-{unique}.txt", std::process::id()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&capture)?;
    let launched = Command::new(helper)
        .arg("--ocr")
        .arg(image)
        .stdout(Stdio::from(file))
        .stderr(Stdio::null())
        .spawn();
    let mut child = match launched {
        Ok(child) => child,
        Err(e) => {
            let _ = std::fs::remove_file(&capture);
            return Err(e).with_context(|| format!("start Vision helper {}", helper.display()));
        }
    };
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&capture);
                    bail!("Vision OCR exceeded {} s for {}", timeout.as_secs(), image.display());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&capture);
                return Err(e.into());
            }
        }
    };
    let bytes = std::fs::read(&capture);
    let _ = std::fs::remove_file(&capture);
    if !status.success() {
        bail!("Vision OCR failed for {} (status: {})", image.display(), status);
    }
    let bytes = bytes?;
    if bytes.len() > 512 * 1024 {
        bail!("Vision OCR output too large for {}", image.display());
    }
    Ok(String::from_utf8(bytes)?.trim().to_owned())
}

/// Process *at most* `limit` pending image rows, retaining the original file.
/// Errors are counted but not marked successful, allowing controlled retries.
/// This function does not run automatically during the filesystem walk.
pub fn index_pending(conn: &Connection, helper: &Path, limit: usize) -> Result<VisionBatch> {
    if limit == 0 {
        return Ok(VisionBatch::default());
    }
    let pending: Vec<(i64, String, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT f.id, f.path, f.mtime FROM files f
             LEFT JOIN file_ocr_failures e ON e.file_id=f.id AND e.mtime=f.mtime
             WHERE lower(f.ext) IN ('png','jpg','jpeg','heic','heif','tif','tiff','webp','bmp','gif')
               AND (f.ocr_mtime IS NULL OR f.ocr_mtime != f.mtime)
               AND (e.file_id IS NULL OR e.retry_at <= CAST(strftime('%s','now') AS INTEGER))
             ORDER BY f.id LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut report = VisionBatch::default();
    for (id, filename, mtime) in pending {
        report.attempted += 1;
        let path = Path::new(&filename);
        let result = recognize(helper, path, Duration::from_secs(20));
        match result {
            Ok(text) if !text.is_empty() => {
                // Check file metadata after OCR: do not index a stale version.
                let current = std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64);
                if current != Some(mtime) {
                    report.failed += 1;
                    continue;
                }
                let changed = conn.execute(
                    "UPDATE files SET content = ?2, ocr_mtime = ?3
                     WHERE id = ?1 AND mtime = ?3",
                    params![id, text, mtime]
                )?;
                report.indexed += changed;
            }
            Ok(_) => {
                // No legible text: remember the attempt until the image changes.
                conn.execute(
                    "UPDATE files SET ocr_mtime = ?2 WHERE id = ?1 AND mtime = ?2",
                    params![id, mtime]
                )?;
            }
            Err(e) => {
                log::warn!("native Vision OCR: {}: {e:#}", path.display());
                // Retry in an hour, or immediately after the image changes.
                // Unlike marking ocr_mtime, this never permanently loses OCR.
                conn.execute(
                    "INSERT INTO file_ocr_failures(file_id,mtime,attempts,retry_at)
                     VALUES(?1,?2,1,CAST(strftime('%s','now') AS INTEGER)+3600)
                     ON CONFLICT(file_id) DO UPDATE SET
                       attempts=CASE WHEN file_ocr_failures.mtime=excluded.mtime
                                     THEN file_ocr_failures.attempts+1 ELSE 1 END,
                       mtime=excluded.mtime, retry_at=excluded.retry_at",
                    params![id,mtime],
                )?;
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{db, files};
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn native_vision_indexes_and_deduplicates_russian_ocr() {
        let home = std::env::temp_dir().join(format!(
            "local-search-vision-test-{}", std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        // Keep the OCR mock outside the indexed root: otherwise its script
        // body is indexed as plain text and matches before OCR even runs.
        let indexed_folder = home.join("indexed");
        std::fs::create_dir_all(&indexed_folder).unwrap();
        let file = indexed_folder.join("sample.heic");
        std::fs::write(&file, "fixture-not-decoded-by-test").unwrap();
        let helper = home.join("mock-ocr.sh");
        std::fs::write(&helper, "#!/bin/sh\nprintf 'Замена масла в редукторе\\n'\n").unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o700)).unwrap();
        let conn = db::open_in_memory().unwrap();
        files::add_folder(&conn, indexed_folder.to_str().unwrap()).unwrap();
        files::index_folders(&conn, |_| {}).unwrap();
        assert!(files::files_fts_search(&conn, "редукторе", 10).unwrap().is_empty());
        let first = index_pending(&conn, &helper, 1).unwrap();
        assert_eq!(first, VisionBatch { attempted: 1, indexed: 1, failed: 0 });
        assert_eq!(files::files_fts_search(&conn, "редукторе", 10).unwrap().len(), 1);
        assert_eq!(index_pending(&conn, &helper, 1).unwrap(), VisionBatch::default());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_helper_does_not_mark_file_processed() {
        let home = std::env::temp_dir().join(format!(
            "local-search-vision-fail-{}", std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(home.join("photo.png"), "not-real-png").unwrap();
        let conn = db::open_in_memory().unwrap();
        files::add_folder(&conn, home.to_str().unwrap()).unwrap();
        files::index_folders(&conn, |_| {}).unwrap();
        let result = index_pending(&conn, &home.join("missing"), 1).unwrap();
        assert_eq!(result, VisionBatch { attempted: 1, indexed: 0, failed: 1 });
        // Failure waits instead of monopolizing the queue on every watcher tick.
        assert_eq!(index_pending(&conn, &home.join("missing"), 1).unwrap().attempted, 0);
        // A due retry becomes eligible again.
        conn.execute("UPDATE file_ocr_failures SET retry_at=0", []).unwrap();
        assert_eq!(index_pending(&conn, &home.join("missing"), 1).unwrap().attempted, 1);
        let _ = std::fs::remove_dir_all(&home);
    }
}
