//! On-demand Russian OCR for image-only PDFs using macOS Vision PDFKit helper.
//! Keeps source PDFs untouched. Errors leave items pending, not silently indexed.
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};
use std::fs::OpenOptions;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Default, PartialEq, Eq)]
pub struct PdfVisionBatch {
    pub attempted: usize,
    pub indexed: usize,
    pub failed: usize,
}

/// Spawn one OCR job, without a shell and with bounded time and output.
fn recognize_pdf(helper: &Path, path: &Path, timeout: Duration) -> Result<String> {
    if !helper.is_file() { bail!("PDF Vision helper missing: {}", helper.display()); }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let output = std::env::temp_dir().join(format!(
        "local-search-pdf-ocr-{}-{stamp}.txt", std::process::id()
    ));
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)] {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let handle = opts.open(&output)?;
    let started = Command::new(helper)
        .arg("--ocr-pdf")
        .arg(path)
        .stdout(Stdio::from(handle))
        .stderr(Stdio::null())
        .spawn();
    let mut child = match started {
        Ok(c) => c,
        Err(err) => {
            let _ = std::fs::remove_file(&output);
            return Err(err).context("failed to start PDF Vision helper");
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
                    let _ = std::fs::remove_file(&output);
                    bail!("PDF Vision OCR timeout for {}", path.display());
                }
                std::thread::sleep(Duration::from_millis(60));
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&output);
                return Err(err.into());
            }
        }
    };
    let bytes = std::fs::read(&output);
    let _ = std::fs::remove_file(&output);
    if !status.success() {
        bail!("PDF Vision helper returned {} for {}", status, path.display());
    }
    let bytes = bytes?;
    if bytes.len() > 2 * 1024 * 1024 { bail!("OCR text exceeded 2 MiB"); }
    Ok(String::from_utf8(bytes)?.trim().to_owned())
}

/// Only image-only PDFs whose native text extraction returned no text.
/// For mixed PDFs a page-level merge is a separate (future) task.
/// Index at most `limit` files per call, no implicit filesystem traversal.
pub fn index_pending_scanned_pdfs(conn: &Connection, helper: &Path, limit: usize) -> Result<PdfVisionBatch> {
    if limit == 0 { return Ok(PdfVisionBatch::default()); }
    let pending: Vec<(i64, String, i64)> = {
        let mut query = conn.prepare(
            "SELECT f.id, f.path, f.mtime FROM files f
             LEFT JOIN file_ocr_failures e ON e.file_id=f.id AND e.mtime=f.mtime
             WHERE lower(f.ext) = 'pdf'
               AND (f.content IS NULL OR trim(f.content) = '')
               AND (f.ocr_mtime IS NULL OR f.ocr_mtime != f.mtime)
               AND (e.file_id IS NULL OR e.retry_at <= CAST(strftime('%s','now') AS INTEGER))
             ORDER BY f.id LIMIT ?1"
        )?;
        let rows = query.query_map([limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut report = PdfVisionBatch::default();
    for (id, filename, mtime) in pending {
        report.attempted += 1;
        let path = Path::new(&filename);
        match recognize_pdf(helper, path, Duration::from_secs(90)) {
            Ok(text) if !text.is_empty() => {
                // Don't write OCR output when a document changed during processing.
                let current_mtime = std::fs::metadata(path)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|t| t.as_secs() as i64);
                if current_mtime != Some(mtime) {
                    report.failed += 1;
                    continue;
                }
                report.indexed += conn.execute(
                    "UPDATE files SET content=?2, ocr_mtime=?3
                     WHERE id=?1 AND mtime=?3 AND (content IS NULL OR trim(content)='')",
                    params![id, text, mtime]
                )?;
            }
            Ok(_) => {
                // Blank document is considered attempted; do not spin forever.
                conn.execute(
                    "UPDATE files SET ocr_mtime=?2 WHERE id=?1 AND mtime=?2",
                    params![id, mtime]
                )?;
            }
            Err(err) => {
                log::warn!("PDF Vision OCR {}: {err:#}", path.display());
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

    fn fixture(name: &str) -> (std::path::PathBuf, rusqlite::Connection) {
        let parent = std::env::temp_dir().join(format!(
            "local-search-pdf-{}-{}", name, std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&parent);
        let root = parent.join("indexed");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("scanned.pdf"), "fake pdf bytes; OCR mocked").unwrap();
        let conn = db::open_in_memory().unwrap();
        files::add_folder(&conn, root.to_str().unwrap()).unwrap();
        files::index_folders(&conn, |_| {}).unwrap();
        (parent, conn)
    }

    #[test]
    fn russian_scanned_pdf_becomes_searchable_and_is_not_repeated() {
        let (parent, conn) = fixture("success");
        let executable = parent.join("mock.sh");
        std::fs::write(&executable,
            "#!/bin/sh\n[ \"$1\" = \"--ocr-pdf\" ] || exit 2\nprintf 'Замена масла в редукторе\\n'\n"
        ).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(files::files_fts_search(&conn, "редукторе", 10).unwrap().is_empty());
        assert_eq!(index_pending_scanned_pdfs(&conn, &executable, 1).unwrap(),
                   PdfVisionBatch { attempted: 1, indexed: 1, failed: 0 });
        assert_eq!(files::files_fts_search(&conn, "редукторе", 10).unwrap().len(), 1);
        assert_eq!(index_pending_scanned_pdfs(&conn, &executable, 1).unwrap(), PdfVisionBatch::default());
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn failed_pdf_ocr_keeps_file_pending() {
        let (parent, conn) = fixture("failure");
        let missing = parent.join("missing-pdf-helper");
        assert_eq!(index_pending_scanned_pdfs(&conn, &missing, 1).unwrap(),
                   PdfVisionBatch { attempted: 1, indexed: 0, failed: 1 });
        assert_eq!(index_pending_scanned_pdfs(&conn, &missing, 1).unwrap().attempted, 0);
        conn.execute("UPDATE file_ocr_failures SET retry_at=0", []).unwrap();
        assert_eq!(index_pending_scanned_pdfs(&conn, &missing, 1).unwrap().attempted, 1);
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn pdf_with_existing_text_is_never_ocrd() {
        let (parent, conn) = fixture("existing-text");
        conn.execute("UPDATE files SET content='Документ с текстовым слоем' WHERE ext='pdf'", []).unwrap();
        let report = index_pending_scanned_pdfs(&conn, &parent.join("missing"), 5).unwrap();
        assert_eq!(report, PdfVisionBatch::default());
        let _ = std::fs::remove_dir_all(&parent);
    }
}
