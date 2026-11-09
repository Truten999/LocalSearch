//! Integrate the previously tested macOS Vision helpers with Magpie's GUI index pass.
//! Runs synchronously inside the existing spawn_blocking scan task, before embeddings.
//! No external OCR model is downloaded and no server/network request is made.
use anyhow::{Result, Context};
use magpie_core::rusqlite::Connection;
use std::path::Path;

/// Hard limits per *GUI index pass*, keeping indexing responsive on an Air.
const IMAGE_BATCH: usize = 24;
const PDF_BATCH: usize = 6;

#[derive(Debug, Default)]
pub struct NativeOcrSummary {
    pub attempted: usize,
    pub indexed: usize,
    pub failed: usize,
}

/// OCR is local and bounded. Failures of individual images/PDFs are counted
/// by the core helpers and do not abort folder scanning or embedding.
pub fn after_scan(
    conn: &Connection,
    helper_dir: &Path,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<NativeOcrSummary> {
    let mut summary = NativeOcrSummary::default();
    let img = helper_dir.join("local-search-ocr-step5");
    if img.is_file() {
        progress("vision-images", 0, IMAGE_BATCH);
        let report = magpie_core::native_vision::index_pending(conn, &img, IMAGE_BATCH)
            .context("Apple Vision images")?;
        summary.attempted += report.attempted;
        summary.indexed += report.indexed;
        summary.failed += report.failed;
        progress("vision-images", report.attempted, report.attempted);
    } else {
        log::warn!("local-search Apple Vision image helper not found: {}", img.display());
    }

    let pdf = helper_dir.join("local-search-pdf-ocr-step6");
    if pdf.is_file() {
        progress("vision-pdf", 0, PDF_BATCH);
        let report = magpie_core::native_pdf_vision::index_pending_scanned_pdfs(conn, &pdf, PDF_BATCH)
            .context("Apple Vision scanned PDFs")?;
        summary.attempted += report.attempted;
        summary.indexed += report.indexed;
        summary.failed += report.failed;
        progress("vision-pdf", report.attempted, report.attempted);
    } else {
        log::warn!("local-search Apple Vision PDF helper not found: {}", pdf.display());
    }
    Ok(summary)
}
