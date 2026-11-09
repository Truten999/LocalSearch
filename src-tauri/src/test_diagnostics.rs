//! Isolated Test-profile diagnostics and opt-in media indexing.
//! Never touches a production profile or rebuilds a text/vector index.
use super::{apply_semantic, models_allowed, spawn_siglip_init, AppState};
use magpie_core::{db, rusqlite::params};
use serde_json::{json, Value};
use std::{path::PathBuf, sync::atomic::{AtomicBool, Ordering}, sync::{Mutex, OnceLock}, time::{Duration, Instant}};
use tauri::{AppHandle, Manager};

static AUDIO_BUSY: AtomicBool = AtomicBool::new(false);
static AUDIO_STATUS: OnceLock<Mutex<String>> = OnceLock::new();
fn audio_status() -> &'static Mutex<String> { AUDIO_STATUS.get_or_init(|| Mutex::new("Не запускалось".into())) }
fn update_audio(s: impl Into<String>) { if let Ok(mut guard) = audio_status().lock() { *guard = s.into(); } }

// Computing the precise text counter reads the entire document-content column.
// Keep it exact, but do not rescan the large SQLite table on every UI refresh.
static TEXT_COUNT_CACHE: OnceLock<Mutex<Option<(Instant, i64)>>> = OnceLock::new();
const TEXT_COUNT_TTL: Duration = Duration::from_secs(120);

fn files_with_text_cached(conn: &magpie_core::rusqlite::Connection) -> Result<i64, String> {
    let cache = TEXT_COUNT_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().map_err(|_| "Text counter cache poisoned".to_string())?;
    if let Some((measured_at, count)) = *guard {
        if measured_at.elapsed() < TEXT_COUNT_TTL {
            return Ok(count);
        }
    }
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE length(trim(coalesce(content,'')))>0",
        [],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    *guard = Some((Instant::now(), count));
    Ok(count)
}

/// Runs in Tauri's blocking pool, never on the AppKit/WebKit main thread.
/// Raw counters, not proof that every file has been semantically indexed.
#[tauri::command]
pub async fn test_module_diagnostics(app: AppHandle) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || test_module_diagnostics_blocking(app))
        .await
        .map_err(|e| format!("Diagnostic worker failed: {e}"))?
}

fn test_module_diagnostics_blocking(app: AppHandle) -> Result<Value, String> {
    let st = app.state::<AppState>();
    let conn = db::open(&st.db_path).map_err(|e| e.to_string())?;
    let n = |sql: &str| -> Result<i64, String> { conn.query_row(sql, [], |r| r.get(0)).map_err(|e| e.to_string()) };
    let images = "('jpg','jpeg','png','webp','bmp','gif','heic','heif','tif','tiff')";
    let media = "('wav','mp3','m4a','aac','ogg','flac','mp4','mov','mkv')";
    let count_files = n("SELECT COUNT(*) FROM files")?;
    let text = files_with_text_cached(&conn)?;
    let images_all = n(&format!("SELECT COUNT(*) FROM files WHERE lower(ext) IN {images}"))?;
    let images_vectors = n(&format!("SELECT COUNT(*) FROM image_embeddings ie JOIN files f ON ie.file_id=f.id WHERE lower(f.ext) IN {images} AND ie.dim>0"))?;
    let images_failed = n(&format!("SELECT COUNT(*) FROM image_embeddings ie JOIN files f ON ie.file_id=f.id WHERE lower(f.ext) IN {images} AND ie.dim=0"))?;
    let missing = images_all.saturating_sub(images_vectors + images_failed);
    let audio = n(&format!("SELECT COUNT(*) FROM files WHERE lower(ext) IN {media}"))?;
    let asr_done = n("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_audio_transcripts'")?;
    let asr_done = if asr_done>0 {n("SELECT COUNT(*) FROM test_audio_transcripts")?}else{0};
    let asr_errors = n("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_audio_errors'")?;
    let asr_errors = if asr_errors>0 {n("SELECT COUNT(*) FROM test_audio_errors")?}else{0};
    let status = crate::offline_voice::status(&st.model_dir);
    Ok(json!({
        "files":count_files,"with_text":text,"images":images_all,
        "image_vectors":images_vectors,"image_failures":images_failed,"image_pending":missing,
        "image_status":st.siglip_status.lock().map(|v|v.clone()).unwrap_or_else(|_|"неизвестно".into()),
        "image_loading":st.siglip_initing.load(Ordering::SeqCst),
        "semantic_enabled":models_allowed(&st),
        "semantic_paused":st.models_paused.load(Ordering::SeqCst),
        "image_loaded":st.siglip.try_lock().map(|v|v.is_some()).unwrap_or(false),
        "audio_files":audio,"audio_completed":asr_done,"audio_errors":asr_errors,
        "audio_running":AUDIO_BUSY.load(Ordering::SeqCst),
        "audio_progress":audio_status().lock().map(|v|v.clone()).unwrap_or_default(),
        "voice_runtime":status
    }))
}

#[tauri::command]
pub fn test_visual_retry(app: AppHandle) -> Result<String, String> {
    let state = app.state::<AppState>();
    if state.models_paused.load(Ordering::SeqCst) {
        return Err("Семантический поиск поставлен на паузу. Возобнови его в настройках, затем повтори проверку.".into());
    }
    // Explicit user click authorizes turning semantic search on in the TEST database only.
    if !models_allowed(&state) {
        let conn = db::open(&state.db_path).map_err(|e|e.to_string())?;
        magpie_core::semantic::set(&conn,true).map_err(|e|e.to_string())?;
        drop(conn);
        apply_semantic(&app,true);
    }
    if state.siglip_initing.load(Ordering::SeqCst) {return Ok("Модель изображений уже загружается".into());}
    // Safe for the Test database: recomputes only missing/stale images, no full rescan.
    spawn_siglip_init(app.clone());
    Ok("SigLIP запускается. Следи за состоянием модели и количеством готовых векторов; ранее созданные векторы сохраняются.".into())
}

#[tauri::command]
pub fn test_voice_install_runtime(app: AppHandle) -> Result<String, String> {
    let model_dir = app.state::<AppState>().model_dir.clone();
    // The installer never touches the production profile, no Homebrew calls.
    std::thread::spawn(move || {
        update_audio("Подготовка локального движка whisper.cpp…");
        match crate::offline_voice::install_runtime(&model_dir) {
            Ok(()) => update_audio("Движок whisper.cpp готов"),
            Err(e) => update_audio(format!("Ошибка установки движка: {e}")),
        }
    });
    Ok("Начата подготовка локального движка whisper.cpp".into())
}

fn init_audio_tables(conn: &magpie_core::rusqlite::Connection) -> Result<(), String> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS test_audio_transcripts (\
         file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,\
         mtime INTEGER NOT NULL, size INTEGER NOT NULL, model TEXT NOT NULL, transcript TEXT NOT NULL);\
         CREATE TABLE IF NOT EXISTS test_audio_errors (\
         file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE, message TEXT NOT NULL);")
         .map_err(|e| format!("Не удалось создать таблицы аудиомодуля: {e}"))
}

fn run_audio(db_path: PathBuf, model_dir: PathBuf) -> Result<(), String> {
    let conn=db::open(&db_path).map_err(|e|e.to_string())?;
    init_audio_tables(&conn)?;
    let mut stmt=conn.prepare("SELECT id, path, mtime, size FROM files WHERE lower(ext) IN \
       ('mp3','wav','m4a','aac','flac','ogg','mov','mp4','mkv') ORDER BY id")
       .map_err(|e|e.to_string())?;
    let jobs=stmt.query_map([], |r| Ok((r.get::<_,i64>(0)?, r.get::<_,String>(1)?,
        r.get::<_,i64>(2)?, r.get::<_,i64>(3)?))).map_err(|e|e.to_string())?
        .collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;
    drop(stmt);
    if jobs.is_empty() { update_audio("В индексе пока нет аудио/видео файлов"); return Ok(()); }
    let mut ok=0; let mut failed=0;
    for (idx,(id,path,mtime,size)) in jobs.iter().enumerate() {
        update_audio(format!("Аудио {}/{} · успешно {} · ошибок {}",idx+1,jobs.len(),ok,failed));
        let last: Option<(i64,i64,String)> = conn.query_row("SELECT mtime,size,transcript FROM test_audio_transcripts WHERE file_id=?1",
            [id], |r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).ok();
        let transcript = if let Some((old_time,old_size,ref old_text))=last {
            if old_time==*mtime && old_size==*size {Some(old_text.clone())}else{None}
        } else {None};
        let transcript = match transcript {
            Some(t)=>Ok(t),
            None => crate::offline_voice::transcribe_file(&model_dir, std::path::Path::new(path)),
        };
        match transcript {
            Ok(t) if !t.trim().is_empty() => {
                conn.execute("INSERT INTO test_audio_transcripts(file_id,mtime,size,model,transcript) VALUES(?1,?2,?3,'whisper.cpp',?4) \
                    ON CONFLICT(file_id) DO UPDATE SET mtime=excluded.mtime,size=excluded.size,model=excluded.model,transcript=excluded.transcript",
                    params![id,mtime,size,t]).map_err(|e|e.to_string())?;
                // Only replace name-only content. Text already indexed elsewhere is preserved.
                conn.execute("UPDATE files SET content=?2 WHERE id=?1 AND (content IS NULL OR trim(content)='')",
                    params![id,t]).map_err(|e|e.to_string())?;
                conn.execute("DELETE FROM test_audio_errors WHERE file_id=?1",[id]).map_err(|e|e.to_string())?;
                ok+=1;
            }
            Ok(_)=>{failed+=1;}
            Err(e)=>{
                conn.execute("INSERT INTO test_audio_errors(file_id,message) VALUES(?1,?2) ON CONFLICT(file_id) DO UPDATE SET message=excluded.message",
                    params![id,e]).map_err(|e|e.to_string())?;
                failed+=1;
            }
        }
    }
    update_audio(format!("Обработка завершена · успешно {ok} · ошибок {failed}"));
    Ok(())
}

#[tauri::command]
pub fn test_asr_transcribe_folder(app: AppHandle) -> Result<String, String> {
    if AUDIO_BUSY.swap(true, Ordering::SeqCst) {return Ok("Обработка звука уже выполняется".into());}
    let state=app.state::<AppState>();
    let db_path=state.db_path.clone(); let model_dir=state.model_dir.clone();
    if !crate::offline_voice::status(&model_dir).get("model_present").and_then(Value::as_bool).unwrap_or(false) {
        AUDIO_BUSY.store(false, Ordering::SeqCst);
        return Err("Сначала скачай модель Whisper в разделе подключаемых моделей".into());
    }
    update_audio("Проверка локального движка и запуск очереди…");
    std::thread::spawn(move || {
        let result=(|| {
            if !crate::offline_voice::status(&model_dir).get("engine_installed").and_then(Value::as_bool).unwrap_or(false) {
                update_audio("Автоматически подготавливаю whisper.cpp…");
                crate::offline_voice::install_runtime(&model_dir)?;
            }
            run_audio(db_path,model_dir)
        })();
        if let Err(e)=result {update_audio(format!("Ошибка очереди: {e}"));}
        AUDIO_BUSY.store(false,Ordering::SeqCst);
    });
    Ok("Расшифровка звуковых файлов запущена".into())
}

#[cfg(test)] mod tests {
    #[test] fn audio_list_only_contains_supported_local_file_extensions() {
        assert!(["mp3","wav","m4a","mov"].iter().all(|x|!x.contains('/')));
    }
}
