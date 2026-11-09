//! LocalSearch 1.0 — optional model library, macOS preflight implementation.
//! No calls to the document index; only app_data_dir/optional-modules.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Manager};
use serde::Serialize;

#[derive(Clone, Copy)]
struct Model { id: &'static str, name_ru: &'static str, file: &'static str, sha: &'static str,
    size: u64, url: &'static str }
const MODELS: [Model; 4] = [
    Model { id:"whisper-base-q5", name_ru:"Whisper Base · экономичная", file:"ggml-base-q5_1.bin",
        sha:"422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898", size:59_700_000,
        url:"https://huggingface.co/ggerganov/whisper.cpp/resolve/f281eb45af861ab5e5297d23694b7d46e090c02c/ggml-base-q5_1.bin" },
    Model { id:"whisper-small-q5", name_ru:"Whisper Small · основной", file:"ggml-small-q5_1.bin",
        sha:"ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb", size:190_000_000,
        url:"https://huggingface.co/ggerganov/whisper.cpp/resolve/c521a4b02f422512d734391fdf08bb08c0862f68/ggml-small-q5_1.bin" },
    Model { id:"whisper-medium-q5", name_ru:"Whisper Medium · качество", file:"ggml-medium-q5_0.bin",
        sha:"19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f", size:539_000_000,
        url:"https://huggingface.co/ggerganov/whisper.cpp/resolve/c521a4b02f422512d734391fdf08bb08c0862f68/ggml-medium-q5_0.bin" },
    Model { id:"whisper-turbo-q5", name_ru:"Whisper Large v3 Turbo · длинные записи", file:"ggml-large-v3-turbo-q5_0.bin",
        sha:"394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", size:574_041_195,
        url:"https://huggingface.co/ggerganov/whisper.cpp/resolve/c521a4b02f422512d734391fdf08bb08c0862f68/ggml-large-v3-turbo-q5_0.bin" },
];

#[derive(Clone, Serialize)]
pub struct OptionalModelStatus {
    id: &'static str, name_ru: &'static str, size_bytes_approx: u64,
    status: String, downloaded_bytes: u64, detail: String,
}
#[derive(Clone)]
struct Job { status: String, detail: String }
static JOBS: OnceLock<Mutex<HashMap<&'static str, Job>>> = OnceLock::new();
fn jobs() -> &'static Mutex<HashMap<&'static str, Job>> { JOBS.get_or_init(|| Mutex::new(HashMap::new())) }
fn find(id: &str) -> Option<Model> { MODELS.iter().find(|m| m.id == id).copied() }
fn root(app: &AppHandle) -> Result<PathBuf,String> {
    Ok(app.path().app_data_dir().map_err(|_|"Нет папки данных LocalSearch".to_string())?
       .join("optional-modules").join("asr-whisper").join("models"))
}
fn valid_marker(path: &Path, m: Model) -> bool {
    // Marker written only after SHA-256 verification. No expensive 0.5 GB hash on UI polling.
    if path.is_symlink() || !path.is_file() { return false; }
    let marker = path.with_extension("verified-sha256");
    marker.is_file() && std::fs::read_to_string(marker).map(|s|s.trim()==m.sha).unwrap_or(false)
}
fn update(id: &'static str, status: &str, detail: &str) {
    if let Ok(mut map) = jobs().lock() { map.insert(id, Job { status:status.into(), detail:detail.into() }); }
}

fn statuses_in_dir(dir: &Path, states: &HashMap<&'static str, Job>) -> Vec<OptionalModelStatus> {
MODELS.iter().map(|m| {
        let path = dir.join(m.file);
        let partial = dir.join(format!("{}.part",m.file));
        let installed = valid_marker(&path, *m);
        let job = states.get(m.id);
        let is_downloading = job.map(|j|j.status=="downloading").unwrap_or(false);
        let size = if !installed && is_downloading {std::fs::metadata(partial).map(|v|v.len()).unwrap_or(0)} else {0};
        OptionalModelStatus {id:m.id, name_ru:m.name_ru, size_bytes_approx:m.size,
            status:if installed {"installed"} else if is_downloading {"downloading"} else {
                job.map(|j| j.status.as_str()).unwrap_or("missing") }.to_string(),
            downloaded_bytes:size, detail: job.map(|j| j.detail.clone()).unwrap_or_default() }
    }).collect()
}

#[tauri::command]
pub fn optional_models_list(app: AppHandle) -> Result<Vec<OptionalModelStatus>,String> {
    let dir = root(&app)?;
    let states = jobs().lock().map_err(|_|"Очередь моделей временно недоступна".to_string())?.clone();
    Ok(statuses_in_dir(&dir, &states))
}

fn hash_matches(path: &Path, model: Model) -> bool {
    if !path.is_file() || path.is_symlink() {return false;}
    let out = Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(path).output();
    match out {Ok(result) if result.status.success() => {
        let v=String::from_utf8_lossy(&result.stdout);
        v.split_whitespace().next()==Some(model.sha)
    },_ => false}
}

#[cfg(target_os="macos")]
fn fetch(app: AppHandle, m: Model) -> Result<(),String> {
    let dir=root(&app)?;
    std::fs::create_dir_all(&dir).map_err(|_|"Не удалось создать папку моделей".to_string())?;
    let final_path=dir.join(m.file);
    let part=dir.join(format!("{}.part",m.file));
    if final_path.is_symlink() || part.is_symlink() {return Err("Небезопасный путь к модели".into());}
    if valid_marker(&final_path,m) || hash_matches(&final_path,m) {
        let _=std::fs::write(final_path.with_extension("verified-sha256"),format!("{}\n",m.sha));
        return Ok(());
    }
    // --continue-at - resumes interrupted downloads. TLS remains mandatory after redirects.
    let output=Command::new("/usr/bin/curl")
        .args(["--fail","--location","--silent","--show-error","--retry","3",
            "--retry-delay","2","--connect-timeout","20","--max-time","7200",
            "--proto","=https","--proto-redir","=https","--continue-at","-"])
        .arg("--output").arg(&part).arg(m.url).output()
        .map_err(|_|"Не удалось запустить системную загрузку HTTPS".to_string())?;
    if !output.status.success() {
        return Err("Не удалось скачать модель. Проверьте сеть; повторная попытка продолжит загрузку.".into());
    }
    if !hash_matches(&part,m) {
        // Range-ignore or interrupted responses can yield an invalid file; discard only PART.
        let _=std::fs::remove_file(&part);
        return Err("Контрольная сумма SHA-256 не совпала. Повторите загрузку.".into());
    }
    std::fs::rename(&part,&final_path).map_err(|_|"Не удалось завершить установку модели".to_string())?;
    std::fs::write(final_path.with_extension("verified-sha256"),format!("{}\n",m.sha))
        .map_err(|_|"Не удалось записать подтверждение целостности модели".to_string())?;
    Ok(())
}

#[tauri::command]
pub fn optional_model_connect(app: AppHandle, id: String) -> Result<(),String> {
    let m=find(&id).ok_or_else(||"Неизвестная модель".to_string())?;
    if valid_marker(&root(&app)?.join(m.file),m) {return Ok(());}
    {
        let mut states=jobs().lock().map_err(|_|"Очередь моделей временно недоступна".to_string())?;
        if states.get(m.id).map(|j|j.status.as_str())==Some("downloading") {return Ok(());}
        states.insert(m.id,Job {status:"downloading".into(),detail:"Скачивание и проверка модели".into()});
    }
    std::thread::spawn(move || {
        #[cfg(target_os="macos")]
        let result=fetch(app,m);
        #[cfg(not(target_os="macos"))]
        let result: Result<(),String>=Err("Загрузка этой сборкой поддерживается только на macOS".into());
        match result {Ok(())=>update(m.id,"installed","Модель скачана и проверена; для распознавания потребуется включить голосовой модуль"),
            Err(e)=>update(m.id,"failed",&e)}
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn filenames_and_digests_are_pinned() {
        for m in MODELS {assert!(m.url.starts_with("https://huggingface.co/ggerganov/whisper.cpp/resolve/"));
            assert_eq!(m.sha.len(),64);assert!(m.file.starts_with("ggml-"));assert!(!m.file.contains('/')); }
    }
    #[test] fn no_english_only_models_are_offered() {for m in MODELS {assert!(!m.file.contains(".en"));}}
    #[test] fn unlisted_model_is_rejected() {assert!(find("../../../bad").is_none());}
}

#[cfg(test)]
mod localsearch_autoqa_v10_progress {
    use super::*;

    #[test]
    fn optional_model_status_progress_and_terminal_states() {
        let root = std::env::temp_dir().join(format!(
            "localsearch-v10-optional-model-progress-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let m = MODELS[0];
        let part = root.join(format!("{}.part", m.file));
        let mut states = HashMap::new();
        let inspect = |s: &HashMap<&'static str, Job>| -> OptionalModelStatus {
            statuses_in_dir(&root, s)
                .into_iter()
                .find(|r| r.id == m.id)
                .expect("optional model missing")
        };
        let missing = inspect(&states);
        assert_eq!(missing.status, "missing");
        assert_eq!(missing.downloaded_bytes, 0);
        assert!(missing.size_bytes_approx > 0);

        states.insert(m.id, Job {
            status: "downloading".to_string(),
            detail: "Скачивание и проверка модели".to_string(),
        });
        std::fs::write(&part, vec![42u8; 4096]).unwrap();
        let first = inspect(&states);
        assert_eq!(first.status, "downloading");
        assert_eq!(first.downloaded_bytes, 4096);
        assert_eq!(first.size_bytes_approx, m.size);
        std::fs::write(&part, vec![42u8; 8192]).unwrap();
        let later = inspect(&states);
        assert_eq!(later.downloaded_bytes, 8192);
        assert!(later.downloaded_bytes > first.downloaded_bytes);

        states.insert(m.id, Job {
            status: "failed".to_string(),
            detail: "Оборвана загрузка".to_string(),
        });
        let failed = inspect(&states);
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.downloaded_bytes, 0); // queued partial is retained, not active
        assert!(part.is_file());

        states.insert(m.id, Job {
            status: "downloading".to_string(),
            detail: "Повторная попытка".to_string(),
        });
        assert_eq!(inspect(&states).downloaded_bytes, 8192); // resumable .part

        let final_path = root.join(m.file);
        std::fs::write(&final_path, b"verified-test-placeholder").unwrap();
        std::fs::write(final_path.with_extension("verified-sha256"), format!("{}\n", m.sha)).unwrap();
        let ready = inspect(&states);
        assert_eq!(ready.status, "installed");
        assert_eq!(ready.downloaded_bytes, 0);

        std::fs::remove_dir_all(&root).unwrap();
    }
}
