//! Optional LOCAL ASR module. Development preview: not enabled in the GUI.
//! No external service, network or document index operations in this file.
//! Model and whisper-cli are provisioned separately and validated by the
//! optional-model manager before installation. This module only runs them.

use base64::Engine as _;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_BASE64: usize = 10_700_000;

fn module_root(model_dir: &Path) -> PathBuf {
    // `model_dir` is the legacy E5 directory. Optional models are installed
    // under app_data_dir/optional-modules by model_library.rs, NOT under E5.
    model_dir.parent().unwrap_or(model_dir).join("optional-modules").join("asr-whisper")
}

const MODEL_FILES: [&str; 4] = [
    "ggml-small-q5_1.bin", "ggml-base-q5_1.bin",
    "ggml-medium-q5_0.bin", "ggml-large-v3-turbo-q5_0.bin",
];

fn installed_voice_model(root: &Path) -> Option<PathBuf> {
    MODEL_FILES.iter().map(|f| root.join("models").join(f)).find(|p| p.is_file())
}

pub fn status(model_dir: &Path) -> Value {
    let root = module_root(model_dir);
    let runtime = root.join("bin/whisper-cli");
    let installed = installed_voice_model(&root);
    json!({
        "id": "asr-whisper",
        "installed": runtime.is_file() && installed.is_some(),
        "engine_installed": runtime.is_file(),
        "model_present": installed.is_some(),
        "message": if !runtime.is_file() {"Модель можно скачать, но whisper.cpp CLI ещё не встроен в тестовую сборку"}
          else if installed.is_none() {"Модель речи не установлена"}
          else {"Локальное распознавание готово"}
    })
}

/// The browser records a PCM/16-bit/mono RIFF WAV at the device sample rate.
/// The command-line engine needs mono 16 kHz; do not assume the audio device
/// itself runs at 16 kHz on Mac.
fn normalize_wav(wav: &[u8]) -> Result<Vec<u8>, String> {
    if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("Ожидается WAV-запись".into());
    }
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let start = pos + 8;
        let end = start.checked_add(size).ok_or("Повреждён WAV-файл")?;
        if end > wav.len() { return Err("Повреждён WAV-файл".into()); }
        match &wav[pos..pos + 4] {
            b"fmt " if size >= 16 => {
                let f = &wav[start..end];
                fmt = Some((u16::from_le_bytes(f[0..2].try_into().unwrap()),
                    u16::from_le_bytes(f[2..4].try_into().unwrap()),
                    u32::from_le_bytes(f[4..8].try_into().unwrap()),
                    u16::from_le_bytes(f[14..16].try_into().unwrap())));
            }
            b"data" => data = Some(&wav[start..end]),
            _ => {}
        }
        pos = end.checked_add(size % 2).ok_or("Повреждён WAV-файл")?;
    }
    let (format, channels, sample_rate, bits) = fmt.ok_or("Нет WAV-заголовка")?;
    if format != 1 || channels != 1 || bits != 16 || !(8000..=96000).contains(&sample_rate) {
        return Err("Поддерживается только моно PCM 16 бит от 8 до 96 кГц".into());
    }
    let data = data.ok_or("В WAV отсутствует звук")?;
    if data.len() < 2 || data.len() % 2 != 0 { return Err("Повреждены звуковые данные".into()); }
    let samples: Vec<f32> = data.chunks_exact(2)
        .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32).collect();
    let n_out = (samples.len() as u64 * 16000 / sample_rate as u64) as usize;
    if n_out == 0 { return Err("Запись слишком короткая".into()); }
    let mut pcm = Vec::with_capacity(n_out * 2);
    for i in 0..n_out {
        // Time-domain weighted average to avoid simple decimation aliasing.
        // Higher-quality resampler can replace this function independently.
        let from = i as f64 * sample_rate as f64 / 16000.0;
        let to = (i + 1) as f64 * sample_rate as f64 / 16000.0;
        let a = from.floor() as usize;
        let b = (to.ceil() as usize).min(samples.len());
        let mut weight = 0.0f64;
        let mut total = 0.0f64;
        for j in a..b {
            let left = from.max(j as f64);
            let right = to.min((j + 1) as f64);
            let w = (right - left).max(0.0);
            weight += w;
            total += samples[j] as f64 * w;
        }
        let value = if weight > 0.0 { total / weight } else { 0.0 };
        pcm.extend_from_slice(&(value.round().clamp(-32768.0, 32767.0) as i16).to_le_bytes());
    }
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36u32 + pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16000u32.to_le_bytes());
    out.extend_from_slice(&32000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(&pcm);
    Ok(out)
}

fn temp_directory() -> Result<PathBuf, String> {
    let nano = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Ошибка часов: {e}"))?.as_nanos();
    let path = std::env::temp_dir().join(format!("localsearch-voice-{}-{nano}", std::process::id()));
    fs::create_dir(&path).map_err(|e| format!("Ошибка создания временной папки: {e}"))?;
    Ok(path)
}

struct TempWorkspace(PathBuf);
impl Drop for TempWorkspace {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn decode_transcript(value: &Value) -> Result<String, String> {
    let rows = value.get("transcription").and_then(Value::as_array)
        .ok_or("Движок вернул некорректный список фраз")?;
    let text = rows.iter().filter_map(|row| row.get("text").and_then(Value::as_str))
        .map(str::trim).filter(|t| !t.is_empty())
        .collect::<Vec<_>>().join(" ");
    if text.is_empty() { return Err("Речь не распознана".into()); }
    Ok(text)
}

pub fn transcribe_voice(model_dir: &Path, audio_base64: &str) -> Result<String, String> {
    if audio_base64.is_empty() || audio_base64.len() > MAX_BASE64 {
        return Err("Недопустимый размер голосовой записи".into());
    }
    let root = module_root(model_dir);
    let executable = root.join("bin/whisper-cli");
    if !executable.is_file() {
        return Err("Установите подключаемый модуль whisper.cpp".into());
    }
    let model = installed_voice_model(&root)
        .ok_or("Сначала скачайте модель распознавания речи")?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(audio_base64)
        .map_err(|_| "Некорректные звуковые данные")?;
    let wav = normalize_wav(&bytes)?;
    let tmp = TempWorkspace(temp_directory()?);
    let input = tmp.0.join("input.wav");
    let output = tmp.0.join("transcript");
    let mut f = File::create(&input).map_err(|e| format!("Не удалось создать WAV: {e}"))?;
    f.write_all(&wav).map_err(|e| format!("Не удалось сохранить WAV: {e}"))?;
    drop(f);
    let mut child = Command::new(executable)
        .arg("-m").arg(model).arg("-f").arg(input)
        .arg("-l").arg("ru").arg("-oj").arg("-of").arg(&output)
        .arg("-np").arg("-t").arg("4")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e| format!("Движок распознавания не запущен: {e}"))?;
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(result)) => break result,
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(90) {
                    let _ = child.kill(); let _ = child.wait();
                    return Err("Превышено время распознавания речи".into());
                }
                std::thread::sleep(Duration::from_millis(80));
            }
            Err(e) => {
                let _ = child.kill(); let _ = child.wait();
                return Err(format!("Ошибка движка распознавания: {e}"));
            }
        }
    };
    if !status.success() { return Err("Локальный движок не смог распознать запись".into()); }
    let path = output.with_extension("json");
    if fs::metadata(&path).map(|m| m.len() > 8 * 1024 * 1024).unwrap_or(true) {
        return Err("Ответ распознавания отсутствует или слишком велик".into());
    }
    let content = fs::read(&path).map_err(|e| format!("Не удалось прочитать результат: {e}"))?;
    let value: Value = serde_json::from_slice(&content)
        .map_err(|_| "Движок вернул некорректный JSON")?;
    decode_transcript(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wav_for(sample_rate: u32, samples: usize) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF"); v.extend_from_slice(&(36 + samples as u32 * 2).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt "); v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes()); v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data"); v.extend_from_slice(&(samples as u32 * 2).to_le_bytes());
        for _ in 0..samples {v.extend_from_slice(&1000i16.to_le_bytes());}
        v
    }
    #[test] fn resamples_48k_mono_to_16k_without_index_access() {
        let out = normalize_wav(&wav_for(48000, 48000)).unwrap();
        assert_eq!(out.len(), 44 + 32000);
        assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), 16000);
    }
    #[test] fn accepts_44k1_mono_and_rejects_malformed_wav() {
        assert_eq!(normalize_wav(&wav_for(44100, 4410)).unwrap().len(), 44 + 3200);
        assert!(normalize_wav(b"not a WAV").is_err());
    }
    #[test] fn parses_russian_json_without_rewriting_speech() {
        let value = json!({"transcription":[{"text":" Договор "},{"text":" номер 58 "}]});
        assert_eq!(decode_transcript(&value).unwrap(), "Договор номер 58");
    }
    #[test] fn unavailable_module_never_uses_fluidvoice_implicitly() {
        let root = std::env::temp_dir().join("localsearch-absent-ASR-test-0989");
        let result = transcribe_voice(&root, "AA==");
        assert!(result.unwrap_err().contains("подключаемый модуль"));
    }
}

/// Download a pinned, hash-verified Apple Silicon-compatible CMake runtime,
/// entirely inside the TEST module. Never require Python pip or Homebrew.
/// Official Kitware CMake v3.31.8 macOS universal asset, release SHA-256.
fn private_cmake(root: &Path) -> Result<PathBuf, String> {
    const URL: &str = "https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-macos-universal.tar.gz";
    const HASH: &str = "d1449f969c54d5c00886d5b643340d493dfb3c81cb39ee29b35453395c11ebf7";
    let tooling = root.join("tooling");
    fs::create_dir_all(&tooling).map_err(|e| format!("Не удалось создать каталог CMake: {e}"))?;
    let archive = tooling.join("cmake-3.31.8-macos-universal.tar.gz");
    let verified = |archive: &Path| -> Result<bool, String> {
        if !archive.is_file() { return Ok(false); }
        let out = Command::new("/usr/bin/shasum").args(["-a", "256"]).arg(archive)
            .output().map_err(|e| format!("Не удалось проверить SHA-256 CMake: {e}"))?;
        if !out.status.success() { return Err("Проверка SHA-256 архива CMake не выполнена".into()); }
        Ok(String::from_utf8_lossy(&out.stdout).split_whitespace().next() == Some(HASH))
    };
    if !verified(&archive)? {
        let partial = tooling.join("cmake-3.31.8-macos-universal.tar.gz.partial");
        let out = Command::new("/usr/bin/curl")
            .args(["--fail", "--location", "--silent", "--show-error", "--retry", "3",
                   "--connect-timeout", "20", "--max-time", "900", "--output"])
            .arg(&partial).arg(URL).output()
            .map_err(|e| format!("Не удалось скачать CMake: {e}"))?;
        if !out.status.success() {
            let _ = fs::remove_file(&partial);
            return Err(format!("Загрузка CMake из официального GitHub не удалась: {}",
                String::from_utf8_lossy(&out.stderr).chars().take(450).collect::<String>()));
        }
        if !verified(&partial)? {
            let _ = fs::remove_file(&partial);
            return Err("Загруженный CMake не прошёл SHA-256: архив отклонён".into());
        }
        fs::rename(&partial, &archive).map_err(|e| format!("Не удалось сохранить CMake: {e}"))?;
    }
    let extracted = tooling.join("cmake-3.31.8-unpacked");
    let candidates = [
        extracted.join("cmake-3.31.8-macos-universal/CMake.app/Contents/bin/cmake"),
        extracted.join("cmake-3.31.8-macos-universal/bin/cmake"),
        extracted.join("CMake.app/Contents/bin/cmake"),
    ];
    let available = || candidates.iter().find(|p| p.is_file() &&
        Command::new(p.as_os_str()).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)).cloned();
    if let Some(binary) = available() { return Ok(binary); }
    if extracted.exists() { fs::remove_dir_all(&extracted).map_err(|e|e.to_string())?; }
    fs::create_dir_all(&extracted).map_err(|e|e.to_string())?;
    let out = Command::new("/usr/bin/tar").args(["-xzf"]).arg(&archive)
        .arg("-C").arg(&extracted).output()
        .map_err(|e|format!("Не удалось распаковать CMake: {e}"))?;
    if !out.status.success() {
        return Err(format!("Не удалось распаковать CMake: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(350).collect::<String>()));
    }
    available().ok_or_else(|| "Официальный архив CMake распакован, но исполняемый файл не обнаружен".into())
}

/// Install a pinned whisper.cpp runtime into the isolated TEST profile.
/// This is an explicit action, not an app startup side effect. No Homebrew or
/// changes to system directories are needed when CMake is already available.
pub fn install_runtime(model_dir: &Path) -> Result<(),String> {
    use std::sync::atomic::{AtomicBool,Ordering};
    static BUILDING: AtomicBool = AtomicBool::new(false);
    struct Release<'a>(&'a AtomicBool);
    impl Drop for Release<'_> {fn drop(&mut self){self.0.store(false,Ordering::SeqCst);}}
    if BUILDING.swap(true,Ordering::SeqCst){return Err("Движок Whisper уже подготавливается".into());}
    let _release=Release(&BUILDING);
    const PIN: &str = "a8d002cfd879315632a579e73f0148d06959de36"; // upstream v1.7.6
    let root=module_root(model_dir);
    let launcher=root.join("bin/whisper-cli");
    if launcher.is_file() {return Ok(());}
    // First use any working CMake already on the Mac; otherwise download
    // Kitware's verified universal bundle privately (no pip/PyPI/Homebrew).
    let cmake = ["/opt/homebrew/bin/cmake", "/usr/local/bin/cmake", "/usr/bin/cmake", "cmake"]
        .iter().find(|c| Command::new(c).arg("--version").output()
            .map(|x| x.status.success()).unwrap_or(false))
        .map(|p| PathBuf::from(*p));
    let cmake = match cmake { Some(p) => p, None => private_cmake(&root)? };
    fs::create_dir_all(&root).map_err(|e| format!("Не удалось создать папку голосового модуля: {e}"))?;
    let source=root.join("runtime-staging");
    // An interrupted build is safely retried from scratch, without touching models.
    if source.exists() {fs::remove_dir_all(&source).map_err(|e|e.to_string())?;}
    let output=Command::new("/usr/bin/git")
        .args(["clone","--depth","1","--branch","v1.7.6","https://github.com/ggml-org/whisper.cpp.git"])
        .arg(&source).output().map_err(|e|format!("Не удалось скачать исходники Whisper: {e}"))?;
    if !output.status.success() {return Err(format!("Не удалось скачать whisper.cpp: {}",
        String::from_utf8_lossy(&output.stderr).chars().take(250).collect::<String>()));}
    let version=Command::new("/usr/bin/git").arg("-C").arg(&source)
        .args(["rev-parse","HEAD"]).output().map_err(|e|e.to_string())?;
    if String::from_utf8_lossy(&version.stdout).trim()!=PIN {
        return Err("Контрольная версия whisper.cpp не совпала: установка остановлена".into());
    }
    let config=Command::new(&cmake).arg("-S").arg(&source).arg("-B").arg(source.join("build"))
        .args(["-DGGML_METAL=ON","-DGGML_METAL_EMBED_LIBRARY=ON",
               "-DBUILD_SHARED_LIBS=OFF","-DCMAKE_BUILD_TYPE=Release","-DWHISPER_BUILD_TESTS=OFF"])
        .output().map_err(|e|format!("Ошибка CMake: {e}"))?;
    if !config.status.success() {return Err(format!("Конфигурация whisper.cpp: {}",
        String::from_utf8_lossy(&config.stderr).chars().take(380).collect::<String>()));}
    let build=Command::new(&cmake).arg("--build").arg(source.join("build"))
        .args(["--target","whisper-cli","--parallel","4"])
        .output().map_err(|e|format!("Ошибка сборки Whisper: {e}"))?;
    if !build.status.success() {return Err(format!("Сборка whisper.cpp: {}",
        String::from_utf8_lossy(&build.stderr).chars().take(400).collect::<String>()));}
    if !source.join("build/bin/whisper-cli").is_file() {return Err("Сборка Whisper не создала whisper-cli".into());}
    let runtime=root.join("runtime");
    if runtime.exists(){fs::remove_dir_all(&runtime).map_err(|e|e.to_string())?;}
    fs::rename(&source,&runtime).map_err(|e|format!("Не удалось разместить движок: {e}"))?;
    fs::create_dir_all(root.join("bin")).map_err(|e|e.to_string())?;
    // Keep the installed module relocatable when a QA profile is migrated.
    let script = "#!/bin/sh\nexec \"$(dirname \"$0\")/../runtime/build/bin/whisper-cli\" \"$@\"\n";
    fs::write(&launcher,script).map_err(|e|e.to_string())?;
    #[cfg(unix)] {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&launcher,fs::Permissions::from_mode(0o700)).map_err(|e|e.to_string())?;
    }
    Ok(())
}

/// Offline file transcription, with one bounded local ffmpeg conversion.
/// Never sends media to an API, modifies originals, or opens the production DB.
pub fn transcribe_file(model_dir: &Path, path: &Path) -> Result<String,String> {
    if !path.is_file(){return Err("Исходный звуковой файл недоступен".into());}
    if path.metadata().map(|m|m.len()>750*1024*1024).unwrap_or(true) {
        return Err("Файл слишком велик для тестового режима (750 МБ)".into());
    }
    let root=module_root(model_dir);
    let executable=root.join("bin/whisper-cli");
    if !executable.is_file(){return Err("Движок whisper.cpp не установлен".into());}
    let model=installed_voice_model(&root).ok_or("Модель Whisper не скачана")?;
    let tmp=TempWorkspace(temp_directory()?);
    let wav=tmp.0.join("input.wav");
    let ffmpeg=magpie_core::videos::ensure_ffmpeg_with(&root, &mut |_| {}).map(|v| v.0)
        .map_err(|e|format!("FFmpeg недоступен: {e}"))?;
    let convert=Command::new(ffmpeg).args(["-nostdin","-hide_banner","-loglevel","error","-y"])
        .arg("-i").arg(path).args(["-t","900","-map","0:a:0","-vn","-ar","16000","-ac","1",
          "-c:a","pcm_s16le","-f","wav"]).arg(&wav)
        .output().map_err(|e|format!("Ошибка декодирования: {e}"))?;
    if !convert.status.success(){return Err(format!("Не удалось извлечь звуковую дорожку: {}",
        String::from_utf8_lossy(&convert.stderr).chars().take(200).collect::<String>()));}
    if wav.metadata().map(|m|m.len()<44).unwrap_or(true){return Err("Пустая звуковая дорожка".into());}
    let output=tmp.0.join("transcript");
    let mut child=Command::new(executable).arg("-m").arg(model).arg("-f").arg(&wav)
        .args(["-l","ru","-oj","-np","-t","4"]).arg("-of").arg(&output)
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn()
        .map_err(|e|format!("Движок Whisper не запустился: {e}"))?;
    let start=Instant::now();
    loop {
        match child.try_wait(){
            Ok(Some(s)) => {if !s.success(){return Err("Whisper не смог распознать запись".into());}break;}
            Ok(None) if start.elapsed()<Duration::from_secs(1800) =>std::thread::sleep(Duration::from_millis(200)),
            Ok(None)=>{let _=child.kill();let _=child.wait();return Err("Превышено время распознавания (30 минут)".into());},
            Err(e)=>{let _=child.kill();let _=child.wait();return Err(e.to_string());}
        }
    }
    let path=output.with_extension("json");
    let bytes=fs::read(path).map_err(|e|format!("Нет JSON-результата Whisper: {e}"))?;
    if bytes.len()>8*1024*1024{return Err("Ответ Whisper слишком велик".into());}
    let value:Value=serde_json::from_slice(&bytes).map_err(|e|format!("Некорректная расшифровка: {e}"))?;
    decode_transcript(&value)
}
