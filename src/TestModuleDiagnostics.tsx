import { useCallback, useEffect, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import './TestModuleDiagnostics.css';

type Diagnostic = {
  files: number; with_text: number; images: number; image_vectors: number;
  image_failures: number; image_pending: number; image_status: string;
  image_loading: boolean; image_loaded: boolean; semantic_enabled: boolean;
  semantic_paused: boolean;
  audio_files: number; audio_completed: number; audio_errors: number;
  audio_running: boolean; audio_progress: string;
  voice_runtime: { installed: boolean; engine_installed: boolean; model_present: boolean; message: string };
};

function imageState(data: Diagnostic): string {
  if (data.image_loading) return 'Модель загружается или обрабатывает изображения';
  if (data.semantic_paused) return 'Семантический поиск поставлен на паузу';
  if (!data.semantic_enabled) return 'Семантический поиск выключен';
  if (data.image_loaded) return 'Модель загружена';
  if (data.image_status === 'idle') return 'Модель пока не запускалась';
  if (data.image_status === 'ready') return 'Модель готова';
  return data.image_status;
}

/** Diagnostics and actions use only the current QA profile, independently of retired RC9/Test profiles. */
export default function TestModuleDiagnostics() {
  const [data, setData] = useState<Diagnostic | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const refreshInFlight = useRef(false);
  const refresh = useCallback(async () => {
    // A slow diagnostic must not accumulate overlapping backend requests.
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    try { setData(await invoke<Diagnostic>('test_module_diagnostics')); }
    catch (error) { setMessage(`Ошибка чтения диагностики: ${String(error)}`); }
    finally { refreshInFlight.current = false; }
  }, []);
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 30000);
    return () => clearInterval(timer);
  }, [refresh]);
  const action = async (command: string) => {
    setBusy(true);
    setMessage('');
    try { setMessage(await invoke<string>(command)); await refresh(); }
    catch (error) { setMessage(`Не удалось выполнить действие: ${String(error)}`); }
    finally { setBusy(false); }
  };

  return <section className="ls-test-diagnostics" aria-label="Диагностика модулей тестовой версии">
    <div className="ls-test-heading">
      <div>
        <h3>Диагностика и обработка</h3>
        <p>Только отдельный тестовый индекс. Рабочая RC9 не используется.</p>
      </div>
      <button className="source" type="button" onClick={() => void refresh()} disabled={busy}>Обновить</button>
    </div>
    {!data ? <p>Получаю состояние индекса…</p> : <>
      <div className="ls-test-panel">
        <h4>Документы и изображения</h4>
        <div className="ls-test-metrics">
          <div><strong>{data.files}</strong><span>файлов в базе</span></div>
          <div><strong>{data.with_text}</strong><span>с текстом</span></div>
          <div><strong>{data.images}</strong><span>изображений</span></div>
          <div><strong>{data.image_vectors}</strong><span>с векторами</span></div>
        </div>
        <p>Ожидают визуальной обработки: <b>{data.image_pending}</b>; ошибок декодирования: <b>{data.image_failures}</b>.</p>
      </div>
      <div className="ls-test-panel">
        <h4>SigLIP — поиск по описанию фотографий</h4>
        <p className="ls-test-status">{imageState(data)}</p>
        <p>Векторов готово: {data.image_vectors} из {data.images}. HEIC, HEIF и TIFF читаются через macOS. Повторная проверка не удаляет готовые векторы.</p>
        <div className="ls-test-actions">
          <button className="source" type="button" disabled={busy || data.image_loading}
            onClick={() => void action('test_visual_retry')}>
            {data.image_loading ? 'Идёт обработка…' : !data.semantic_enabled ? 'Включить визуальный поиск' : 'Проверить и дополнить векторы'}
          </button>
        </div>
      </div>
      <div className="ls-test-panel">
        <h4>Whisper — звук и аудиодорожки видео</h4>
        <p>Файлов: {data.audio_files}; расшифровано: {data.audio_completed}; ошибок: {data.audio_errors}.</p>
        <p className="ls-test-status">{data.voice_runtime.message}</p>
        <p>Очередь: {data.audio_progress}</p>
        <div className="ls-test-actions">
          {!data.voice_runtime.engine_installed && <button className="source" type="button" disabled={busy || data.audio_running}
            onClick={() => void action('test_voice_install_runtime')}>Подготовить движок</button>}
          <button className="source" type="button" disabled={busy || data.audio_running || !data.voice_runtime.model_present}
            onClick={() => void action('test_asr_transcribe_folder')}>
            {data.audio_running ? 'Идёт обработка…' : data.voice_runtime.engine_installed ? 'Расшифровать файлы' : 'Подготовить и расшифровать'}
          </button>
        </div>
        <p className="ls-test-note">Если движок ещё не подготовлен, приложение скачает CMake и исходники whisper.cpp в отдельный тестовый профиль. Не требуется Homebrew, pip или FluidVoice. Это первая установка движка; веса Whisper Small сохраняются.</p>
      </div>
    </>}
    {message && <p className="ls-test-message" role="status">{message}</p>}
  </section>;
}
