import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type Model = { id:string; name_ru:string; status:string; downloaded_bytes:number; size_bytes_approx:number; detail:string };
type Summary = { files:number; image_pending:number; image_vectors:number; audio_running:boolean; audio_progress:string };
type Props = { coreModel:string; imageModel:string; onOpenModules:()=>void };

/** Compact progress/notification strip: always present even while settings scroll. */
export default function TestStatusBar({coreModel, imageModel, onOpenModules}: Props) {
  const [models, setModels] = useState<Model[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [connectionError, setConnectionError] = useState(false);
  useEffect(() => {
    let alive=true;
    const refresh = async () => {
      const [list, info] = await Promise.allSettled([
        invoke<Model[]>('optional_models_list'), invoke<Summary>('test_module_diagnostics')
      ]);
      if (!alive) return;
      if (list.status === 'fulfilled') setModels(list.value);
      if (info.status === 'fulfilled') {setSummary(info.value);setConnectionError(false);}
      else {setSummary(null);setConnectionError(true);}
    };
    void refresh();
    const timer=window.setInterval(()=>void refresh(), 1800);
    return ()=>{alive=false;window.clearInterval(timer);};
  }, []);
  const downloading=models.find(m=>m.status==='downloading');
  const failed=models.find(m=>m.status==='failed');
  const coreBusy=coreModel.startsWith('downloading') || coreModel.startsWith('loading');
  const imageBusy=imageModel.startsWith('downloading') || imageModel.startsWith('loading');
  // E5 and SigLIP already emit percentages for the CURRENT weight file.
  // This is not an overall multi-file percentage; label it accordingly.
  const activeCoreDownload = [
    {kind:'E5', value:coreModel}, {kind:'SigLIP',value:imageModel}
  ].find(item=>item.value.startsWith('downloading'));
  const rawModelPercent=activeCoreDownload?.value.match(/(?:^|\s)(\d{1,3})%/)?.[1];
  const modelPercent = rawModelPercent === undefined ? null : Math.min(100,Number(rawModelPercent));
  let message='Состояние фоновых операций';
  let description='';
  let current: number | null=null;
  if (downloading) {
    const percent=Math.min(100,Math.max(0,Math.floor(100*downloading.downloaded_bytes / Math.max(1,downloading.size_bytes_approx))));
    message=`Скачивание: ${downloading.name_ru}`;
    description=`${percent}% · ${Math.round(downloading.downloaded_bytes/1_000_000)} из ~${Math.round(downloading.size_bytes_approx/1_000_000)} МБ`;
    current=percent;
  } else if (coreBusy || imageBusy) {
    message=activeCoreDownload ? `Скачивание модели ${activeCoreDownload.kind}`
      : coreBusy ? 'Инициализация E5' : 'Инициализация SigLIP';
    description=modelPercent === null
      ? 'Подготовка модели; числовой прогресс пока недоступен'
      : `${modelPercent}% текущего файла модели`;
    current=modelPercent;
  } else if (summary?.audio_running) {
    message='Расшифровка аудиофайлов';
    description=summary.audio_progress;
  } else if (failed) {
    message=`Ошибка загрузки: ${failed.name_ru}`;
    description=failed.detail;
  } else if (connectionError) {
    message='Нет связи с диагностикой индекса';
    description='Проверьте, запущена ли тестовая программа';
  } else if (summary) {
    message=`В индексе DEV: ${summary.files} файлов`;
    description=`Визуальных векторов: ${summary.image_vectors}; ожидают: ${summary.image_pending}`;
  }
  const busy = downloading != null || coreBusy || imageBusy || Boolean(summary?.audio_running);
  return <div className="ls-test-statusbar" role="status" aria-live="off" title={`${message}. ${description}`}>
    <span className={`ls-status-indicator ${busy ? 'running' : failed ? 'failed' : ''}`} aria-hidden="true"/>
    <span className="ls-status-message">{message}</span>
    {downloading && <progress className="ls-status-progress" max={100} value={current ?? 0} aria-label="Прогресс загрузки модели"/>}
    {!downloading && (coreBusy || imageBusy) && <progress className="ls-status-progress" max={100} value={current === null ? undefined : current} aria-label="Загрузка компонента модели"/>}
    <span className="ls-status-detail">{description}</span>
    <button type="button" className="ls-status-link" onClick={onOpenModules}>Модули</button>
  </div>;
}
