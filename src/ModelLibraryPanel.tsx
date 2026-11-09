import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

export type OptionalModel = {
  id:string; name_ru:string; size_bytes_approx:number;
  status:'installed'|'downloading'|'missing'|'failed';
  downloaded_bytes:number; detail:string;
};
const mb=(n:number)=>`${Math.round(n/1000000)} МБ`;

/** Optional model downloads only; never resets search model caches or stars.db. */
export default function ModelLibraryPanel() {
  const [models,setModels]=useState<OptionalModel[]>([]);
  const [error,setError]=useState('');
  const [loading,setLoading]=useState(true);
  useEffect(()=>{
    let alive=true;
    async function read() {
      try {const list=await invoke<OptionalModel[]>('optional_models_list');
        if(alive){setModels(list);setLoading(false);}}
      catch {if(alive){setError('Не удалось получить список моделей.');setLoading(false);}}
    }
    void read();const t=setInterval(()=>void read(),1000);
    return ()=>{alive=false;clearInterval(t);};
  },[]);
  async function connect(id:string){
    setError('');
    try {await invoke('optional_model_connect',{id});}
    catch {setError('Не удалось начать загрузку. Подробности доступны в журнале приложения.');}
  }
  return <section className="ls-model-library" aria-label="Дополнительные модели LocalSearch">
    <h3>Подключаемые модели</h3>
    <p>Загрузка начинается сразу после нажатия «Подключить». Регистрация, терминал и сторонние программы не нужны.
      Файлы хранятся только на этом Mac; документы никуда не передаются.</p>
    {loading&&<p>Проверка моделей…</p>}
    {error&&<p role="alert">{error}</p>}
    {models.map(m=>{
      const active=m.status==='downloading',ready=m.status==='installed';
      const pct=Math.min(100,Math.max(0,Math.round(m.downloaded_bytes/Math.max(m.size_bytes_approx,1)*100)));
      return <div className="set-row" key={m.id}>
        <div className="set-label">
          <span className="set-name">{m.name_ru}</span>
          <span className="set-desc">{mb(m.size_bytes_approx)} · многоязычная · локальная</span>
          {active&&<><progress className="ls-model-progress" value={pct} max={100} aria-label={`Загрузка ${m.name_ru}`}/>
            <span className="set-desc">Скачано {mb(m.downloaded_bytes)} / ~{mb(m.size_bytes_approx)} ({pct}%) — можно продолжить после обрыва сети</span></>}
          {m.status==='failed'&&<span role="alert" className="set-desc">{m.detail||'Загрузка прервана. Можно повторить.'}</span>}
          {ready&&<span className="set-desc">Вес скачан и проверен; локальную расшифровку файлов можно запустить в диагностике ниже.</span>}
        </div>
        <button className="source" disabled={ready||active} onClick={()=>void connect(m.id)}>
          {ready?'Скачана':active?'Скачивается…':m.status==='failed'?'Повторить':'Подключить'}
        </button>
      </div>;
    })}
    <p className="set-desc">Apple Vision для распознавания рукописей встроен в macOS и не скачивается.
      Дополнительные HTR-модели появятся только после проверки качества и совместимости.
      Скачивание Whisper не отключает FluidVoice и не запускает переиндексацию.</p>
  </section>;
}
