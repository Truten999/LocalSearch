import { useMemo, useState } from 'react';
import './LocalSearchUX.css';

export type SearchRecord = { query: string; source: string; when: number };
const KEY = 'localsearch.search-history.v1';
const DAYS_KEY = 'localsearch.search-history-days.v1';
function daysToKeep(): 7 | 30 {
  try { return localStorage.getItem(DAYS_KEY) === '7' ? 7 : 30; } catch { return 30; }
}
const LIMIT = 250;
const MAX_QUERY_LENGTH = 240;
const MAX_STORAGE = 95000;
const SOURCES: Record<string,string> = {
  local: 'Локальные файлы', web: 'Интернет', clips: 'Буфер обмена',
  'github-stars': 'Избранное GitHub'
};

/** The history is private, local-only, bounded in age, count and bytes. */
export function cleanSearchHistory(items: SearchRecord[], now = Date.now()): SearchRecord[] {
  const from = now - daysToKeep() * 24 * 3600 * 1000;
  return items.filter(x => typeof x.query === 'string' && typeof x.source === 'string'
    && Number.isFinite(x.when) && x.when <= now && x.when >= from
    && x.query.length > 0 && x.query.length <= MAX_QUERY_LENGTH)
    .sort((a,b) => b.when - a.when).slice(0,LIMIT);
}
export function readSearchHistory(): SearchRecord[] {
  try {
    const x = JSON.parse(localStorage.getItem(KEY) || '[]');
    return cleanSearchHistory(Array.isArray(x) ? x : []);
  } catch { return []; }
}
function persist(items: SearchRecord[]): SearchRecord[] {
  const clean = cleanSearchHistory(items);
  try {
    // Quota protection: never fill profile storage with history.
    while (clean.length && JSON.stringify(clean).length > MAX_STORAGE) clean.pop();
    localStorage.setItem(KEY, JSON.stringify(clean));
  } catch { /* History is optional; search must keep working. */ }
  return clean;
}
export function rememberSearch(query: string, source: string): void {
  const q = query.trim().replace(/\s+/g, ' ').slice(0,MAX_QUERY_LENGTH);
  if (q.length < 2) return;
  const items = readSearchHistory();
  // Keep the latest instance of an identical query in the same source.
  persist([{ query:q, source, when:Date.now() }, ...items.filter(x => !(x.source === source && x.query === q))]);
}
export function eraseSearchHistoryItem(entry: SearchRecord): void {
  persist(readSearchHistory().filter(x => !(x.when === entry.when && x.query === entry.query && x.source === entry.source)));
}
export function clearSearchHistory(): void {
  try {localStorage.removeItem(KEY);} catch { /* no-op */ }
}
/** Show Russian UI text even when the Tauri error is an English Rust message. */
export function explainErrorRu(error: unknown): string {
  const x = String(error || '');
  if (/contains already-indexed folder/i.test(x))
    return 'Внутри выбранной папки уже есть отдельный источник индекса. Используйте объединение вложенных папок без удаления данных.';
  if (/already covered by indexed folder/i.test(x))
    return 'Эта папка уже включена в индекс через родительскую папку.';
  if (/not a directory/i.test(x)) return 'Выбранный путь не является доступной папкой.';
  if (/permission denied|operation not permitted|not authorized/i.test(x))
    return 'Нет доступа к папке или файлу. Проверьте разрешения macOS.';
  if (/database is locked|SQLITE_BUSY/i.test(x))
    return 'Поисковая база сейчас занята. Попробуйте ещё раз через несколько секунд.';
  if (/no such file|not found|does not exist/i.test(x))
    return 'Файл или папка не найдены либо временно недоступны.';
  if (/timeout|timed out/i.test(x)) return 'Время ожидания истекло. Повторите действие.';
  // Existing Russian diagnostics may include paths/technical tokens; keep the message.
  if (/[А-Яа-яЁё]{3,}/.test(x)) return x;
  // Avoid exposing untranslated English exceptions to users. Preserve detail in log.
  if (x) console.warn('Техническая ошибка LocalSearch:',x);
  return 'Не удалось выполнить действие. Подробности записаны в технический журнал приложения.';
}

export default function SearchHistoryPanel({ onSearch, refreshToken }: {
  onSearch: (query:string,source:string) => void;
  refreshToken?: number;
}) {
  const [filter, setFilter] = useState('');
  const [retention, setRetention] = useState(daysToKeep);
  const [revision, setRevision] = useState(0);
  const entries = useMemo(() => readSearchHistory(), [revision,refreshToken]);
  const filtered = entries.filter(x => x.query.toLocaleLowerCase('ru-RU').includes(filter.toLocaleLowerCase('ru-RU')));
  const fmt = (when:number) => new Date(when).toLocaleString('ru-RU',{day:'2-digit',month:'long',hour:'2-digit',minute:'2-digit'});
  return <div className="ux-history">
    <div className="ux-history-hint">Запросы хранятся только на этом Mac. Ограничение — 250 записей, срок выбирается ниже. Старые удаляются автоматически.</div>
    <div className="ux-history-controls">
      <input aria-label="Поиск по истории" value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Поиск по истории запросов" />
      <label className="ux-history-retention">Срок хранения
        <select value={retention} onChange={e => {
          const days = e.target.value === '7' ? 7 : 30;
          try { localStorage.setItem(DAYS_KEY,String(days)); } catch { /* optional */ }
          setRetention(days); setRevision(n=>n+1);
        }}>
          <option value={7}>7 дней</option><option value={30}>30 дней</option>
        </select>
      </label>
      <button className="ghost-btn" onClick={()=>{ if(window.confirm('Очистить всю историю поисковых запросов?')) { clearSearchHistory(); setRevision(n=>n+1); } }} disabled={!entries.length}>Очистить историю</button>
    </div>
    {filtered.length===0 ? <p className="ux-history-empty">{filter ? 'Подходящих запросов нет.' : 'История пока пуста.'}</p> :
      <div className="ux-history-list">{filtered.map(e=><div className="ux-history-row" key={`${e.when}:${e.source}:${e.query}`}>
        <button className="ux-history-recall" onClick={()=>onSearch(e.query,e.source)} title="Повторить поиск">
          <span>{e.query}</span><small>{fmt(e.when)} · {SOURCES[e.source] ?? 'Поиск'}</small>
        </button>
        <button className="ux-history-remove" aria-label="Удалить запрос" title="Удалить запрос" onClick={()=>{eraseSearchHistoryItem(e);setRevision(n=>n+1)}}>×</button>
      </div>)}</div>}
  </div>;
}
