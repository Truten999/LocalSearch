import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./IndexSourcesPanel.css";
import { explainErrorRu } from "./LocalSearchUX";

type Counters = { files: number; with_text: number; with_vectors: number };
type Folder = { id: number; path: string; available: boolean; files: number };
type TreeNode = {
  name: string;
  path: string;
  counts: Counters;
  direct_files: number;
  children: TreeNode[];
};
type Group = { id: string; counts: Counters; roots: TreeNode[] };
type Inventory = {
  counts: Counters;
  registered: Folder[];
  groups: Group[];
  inconsistent_paths: number;
};
const NAMES: Record<string, string> = {
  local: "На компьютере", icloud: "iCloud Drive и документы iCloud",
  dropbox: "Dropbox", google: "Google Диск", yandex: "Яндекс Диск",
  other_cloud: "Другие облачные папки", external: "Внешние диски",
};
const number = (n: number) => n.toLocaleString("ru-RU");

// Filter *folder paths*, not individual files. Preserve the real subtree totals.
// A matching folder expands all its descendants; a matching descendant keeps
// every parent visible, even when the top-level source group does not match.
export function matchingTree(node: TreeNode, needle: string): TreeNode | null {
  if (!needle || nodeMatches(node, needle)) return node;
  const children = node.children.map((child) => matchingTree(child, needle))
    .filter((child): child is TreeNode => child !== null);
  return children.length ? { ...node, children } : null;
}


function nodeMatches(node: TreeNode, needle: string): boolean {
  const part = needle.includes("/") ? node.path : node.name;
  return part.toLocaleLowerCase("ru-RU").includes(needle);
}
function containsSearchHit(node: TreeNode, needle: string): boolean {
  return nodeMatches(node, needle) ||
    node.children.some((child) => containsSearchHit(child, needle));
}

function FolderBranch({ node, depth, expanded, toggle, roots, setError, needle }: {
  node: TreeNode; depth: number; expanded: Set<string>; toggle: (key: string) => void;
  roots: string[]; setError: (s: string | null) => void; needle: string;
}) {
  const isOpen = expanded.has(node.path) || (Boolean(needle) && !nodeMatches(node, needle) &&
    node.children.some((child) => containsSearchHit(child, needle)));
  const canOpen = roots.some((root) => node.path === root || node.path.startsWith(`${root}/`));
  const containsChildren = node.children.length > 0;
  return <div className="idx-branch">
    <div className="idx-item" style={{ paddingLeft: Math.min(depth, 12) * 13 + 8 }}>
      <button className="idx-expand" onClick={() => toggle(node.path)} disabled={!containsChildren}
        aria-expanded={containsChildren ? isOpen : undefined} aria-label={`${isOpen ? "Свернуть" : "Раскрыть"} ${node.name}`}>
        {containsChildren ? (isOpen ? "▾" : "▸") : "·"}
      </button>
      <span className="idx-folder-name" title={node.path}>{node.name}</span>
      <span className="idx-folder-count" title={`${number(node.direct_files)} файлов непосредственно в этой папке`}>{number(node.counts.files)}</span>
      {canOpen && <button className="idx-open" title={`Открыть в Finder: ${node.path}`}
        onClick={() => { void invoke("open_path_default", { path: node.path }).catch((e) => setError(String(e))); }}>
        ↗
      </button>}
    </div>
    {isOpen && <div>{node.children.map((child) => <FolderBranch key={child.path}
      node={child} depth={depth + 1} expanded={expanded} toggle={toggle} roots={roots} setError={setError} needle={needle} />)}</div>}
  </div>;
}

export default function IndexSourcesPanel() {
  const [data, setData] = useState<Inventory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [folderQuery, setFolderQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    void invoke<Inventory>("indexed_sources")
      .then((r) => {
        setData(r);
        setExpanded(new Set(r.groups.filter((g) => g.counts.files > 0).map((g) => `group:${g.id}`)));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => { load(); }, [load]);
  const toggle = (key: string) => setExpanded((before) => {
    const next = new Set(before);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });
  const needle = folderQuery.trim().toLocaleLowerCase("ru-RU");
  const visibleGroups = useMemo(() => data?.groups.map((group) => ({
    ...group,
    roots: group.roots.map((root) => matchingTree(root, needle))
      .filter((root): root is TreeNode => root !== null),
  })) ?? [], [data, needle]);
  const roots = data?.registered.map((r) => r.path) ?? [];
  return <div className="idx-report">
    <div className="idx-report-head">
      <div>
        <h2>Источники индекса</h2>
        <p>Показываются фактически записанные в поисковую базу файлы и папки, а не все файлы на диске.</p>
      </div>
      <button className="ghost-btn" onClick={load} disabled={loading}>{loading ? "Считывание…" : "Обновить"}</button>
    </div>
    {error && <p className="error-line" role="alert">{explainErrorRu(error)}</p>}
    {data && <>
      <div className="idx-totals">
        <div><strong>{number(data.counts.files)}</strong><span>файлов в индексе</span></div>
        <div><strong>{number(data.counts.with_text)}</strong><span>с извлечённым текстом</span></div>
        <div><strong>{number(data.counts.with_vectors)}</strong><span>с семантическими векторами</span></div>
      </div>
      <p className="idx-hint">Облачные службы определяются по расположению папок. Это не подтверждение их синхронизации или наличия файлов в облаке. Числа показывают содержимое базы на момент обновления.</p>
      <h3>Структура индексированных файлов</h3>
      <label className="idx-filter-label" htmlFor="idx-folder-search">Найти папку в индексе</label>
      <input id="idx-folder-search" className="idx-filter" value={folderQuery}
        onChange={(e) => setFolderQuery(e.target.value)}
        placeholder="Название папки или часть пути" spellCheck={false}/>
      {needle && <p className="idx-hint">Показаны совпадающие ветки. Счётчики файлов относятся ко всей исходной папке.</p>}
      {visibleGroups.map((group) => {
        const groupKey = `group:${group.id}`;
        const isOpen = needle.length > 0 ? group.roots.length > 0 : expanded.has(groupKey);
        return <section className="idx-group" key={group.id}>
          <button className="idx-group-head" onClick={() => toggle(groupKey)} aria-expanded={isOpen}>
            <span>{isOpen ? "▾" : "▸"} {NAMES[group.id] ?? group.id}</span>
            <span className="idx-folder-count">{number(group.counts.files)} файлов</span>
          </button>
          {isOpen && <div className="idx-tree">
            {group.roots.length === 0
              ? <p className="idx-empty">В индексе таких файлов нет.</p>
              : group.roots.map((root) => <div className="idx-root" key={root.path}>
                  <p className="idx-fullpath" title={root.path}>{root.path}</p>
                  <FolderBranch node={root} depth={0} expanded={expanded} toggle={toggle} roots={roots} setError={setError} needle={needle} />
                </div>)}
          </div>}
        </section>;
      })}
      <h3>Папки, добавленные для индексации</h3>
      <p className="idx-hint">Эти папки приложение сканирует. Если папка недоступна, её ранее созданный индекс может сохраняться.</p>
      <div className="idx-registered">
        {data.registered.length ? data.registered.map((folder) => <div className="idx-register-row" key={folder.id}>
          <span className={folder.available ? "idx-available" : "idx-unavailable"} title={folder.available ? "Доступна" : "Сейчас недоступна"}>●</span>
          <span className="idx-registered-path" title={folder.path}>{folder.path}</span>
          <strong>{number(folder.files)}</strong>
        </div>) : <p className="idx-empty">Папки ещё не добавлены.</p>}
      </div>
      {data.inconsistent_paths > 0 && <p className="error-line">Некорректных записей пути: {number(data.inconsistent_paths)}. Они не включены в дерево.</p>}
    </>}
  </div>;
}
