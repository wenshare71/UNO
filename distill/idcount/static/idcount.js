/* 身份留存计数前端:一屏一图,左侧每个参考主体单独问是/否,右侧是生成图。
 *
 * 盲法纪律照抄 blind_eval:`/api/items` 拿到的每个 item **不含** variant / img_path /
 * image_paths(见 server.py 模块 docstring——img_path 文件名里嵌着 variant,原样下发
 * 等于换个地方泄漏)。图片一律走 `/api/img?v=<tag>&k=<opaque key>` 拿,不直接用路径。
 *
 * 交互约束(需求原文):全部参考主体答完才能翻到下一张;`1`=当前待答项选"是",
 * `2`=选"否",`Backspace`=撤销上一个已答项,`←/→`=翻页(`→` 在未答完时会被挡)。 */
"use strict";

let items = [];        // [{item_id, task_id, stratum, prompt, ref_names, n_refs}, ...]
let marksMap = {};     // item_id -> {answers, dwell_ms, ts}(服务器已存的标注)
let localAnswers = []; // 当前 item 的 [true|false|null, ...],长度 = n_refs
let cur = 0;
let assetTag = "";
let nTotal = 0;
let nMarked = 0;
let itemStartTs = 0;   // 当前 item 何时进入视野,用于算 dwell_ms

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`${path} → ${r.status} ${detail}`.trim());
  }
  return r.json();
}

/* 版本戳必须进 URL:key 只含下标,换一批清单后同一下标可能指向不同的图而 URL 不变,
 * 浏览器会命中上一批的缓存——这正是 blind_eval 首次上机踩过的坑,这里原样防一遍。 */
function imgUrl(kind, idx, i) {
  return kind === "gen"
    ? `/api/img?v=${assetTag}&k=gen:${idx}`
    : `/api/img?v=${assetTag}&k=ref:${idx}:${i}`;
}

function isComplete(answers) {
  return answers.length > 0 && answers.every((a) => a !== null);
}

function pendingIndex(answers) {
  return answers.findIndex((a) => a === null);
}

function loadLocalAnswersForCurrent() {
  const it = items[cur];
  const saved = marksMap[it.item_id];
  localAnswers = saved
    ? saved.answers.slice()
    : new Array(it.n_refs).fill(null);
  itemStartTs = Date.now();
}

function updateProgress() {
  $("progress-fill").style.width = nTotal ? `${(100 * nMarked) / nTotal}%` : "0%";
  $("progress-text").textContent = `已标 ${nMarked} / ${nTotal}`;
}

async function loadItems() {
  const data = await api("/api/items");
  items = data.items;
  marksMap = data.marks;
  assetTag = data.tag;
  nTotal = data.n_total;
  nMarked = data.n_marked;
  updateProgress();
}

function firstUnmarkedIndex() {
  for (let i = 0; i < items.length; i++) {
    if (!marksMap[items[i].item_id]) return i;
  }
  return -1;
}

function render() {
  const it = items[cur];
  if (!it) return;
  const done = marksMap[it.item_id] !== undefined;

  $("item-label").textContent =
    `#${cur + 1} / ${items.length}  [${it.stratum}]` + (done ? "  已标注" : "  未标注");
  $("item-prompt").textContent = `"${it.prompt}"`;

  const pending = pendingIndex(localAnswers);

  const onImgError = (what) => () => {
    // 图加载失败必须显式报出来,不能静默显示破图——判断正是基于这张图。
    $("status").textContent = `${what} 加载失败——请刷新页面;若仍失败,先别标注。`;
  };

  const refs = $("refs");
  refs.innerHTML = "";
  for (let i = 0; i < it.n_refs; i++) {
    const block = document.createElement("div");
    block.className = "ref-block" + (i === pending ? " pending" : "");

    const img = document.createElement("img");
    img.onerror = onImgError(`参考图 ${i + 1}(${it.ref_names[i]})`);
    img.src = imgUrl("ref", cur, i);
    img.alt = it.ref_names[i];
    block.appendChild(img);

    const name = document.createElement("div");
    name.className = "ref-name";
    name.textContent = it.ref_names[i];
    block.appendChild(name);

    const btns = document.createElement("div");
    btns.className = "ref-buttons";
    const yes = document.createElement("button");
    yes.className = "ans yes" + (localAnswers[i] === true ? " active" : "");
    yes.innerHTML = `是${i === pending ? " <kbd>1</kbd>" : ""}`;
    yes.addEventListener("click", () => setAnswer(i, true));
    const no = document.createElement("button");
    no.className = "ans no" + (localAnswers[i] === false ? " active" : "");
    no.innerHTML = `否${i === pending ? " <kbd>2</kbd>" : ""}`;
    no.addEventListener("click", () => setAnswer(i, false));
    btns.appendChild(yes);
    btns.appendChild(no);
    block.appendChild(btns);

    refs.appendChild(block);
  }

  $("gen-img").onerror = onImgError("生成图");
  $("gen-img").src = imgUrl("gen", cur, null);

  $("status").textContent = "";
}

async function submitIfComplete() {
  if (!isComplete(localAnswers)) return;
  const it = items[cur];
  try {
    const dwell_ms = Date.now() - itemStartTs;
    const r = await api("/api/mark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: it.item_id, answers: localAnswers, dwell_ms }),
    });
    marksMap[it.item_id] = { answers: localAnswers.slice(), dwell_ms };
    nMarked = r.n_marked;
    nTotal = r.n_total;
    updateProgress();
    const nxt = firstUnmarkedIndex();
    if (nxt >= 0) {
      cur = nxt;
      loadLocalAnswersForCurrent();
    } else {
      $("status").textContent = "全部 66 个 item 已标注完成。";
    }
    render();
  } catch (e) {
    $("status").textContent = `保存失败:${e.message}(本地答案未丢,可重试)`;
  }
}

function setAnswer(i, val) {
  localAnswers[i] = val;
  render();
  submitIfComplete();
}

/* `→` 在未答完当前 item 时必须被挡——这是需求明确写的约束,不是可选项:
 * 单图是/否要求每个参考主体都过一遍判断,漏答一个就翻页,数据就带洞。 */
function move(delta) {
  if (delta > 0 && !isComplete(localAnswers)) {
    $("status").textContent = "先答完当前图的所有参考主体,才能翻到下一张。";
    return;
  }
  const next = cur + delta;
  if (next < 0 || next >= items.length) return;
  cur = next;
  loadLocalAnswersForCurrent();
  render();
}

function undo() {
  for (let i = localAnswers.length - 1; i >= 0; i--) {
    if (localAnswers[i] !== null) {
      localAnswers[i] = null;
      render();
      return;
    }
  }
  $("status").textContent = "当前图还没有已答项可撤销。";
}

function bind() {
  $("btn-prev").addEventListener("click", () => move(-1));
  $("btn-next").addEventListener("click", () => move(1));
  $("btn-undo").addEventListener("click", undo);

  document.addEventListener("keydown", (e) => {
    if (e.key === "1") {
      const p = pendingIndex(localAnswers);
      if (p >= 0) setAnswer(p, true);
    } else if (e.key === "2") {
      const p = pendingIndex(localAnswers);
      if (p >= 0) setAnswer(p, false);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      undo();
    } else if (e.key === "ArrowLeft") {
      move(-1);
    } else if (e.key === "ArrowRight") {
      move(1);
    }
  });
}

(async function main() {
  bind();
  await loadItems();
  const nxt = firstUnmarkedIndex();
  cur = nxt >= 0 ? nxt : 0;
  loadLocalAnswersForCurrent();
  render();
})();
