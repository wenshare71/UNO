/* M4 盲评前端:一屏一条,左 A 右 B,三选一(A更好/一样好/B更好)。
 * 注意:本文件只接触 idx 与 A/B,服务端从不下发任何变体信息——盲法靠这个维持。 */
"use strict";

let tasks = [];
let cur = 0;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

function imgUrl(kind, idx, sub) {
  return `/api/img?k=${kind}:${idx}:${sub}`;
}

async function loadTasks() {
  const data = await api("/api/tasks");
  tasks = data.tasks;
  updateProgress(data.n_annotated, data.n_total);
}

function updateProgress(done, total) {
  $("progress-fill").style.width = total ? `${(100 * done) / total}%` : "0%";
  $("progress-text").textContent = `${done} / ${total}`;
}

function render() {
  const t = tasks[cur];
  if (!t) return;
  $("task-label").textContent = `#${cur + 1} / ${tasks.length}  [${t.stratum}]` +
    (t.annotated ? `  已评:${choiceText(t.choice)}` : "  未评");
  $("task-prompt").textContent = `"${t.prompt}"`;

  const refs = $("refs");
  refs.innerHTML = "";
  for (let i = 0; i < t.n_refs; i++) {
    const img = document.createElement("img");
    img.src = imgUrl("ref", cur, i);
    img.alt = `ref${i + 1}`;
    refs.appendChild(img);
  }

  // 加时间戳防浏览器缓存把换题后的图搞混(同一 key 内容不变,换题 key 变)
  $("img-a").src = imgUrl("cand", cur, "A");
  $("img-b").src = imgUrl("cand", cur, "B");

  document.querySelectorAll(".choice").forEach((b) => {
    b.classList.toggle("active", t.annotated && t.choice === b.dataset.choice);
  });
  $("status").textContent = "";
}

function choiceText(c) {
  return { A: "A 更好", B: "B 更好", tie: "一样好" }[c] || c;
}

function nextUnmarked(from) {
  for (let i = from + 1; i < tasks.length; i++) if (!tasks[i].annotated) return i;
  for (let i = 0; i <= from; i++) if (!tasks[i].annotated) return i;
  return -1;
}

async function mark(choice) {
  const t = tasks[cur];
  if (!t) return;
  try {
    const r = await api("/api/mark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idx: cur, choice }),
    });
    t.annotated = true;
    t.choice = choice;
    updateProgress(r.n_annotated, r.n_total);
    const nxt = nextUnmarked(cur);
    if (nxt >= 0 && nxt !== cur) {
      cur = nxt;
    } else if (r.n_annotated >= r.n_total) {
      $("status").textContent = "全部评完,可以揭盲了。";
    }
    render();
  } catch (e) {
    $("status").textContent = `保存失败:${e.message}(标注未丢,可重试)`;
  }
}

async function reveal() {
  const data = await api("/api/stats?reveal=0");
  const msg = data.complete
    ? "已评完,确认揭盲?"
    : `只评了 ${data.n_annotated}/${data.n_total},数字不是最终结果。仍要揭盲?`;
  if (!confirm(msg)) return;
  const s = await api("/api/stats?reveal=1");
  const fmt = (x) => (x.score === null ? "-" : x.score.toFixed(4));
  let html = `<p>总体:n=${s.overall.n}  T(teacher 更好)=${s.overall.T}  ` +
    `S(student 更好)=${s.overall.S}  B(一样好)=${s.overall.B}</p>` +
    `<p class="score">分数 (S+B)/(T+B) = ${fmt(s.overall)}</p>` +
    `<table><tr><th>层</th><th>n</th><th>T</th><th>S</th><th>B</th><th>分数</th></tr>`;
  for (const [st, v] of Object.entries(s.by_stratum)) {
    html += `<tr><td>${st}</td><td>${v.n}</td><td>${v.T}</td><td>${v.S}</td>` +
      `<td>${v.B}</td><td>${fmt(v)}</td></tr>`;
  }
  html += "</table>";
  if (!s.complete) html += `<p class="warn">注意:还有未评条目,以上为部分结果。</p>`;
  $("modal-content").innerHTML = html;
  $("modal").classList.remove("hidden");
}

function bind() {
  document.querySelectorAll(".choice").forEach((b) =>
    b.addEventListener("click", () => mark(b.dataset.choice)));
  $("btn-prev").addEventListener("click", () => { if (cur > 0) { cur--; render(); } });
  $("btn-next").addEventListener("click", () => {
    if (cur < tasks.length - 1) { cur++; render(); }
  });
  $("btn-jump").addEventListener("click", () => {
    const nxt = nextUnmarked(cur - 1);
    if (nxt >= 0) { cur = nxt; render(); }
    else $("status").textContent = "没有未评条目了。";
  });
  $("btn-reveal").addEventListener("click", reveal);
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));

  document.addEventListener("keydown", (e) => {
    if (!$("modal").classList.contains("hidden")) {
      if (e.key === "Escape") $("modal").classList.add("hidden");
      return;
    }
    if (e.key === "1" || e.key === "ArrowLeft") mark("A");
    else if (e.key === "2" || e.key === "ArrowDown") mark("tie");
    else if (e.key === "3" || e.key === "ArrowRight") mark("B");
  });
}

(async function main() {
  bind();
  await loadTasks();
  const nxt = nextUnmarked(-1);
  cur = nxt >= 0 ? nxt : 0;
  render();
})();
