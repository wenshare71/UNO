/* 盲评前端:一屏一对,左/右三选一(左更好 / 一样好 / 右更好)。
 *
 * v2(§11 步骤 0):后端改成配对清单驱动,前端随之改三处——
 *   ① 槽位从 A/B 改名 L/R(与后端 `CHOICES` 一致);
 *   ② 揭盲结果按 kind 分组呈现,主指标是非平局胜率 + Wilson CI,不再是单一 score;
 *   ③ 回顾模式显示两侧的语义标签(可能是 post/pre、teacher 的两次跑,等等),
 *      不再假设"一边是 teacher 一边是 student"。
 *
 * 盲法纪律:盲评阶段只调 /api/tasks(其返回**不含** pair_id / kind / 语义标签);
 * /api/review 只在用户显式开启回顾模式时才调用,调用即视为揭盲。 */
"use strict";

let tasks = [];
let cur = 0;
// ---- 回顾模式状态(只存内存,刷新即回到盲评模式) ----
let reviewMode = false;
let reviewMap = {};        // idx -> review row
let filter = "all";        // all / key0 / key1 / tie / unmarked

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
  if (data.question) $("task-question").textContent = data.question;
  updateProgress(data.n_annotated, data.n_total);
}

function updateProgress(done, total) {
  $("progress-fill").style.width = total ? `${(100 * done) / total}%` : "0%";
  $("progress-text").textContent = `${done} / ${total}`;
}

function choiceText(c) {
  return { L: "左更好", R: "右更好", tie: "一样好" }[c] || c;
}

// ---- 过滤器:返回当前过滤条件下的 idx 列表(盲评模式恒为全部) ----
function filteredIndices() {
  if (!reviewMode || filter === "all") return tasks.map((t) => t.idx);
  return tasks.filter((t) => {
    const r = reviewMap[t.idx];
    if (!r) return filter === "unmarked";
    if (filter === "unmarked") return r.choice === null;
    if (filter === "tie") return r.winner === "tie";
    if (filter === "key0") return r.winner === r.focus_key;
    if (filter === "key1") return r.winner === r.baseline_key;
    return true;
  }).map((t) => t.idx);
}

function render() {
  const t = tasks[cur];
  if (!t) return;
  const r = reviewMode ? reviewMap[cur] : null;

  $("task-label").textContent = `#${cur + 1} / ${tasks.length}  [${t.stratum}]` +
    (t.annotated ? `  已评:${choiceText(t.choice)}` : "  未评") +
    (reviewMode && r ? `  (${r.kind} · ${r.src_task_id})` : "");
  $("task-prompt").textContent = `"${t.prompt}"`;

  // ---- 判定行(仅回顾模式) ----
  const vd = $("verdict");
  if (reviewMode && r) {
    vd.classList.remove("hidden");
    if (r.choice === null) {
      vd.textContent = "未评";
      vd.className = "verdict-none";
    } else if (r.winner === "tie") {
      vd.textContent = `你的标注:${choiceText(r.choice)} → 平局`;
      vd.className = "verdict-tie";
    } else {
      vd.textContent = `你的标注:${choiceText(r.choice)} → ${r.winner} 胜`;
      vd.className = r.winner === r.focus_key ? "verdict-student" : "verdict-teacher";
    }
  } else {
    vd.classList.add("hidden");
  }

  // ---- 身份徽标(仅回顾模式):直接显示语义标签 ----
  for (const [slot, key] of [["a", "left_key"], ["b", "right_key"]]) {
    const badge = $(`badge-${slot}`);
    if (reviewMode && r) {
      badge.textContent = r[key];
      badge.className = `badge ${r[key] === r.focus_key ? "badge-student" : "badge-teacher"}`;
    } else {
      badge.className = "badge hidden";
    }
  }

  const refs = $("refs");
  refs.innerHTML = "";
  for (let i = 0; i < t.n_refs; i++) {
    const img = document.createElement("img");
    img.src = imgUrl("ref", cur, i);
    img.alt = `ref${i + 1}`;
    refs.appendChild(img);
  }

  $("img-a").src = imgUrl("cand", cur, "L");
  $("img-b").src = imgUrl("cand", cur, "R");

  document.querySelectorAll(".choice").forEach((b) => {
    b.classList.toggle("active", t.annotated && t.choice === b.dataset.choice);
  });
  $("status").textContent = "";
}

function nextUnmarked(from) {
  for (let i = from + 1; i < tasks.length; i++) if (!tasks[i].annotated) return i;
  for (let i = 0; i <= from; i++) if (!tasks[i].annotated) return i;
  return -1;
}

// 在过滤集内移动;盲评模式过滤集 = 全部
function move(delta) {
  const pool = filteredIndices();
  if (!pool.length) { $("status").textContent = "当前过滤条件下没有条目。"; return; }
  const pos = pool.indexOf(cur);
  cur = pos < 0 ? pool[0] : pool[(pos + delta + pool.length) % pool.length];
  render();
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
    if (reviewMode && reviewMap[cur]) {  // 回顾模式下改判:本地同步身份映射
      const row = reviewMap[cur];
      row.choice = choice;
      row.winner = choice === "tie" ? "tie"
        : (choice === "L" ? row.left_key : row.right_key);
      render();  // 原地刷新判定行,不跳走
    } else {
      const nxt = nextUnmarked(cur);
      if (nxt >= 0 && nxt !== cur) {
        cur = nxt;
      } else if (r.n_annotated >= r.n_total) {
        $("status").textContent = "全部评完,可以揭盲了。";
      }
      render();
    }
  } catch (e) {
    $("status").textContent = `保存失败:${e.message}(标注未丢,可重试)`;
  }
}

const pct = (x) => (x === null || x === undefined ? "—" : `${(100 * x).toFixed(1)}%`);
const ci = (w) => (w ? `[${w[0].toFixed(3)}, ${w[1].toFixed(3)}]` : "—");
const num = (x, d = 3) => (x === null || x === undefined ? "—" : x.toFixed(d));

function statsTable(groups) {
  let html = "<table><tr><th>分组</th><th>n</th><th>win0</th><th>win1</th><th>平局</th>" +
    "<th>平局率</th><th>非平局胜率</th><th>Wilson 95% CI</th><th>旧口径</th></tr>";
  for (const [name, t] of Object.entries(groups)) {
    html += `<tr><td>${name}</td><td>${t.n}</td><td>${t.win_0}</td><td>${t.win_1}</td>` +
      `<td>${t.tie}</td><td>${pct(t.tie_rate)}</td><td>${pct(t.nontie_win_rate)}</td>` +
      `<td>${ci(t.wilson95)}</td><td>${num(t.legacy_score)}</td></tr>`;
  }
  return html + "</table>";
}

async function reveal() {
  const data = await api("/api/stats?reveal=0");
  const msg = data.complete
    ? "已评完,确认揭盲?"
    : `只评了 ${data.n_annotated}/${data.n_total},数字不是最终结果。仍要揭盲?`;
  if (!confirm(msg)) return;
  const s = await api("/api/stats?reveal=1");

  let html = `<p>批次 <b>${s.batch_id}</b> — 已标 ${s.n_annotated}/${s.n_total}</p>`;
  html += "<h3>按比较类型</h3>" + statsTable(s.by_kind);
  html += "<p class='hint'>win0 = 被检验方,win1 = 基线:";
  for (const [k, t] of Object.entries(s.by_kind)) {
    if (t.key_0) html += ` <code>${k}: ${t.key_0} vs ${t.key_1}</code>`;
  }
  html += "</p>";
  html += "<h3>按类型 × 分层</h3>" + statsTable(s.by_kind_stratum);

  html += "<h3>左右偏好</h3><table><tr><th>分组</th><th>L</th><th>R</th><th>平局</th>" +
    "<th>选左率</th><th>Wilson 95% CI</th></tr>";
  const pos = { ...s.position, 总体: s.position_overall };
  for (const [name, p] of Object.entries(pos)) {
    html += `<tr><td>${name}</td><td>${p.L}</td><td>${p.R}</td><td>${p.tie}</td>` +
      `<td>${pct(p.left_rate)}</td><td>${ci(p.wilson95)}</td></tr>`;
  }
  html += "</table><p class='hint'>CI 含 0.5 = 无位置偏好;不含 = 其它数字都要扣掉这个偏差项。</p>";

  if (s.replay) {
    html += `<h3>重放自洽率</h3><p class="score">${s.replay.same} / ${s.replay.n} = ` +
      `${pct(s.replay.agreement)}  ${ci(s.replay.wilson95)}</p>`;
    if (s.replay.flips.length) {
      html += "<table><tr><th>任务</th><th>M4 判定</th><th>本次</th></tr>";
      for (const f of s.replay.flips) {
        html += `<tr><td>${f.src_task_id}</td><td>${f.m4}</td><td>${f.now}</td></tr>`;
      }
      html += "</table>";
    }
  }
  if (!s.complete) html += `<p class="warn">注意:还有未评条目,以上为部分结果。</p>`;
  $("modal-content").innerHTML = html;
  $("modal").classList.remove("hidden");
}

async function toggleReview() {
  if (reviewMode) {  // 关闭:回到盲评展示
    reviewMode = false;
    reviewMap = {};
    filter = "all";
    $("filter").classList.add("hidden");
    $("filter").value = "all";
    $("btn-review").textContent = "回顾模式:关";
    $("btn-review").className = "review-off";
    render();
    return;
  }
  if (!confirm("回顾模式将显示每一对两侧各是什么、以及你的判定结果。\n" +
               "如果你还要继续盲评,请先评完再开启。确认开启?")) return;
  const data = await api("/api/review");
  reviewMap = {};
  for (const row of data.rows) reviewMap[row.idx] = row;
  reviewMode = true;
  $("filter").classList.remove("hidden");
  $("btn-review").textContent = "回顾模式:开";
  $("btn-review").className = "review-on";
  render();
}

function bind() {
  document.querySelectorAll(".choice").forEach((b) =>
    b.addEventListener("click", () => mark(b.dataset.choice)));
  $("btn-prev").addEventListener("click", () => move(-1));
  $("btn-next").addEventListener("click", () => move(1));
  $("btn-jump").addEventListener("click", () => {
    const nxt = nextUnmarked(cur - 1);
    if (nxt >= 0) { cur = nxt; render(); }
    else $("status").textContent = "没有未评条目了。";
  });
  $("btn-reveal").addEventListener("click", reveal);
  $("btn-review").addEventListener("click", toggleReview);
  $("filter").addEventListener("change", (e) => {
    filter = e.target.value;
    const pool = filteredIndices();
    if (pool.length && !pool.includes(cur)) cur = pool[0];
    render();
  });
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));

  document.addEventListener("keydown", (e) => {
    if (!$("modal").classList.contains("hidden")) {
      if (e.key === "Escape") $("modal").classList.add("hidden");
      return;
    }
    if (e.target.tagName === "SELECT") return;  // 下拉框占用方向键
    if (e.key === "1") mark("L");
    else if (e.key === "2" || e.key === "ArrowDown") mark("tie");
    else if (e.key === "3") mark("R");
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "ArrowRight") move(1);
  });
}

(async function main() {
  bind();
  await loadTasks();
  const nxt = nextUnmarked(-1);
  cur = nxt >= 0 ? nxt : 0;
  render();
})();
