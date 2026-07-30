/* M4 盲评前端:一屏一条,左 A 右 B,三选一(A更好/一样好/B更好)。
 * v2:回顾模式——揭盲后逐条回看身份(teacher/student)、自己的标注与判定,可按结果过滤。
 * 盲法纪律:盲评阶段只调 /api/tasks(无身份信息);/api/review 只在用户显式开启
 * 回顾模式时才调用,调用即视为揭盲。 */
"use strict";

let tasks = [];
let cur = 0;
// ---- 回顾模式状态(只存内存,刷新即回到盲评模式) ----
let reviewMode = false;
let reviewMap = {};        // idx -> {task_id, stratum, choice, student_on, winner}
let variantName = {};      // {teacher: "official_full", student: "ours_kv_post4000"}
let filter = "all";        // all / teacher / student / tie / unmarked

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

function choiceText(c) {
  return { A: "A 更好", B: "B 更好", tie: "一样好" }[c] || c;
}

function shortName(v) {  // official_full → teacher,ours_kv_post4000 → student
  if (v === variantName.teacher) return "teacher";
  if (v === variantName.student) return "student";
  return v;
}

// ---- 过滤器:返回当前过滤条件下的 idx 列表(盲评模式恒为全部) ----
function filteredIndices() {
  if (!reviewMode || filter === "all") return tasks.map((t) => t.idx);
  return tasks.filter((t) => {
    const r = reviewMap[t.idx];
    if (!r) return filter === "unmarked";
    if (filter === "unmarked") return r.choice === null;
    if (filter === "tie") return r.winner === "tie";
    if (filter === "teacher") return r.winner === variantName.teacher;
    if (filter === "student") return r.winner === variantName.student;
    return true;
  }).map((t) => t.idx);
}

function render() {
  const t = tasks[cur];
  if (!t) return;
  const r = reviewMode ? reviewMap[cur] : null;

  $("task-label").textContent = `#${cur + 1} / ${tasks.length}  [${t.stratum}]` +
    (t.annotated ? `  已评:${choiceText(t.choice)}` : "  未评") +
    (reviewMode && r ? `  (${r.task_id})` : "");
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
      vd.textContent = `你的标注:${choiceText(r.choice)} → ${shortName(r.winner)} 胜`;
      vd.className = r.winner === variantName.student ? "verdict-student" : "verdict-teacher";
    }
  } else {
    vd.classList.add("hidden");
  }

  // ---- 身份徽标(仅回顾模式) ----
  for (const slot of ["a", "b"]) {
    const badge = $(`badge-${slot}`);
    if (reviewMode && r) {
      const slotUpper = slot.toUpperCase();
      const identity = r.student_on === slotUpper ? "student" : "teacher";
      badge.textContent = identity;
      badge.className = `badge badge-${identity}`;
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

  $("img-a").src = imgUrl("cand", cur, "A");
  $("img-b").src = imgUrl("cand", cur, "B");

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
  const next = pos < 0 ? pool[0] : pool[(pos + delta + pool.length) % pool.length];
  cur = next;
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
        : (choice === row.student_on ? variantName.student : variantName.teacher);
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
  if (!confirm("回顾模式将显示每条的 teacher / student 身份与你的判定结果。\n" +
               "如果你还要继续盲评,请先评完再开启。确认开启?")) return;
  const data = await api("/api/review");
  variantName = { teacher: data.teacher, student: data.student };
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
    if (e.key === "1") mark("A");
    else if (e.key === "2" || e.key === "ArrowDown") mark("tie");
    else if (e.key === "3") mark("B");
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
