// Annotation logic for the distill viewer with server-side pagination.
// Only the current page rows are rendered into the DOM.

(function () {
  const statusEl = document.getElementById("anno-status");
  const progressEl = document.getElementById("anno-progress");
  const headerMetaEl = document.getElementById("header-meta");
  const rowsContainer = document.getElementById("rows-container");
  const setStatus = (t) => { if (statusEl) statusEl.textContent = t; };

  let sourceDesc = "";
  let manifestRows = [];
  let totalRows = 0;
  let page = 0;
  let perPage = 5;
  let totalPages = 1;
  const state = { meta: {}, annotations: {} };
  let saveTimer = null;
  let pendingSave = false;

  const annotatedCount = () =>
    Object.values(state.annotations).filter((a) => a && a.choice).length;
  const updateProgress = () => {
    if (progressEl) {
      const pass = Object.values(state.annotations).filter((a) => a && a.choice === "pass").length;
      const fail = Object.values(state.annotations).filter((a) => a && a.choice === "fail").length;
      progressEl.textContent = `已标注 ${annotatedCount()} / ${totalRows}  (pass=${pass} fail=${fail})`;
    }
  };

  function flushSave() {
    return new Promise((resolve) => {
      if (!saveTimer && !pendingSave) {
        resolve();
        return;
      }
      if (saveTimer) {
        clearTimeout(saveTimer);
        saveTimer = null;
      }
      apiPostAnnotations().then(resolve);
    });
  }

  async function apiPostAnnotations() {
    pendingSave = true;
    state.meta.updated_at = new Date().toISOString();
    try {
      const resp = await fetch("/api/annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations: state.annotations }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setStatus("已保存到服务器 " + new Date().toLocaleTimeString());
    } catch (e) {
      setStatus("保存失败: " + e);
    } finally {
      pendingSave = false;
    }
  }

  const scheduleSave = () => {
    setStatus("未保存…");
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      saveTimer = null;
      await apiPostAnnotations();
    }, 500);
  };

  const getRow = (name) => {
    if (!state.annotations[name]) state.annotations[name] = { choice: "", note: "" };
    return state.annotations[name];
  };

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderImage(rel, label, klass) {
    if (!rel) {
      return `<figure class="${klass}"><figcaption>${escapeHtml(label)}</figcaption><div class="missing">MISSING</div></figure>`;
    }
    const cls = ["lazy-img", "zoomable-img"];
    if (klass) cls.push(klass);
    return (
      `<figure class="${escapeHtml(klass)}">` +
      `<figcaption>${escapeHtml(label)}</figcaption>` +
      `<img class="${cls.join(" ")}" data-src="/api/image?rel=${encodeURIComponent(rel)}" ` +
      `data-full-src="/api/image?rel=${encodeURIComponent(rel)}" alt="${escapeHtml(label)}" ` +
      `title="Click to enlarge" loading="lazy" decoding="async">` +
      `</figure>`
    );
  }

  function renderRow(row, globalIdx) {
    const meta = row.meta || {};
    const nRefs = meta.n_refs || (row.image_paths || []).length;
    const hasAnimal = meta.has_animal;
    const subjects = meta.subjects || [];
    const viewIdx = meta.view_idx || [];
    const seed = meta.seed;
    const templateId = meta.template_id;
    const name = row.name;

    const nrefTag = `<span class="tag tag-${nRefs === 3 ? "3ref" : "2ref"}">${nRefs}-ref</span>`;
    const animalTag = hasAnimal
      ? '<span class="tag tag-animal">animal</span>'
      : '<span class="tag tag-object">object</span>';

    let imagesHtml = '';
    (row.image_paths || []).forEach((rel, i) => {
      const label = `ref${i + 1}: ${subjects[i] || "?"} (view ${viewIdx[i] ?? "?"})`;
      imagesHtml += renderImage(rel, label, "");
    });
    imagesHtml += renderImage(row.image_tgt_path, `teacher: ${row.image_tgt_path || "?"}`, "teacher");

    return (
      `<section class="row" data-name="${escapeHtml(name)}">` +
      `<div class="title">` +
      `<div class="title-left">` +
      `<span class="idx">#${String(globalIdx).padStart(5, "0")}</span>` +
      `${nrefTag}${animalTag}` +
      `<span>subjects: ${escapeHtml((subjects || []).join("+"))}</span>` +
      `<span>view: ${escapeHtml((viewIdx || []).join(","))}</span>` +
      `<span>template: ${templateId ?? "?"}</span>` +
      `<span>seed: ${seed ?? "?"}</span>` +
      `</div>` +
      `<span class="verdict"></span>` +
      `</div>` +
      `<div class="prompt"><b>Prompt:</b> ${escapeHtml(row.prompt || "")}</div>` +
      `<div class="images">${imagesHtml}</div>` +
      `<div class="annotation" data-name="${escapeHtml(name)}">` +
      `<div class="anno-item">` +
      `<span style="font-weight:700;color:#57606a;">标记:</span>` +
      `<button type="button" class="anno-btn" data-val="pass">pass (主体清晰、构图 OK)</button>` +
      `<button type="button" class="anno-btn" data-val="fail">fail (丢主体/构图崩)</button>` +
      `</div>` +
      `<div class="anno-note-wrap">` +
      `<textarea class="anno-note" placeholder="备注(可选):具体问题如'cat 丢失'、'背景不是 mountain'等" rows="1"></textarea>` +
      `</div>` +
      `</div>` +
      `</section>`
    );
  }

  function renderPagination() {
    const startIdx = page * perPage;
    const endIdx = Math.min(startIdx + perPage, totalRows);
    const pageOptions = [5, 10, 20, 50, 100].map((n) =>
      `<option value="${n}" ${n === perPage ? "selected" : ""}>${n} 条/页</option>`
    ).join("");

    let pageButtons = '';
    const maxVisible = 7;
    let startPage = Math.max(0, page - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible);
    if (endPage - startPage < maxVisible) {
      startPage = Math.max(0, endPage - maxVisible);
    }
    for (let i = startPage; i < endPage; i++) {
      const active = i === page ? ' style="background:#0a3069;color:#fff;"' : '';
      pageButtons += `<button type="button" class="page-btn" data-page="${i}"${active}>${i + 1}</button>`;
    }

    return (
      `<div class="pagination-bar">` +
      `<button type="button" class="page-btn" data-page="${page - 1}" ${page <= 0 ? "disabled" : ""}>上一页</button>` +
      `${pageButtons}` +
      `<button type="button" class="page-btn" data-page="${page + 1}" ${page >= totalPages - 1 ? "disabled" : ""}>下一页</button>` +
      `<span class="page-info">第 ${page + 1} / ${totalPages} 页 (${startIdx + 1}-${endIdx} / ${totalRows})</span>` +
      `<select class="per-page-select" title="每页条数">${pageOptions}</select>` +
      `</div>`
    );
  }

  function renderCurrentPage() {
    const startIdx = page * perPage;
    const rowsHtml = manifestRows.map((row, idx) => renderRow(row, startIdx + idx)).join("");
    rowsContainer.innerHTML = renderPagination() + rowsHtml + renderPagination();
    bindPaginationEvents();
    bindAnnotationEvents();
    if (window.initViewer) window.initViewer();
    applyState();
    updateHeaderMeta();
  }

  function updateHeaderMeta() {
    headerMetaEl.innerHTML =
      `任务数 ${totalRows} | 每页 ${perPage} 条 | 来源 ${escapeHtml(sourceDesc)} | ` +
      `快捷键 <code>p</code>=pass、<code>f</code>=fail(在标注按钮/备注框外时生效)`;
  }

  async function loadPage(newPage, newPerPage) {
    if (newPerPage !== undefined) perPage = newPerPage;
    await flushSave();
    setStatus(`正在加载第 ${newPage + 1} 页…`);
    try {
      const resp = await fetch(`/api/manifest?page=${newPage}&per_page=${perPage}`);
      if (!resp.ok) throw new Error(`manifest HTTP ${resp.status}`);
      const data = await resp.json();
      manifestRows = data.rows || [];
      totalRows = data.n_rows || 0;
      page = data.page || 0;
      totalPages = data.pages || 1;
      perPage = data.per_page || perPage;
      sourceDesc = data.source || "";
      state.meta.source = sourceDesc;
      state.meta.n_rows = totalRows;
      renderCurrentPage();
      setStatus("已加载，改动自动保存到服务器");
    } catch (e) {
      setStatus("加载失败: " + e);
    }
  }

  function bindPaginationEvents() {
    document.querySelectorAll(".page-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const targetPage = parseInt(btn.dataset.page, 10);
        if (targetPage >= 0 && targetPage < totalPages) {
          await loadPage(targetPage);
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });
    });
    document.querySelectorAll(".per-page-select").forEach((sel) => {
      sel.addEventListener("change", async () => {
        const newPerPage = parseInt(sel.value, 10);
        await loadPage(0, newPerPage);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });
  }

  const applyState = () => {
    document.querySelectorAll(".annotation").forEach((anno) => {
      const name = anno.dataset.name;
      const a = state.annotations[name];
      const choice = a && a.choice;
      anno.querySelectorAll(".anno-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.val === choice);
      });
      const note = anno.querySelector(".anno-note");
      if (note) note.value = a && a.note ? a.note : "";
      const row = anno.closest(".row");
      const verdict = row ? row.querySelector(".verdict") : null;
      if (row) row.classList.remove("pass", "fail");
      if (verdict) verdict.textContent = "";
      if (choice === "pass" && row && verdict) {
        row.classList.add("pass");
        verdict.textContent = "PASS";
        verdict.className = "verdict pass";
      } else if (choice === "fail" && row && verdict) {
        row.classList.add("fail");
        verdict.textContent = "FAIL";
        verdict.className = "verdict fail";
      }
    });
    updateProgress();
    applyOnlyTodo();
  };

  function bindAnnotationEvents() {
    document.querySelectorAll(".annotation").forEach((anno) => {
      const name = anno.dataset.name;
      anno.querySelectorAll(".anno-btn").forEach((b) => {
        b.addEventListener("click", () => {
          const row = getRow(name);
          if (row.choice === b.dataset.val) {
            row.choice = "";
          } else {
            row.choice = b.dataset.val;
          }
          row.updated_at = new Date().toISOString();
          applyState();
          scheduleSave();
        });
      });
      const note = anno.querySelector(".anno-note");
      if (note) {
        note.addEventListener("input", () => {
          getRow(name).note = note.value;
          scheduleSave();
        });
      }
    });
  }

  function applyOnlyTodo() {
    const onlyTodo = document.getElementById("anno-only-todo");
    if (!onlyTodo) return;
    const on = onlyTodo.checked;
    document.querySelectorAll("section.row").forEach((sec) => {
      const anno = sec.querySelector(".annotation");
      if (!anno) return;
      const a = state.annotations[anno.dataset.name];
      const done = a && a.choice;
      sec.style.display = on && done ? "none" : "";
    });
  }

  function findFirstUnannotated() {
    for (let i = 0; i < totalRows; i++) {
      const a = state.annotations[String(i)];
      if (!a || !a.choice) return i;
    }
    return -1;
  }

  async function jumpToFirstUnannotated() {
    await flushSave();
    const idx = findFirstUnannotated();
    if (idx === -1) {
      setStatus("全部已标注");
      return;
    }
    const targetPage = Math.floor(idx / perPage);
    await loadPage(targetPage);
    window.scrollTo({ top: 0, behavior: "smooth" });
    setStatus(`已跳到第 ${idx + 1} 条（第 ${targetPage + 1} 页）`);
  }

  async function init() {
    setStatus("正在加载…");
    try {
      const annoResp = await fetch("/api/annotations");
      if (!annoResp.ok) throw new Error(`annotations HTTP ${annoResp.status}`);
      const annoData = await annoResp.json();
      state.annotations = annoData.annotations || {};
      state.meta = Object.assign({}, state.meta, annoData.meta || {});

      // Get total rows so we can resume from the first unannotated row.
      const healthResp = await fetch("/api/health");
      if (healthResp.ok) {
        const health = await healthResp.json();
        totalRows = health.rows || 0;
      }
      updateProgress();

      const idx = findFirstUnannotated();
      const targetPage = idx === -1 ? 0 : Math.floor(idx / perPage);
      await loadPage(targetPage);
    } catch (e) {
      setStatus("初始化失败: " + e);
      headerMetaEl.textContent = "初始化失败: " + e;
    }
  }

  // Toolbar buttons
  document.getElementById("anno-save").addEventListener("click", async () => {
    await flushSave();
  });

  document.getElementById("anno-jump-todo").addEventListener("click", async () => {
    await jumpToFirstUnannotated();
  });

  document.getElementById("anno-download").addEventListener("click", () => {
    state.meta.updated_at = new Date().toISOString();
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations.json";
    a.click();
    URL.revokeObjectURL(url);
    setStatus("已下载 " + new Date().toLocaleTimeString());
  });

  document.getElementById("anno-import").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const data = JSON.parse(reader.result);
        if (data && data.annotations) {
          state.annotations = Object.assign({}, state.annotations, data.annotations);
          applyState();
          await flushSave();
          setStatus("已导入 " + file.name);
        } else {
          setStatus("文件里没有 annotations 字段");
        }
      } catch (err) {
        setStatus("导入失败: " + err);
      }
    };
    reader.readAsText(file);
  });

  document.getElementById("anno-clear").addEventListener("click", async () => {
    if (!confirm(`确定清空全部 ${totalRows} 条标注吗？（服务器会自动备份当前文件）`)) return;
    try {
      const resp = await fetch("/api/annotations/clear", { method: "POST" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.annotations = {};
      applyState();
      setStatus("已清空全部标注（旧数据已备份）" + new Date().toLocaleTimeString());
    } catch (e) {
      setStatus("清空失败: " + e);
    }
  });

  document.getElementById("anno-only-todo").addEventListener("change", () => {
    applyOnlyTodo();
  });

  // Keyboard shortcuts: p = pass, f = fail (current page only)
  document.addEventListener("keydown", async (event) => {
    if (event.target && (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT")) return;
    if (event.key !== "p" && event.key !== "f") return;
    const rows = Array.from(document.querySelectorAll("section.row")).filter((r) => r.style.display !== "none");
    for (const row of rows) {
      const rect = row.getBoundingClientRect();
      if (rect.bottom > 0 && rect.top < window.innerHeight) {
        const anno = row.querySelector(".annotation");
        if (!anno) continue;
        const btn = anno.querySelector(`.anno-btn[data-val="${event.key === "p" ? "pass" : "fail"}"]`);
        if (btn) { btn.click(); event.preventDefault(); }
        return;
      }
    }
  });

  init();
})();
