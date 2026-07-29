#!/usr/bin/env python3
"""为 distill_multiref 数据集生成人工质检 HTML。

复用 /kaimm-distill/wuwenxuan/upload_compare.py 的 CSS/JS 风格(懒加载、点击放大、
G/S/B 风格的标注面板 + localStorage 缓存 + JSON 下载),把"teacher vs student"对比
改造为"prompt + N 张 ref 图 + teacher 生成图"的人工质检面板。

发布方式:本地 file:// 打开,图片全部用相对路径,所以脚本会把 HTML 输出到
datasets/distill_multiref/ 目录下,这样 images/000000.jpg 与 ../dreambooth/dataset/...
都能被浏览器正确解析。

用法:
    python distill/inspect_html.py                       # 默认全量 8000
    python distill/inspect_html.py --limit 200           # 只前 200 条
    python distill/inspect_html.py --shuffle --limit 200 # 随机抽 200
    python distill/inspect_html.py --shard 0             # 只看 shard0 的 1000 条
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "distill_multiref" / "manifest_raw.json"
OUT_DIR = ROOT / "datasets" / "distill_multiref"


# ---------------------------------------------------------------------------
# 复制并改造自 upload_compare.py 的 CSS/JS
# 主要差异:
#   - 把"teacher | student1 | student2..."的并列 figure,改成"ref 1 | ref 2 | ref 3 | teacher"
#   - 标注按钮从 G/S/B 改为 pass/fail + 备注
#   - 不依赖 share-tools,直接相对路径 file://
# ---------------------------------------------------------------------------
HEAD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UNO Distill 人工质检面板</title>
<style>
body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f6f7f9; color: #1f2328; }
header { position: sticky; top: 0; z-index: 1; padding: 14px 20px; background: #ffffff; border-bottom: 1px solid #d8dee4; }
h1 { margin: 0 0 6px; font-size: 20px; }
.meta { color: #57606a; font-size: 13px; line-height: 1.6; }
.row { margin: 16px; padding: 16px; background: #ffffff; border: 1px solid #d8dee4; border-radius: 10px; content-visibility: auto; contain-intrinsic-size: 900px; }
.row.fail { border-color: #cf222e; background: #fff8f8; }
.row.pass { border-color: #1f883d; background: #f6fcf8; }
.row.skipped { opacity: 0.55; }
.title { font-weight: 700; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
.title-left { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
.title-left .idx { color: #57606a; font-weight: 600; }
.title-left .tag { font-size: 12px; padding: 2px 8px; border-radius: 999px; }
.tag-2ref { background: #ddf4ff; color: #0a3069; }
.tag-3ref { background: #fff8c5; color: #633c01; }
.tag-animal { background: #f6e6ff; color: #5d2f8c; }
.tag-object { background: #eaeef2; color: #57606a; }
.verdict { font-size: 14px; font-weight: 700; padding: 4px 10px; border-radius: 999px; }
.verdict.pass { background: #1f883d; color: #fff; }
.verdict.fail { background: #cf222e; color: #fff; }
.prompt { white-space: pre-wrap; line-height: 1.6; margin: 8px 0 14px; font-size: 15px; }
.images { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; align-items: start; justify-content: start; overflow-x: auto; }
figure { margin: 0; }
figure.teacher { outline: 2px solid #0a3069; outline-offset: -2px; border-radius: 8px; }
figcaption { margin-bottom: 6px; font-size: 12px; font-weight: 700; color: #57606a; overflow: hidden; overflow-wrap: anywhere; line-height: 1.3; }
img { display: block; width: 100%; height: auto; border-radius: 8px; border: 1px solid #d8dee4; background: #f0f0f0; }
img.zoomable-img { cursor: zoom-in; }
.image-modal { position: fixed; inset: 0; z-index: 10; display: none; align-items: center; justify-content: center; padding: 24px; overflow: auto; background: rgba(0, 0, 0, 0.86); }
.image-modal.is-open { display: flex; }
.image-modal img { max-width: 96vw; max-height: 92vh; width: auto; height: auto; border: 0; border-radius: 8px; background: transparent; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); cursor: zoom-in; transform-origin: center center; user-select: none; -webkit-user-select: none; -webkit-user-drag: none; touch-action: none; }
.image-modal-close { position: fixed; top: 16px; right: 20px; border: 0; border-radius: 999px; padding: 8px 12px; font-size: 24px; line-height: 1; color: #ffffff; background: rgba(255, 255, 255, 0.18); cursor: pointer; }
.image-modal-tools { position: fixed; top: 16px; right: 72px; display: flex; gap: 8px; }
.image-modal-tools button { border: 0; border-radius: 999px; padding: 8px 12px; font-size: 14px; line-height: 1; color: #ffffff; background: rgba(255, 255, 255, 0.18); cursor: pointer; }
.image-modal-nav { position: fixed; top: 50%; transform: translateY(-50%); border: 0; border-radius: 999px; width: 48px; height: 64px; font-size: 42px; line-height: 1; color: #ffffff; background: rgba(255, 255, 255, 0.18); cursor: pointer; }
.image-modal-prev { left: 20px; }
.image-modal-next { right: 20px; }
.missing { min-height: 200px; display: flex; align-items: center; justify-content: center; border: 1px dashed #d8dee4; border-radius: 8px; color: #cf222e; background: #fff8f8; font-size: 13px; }
.anno-toolbar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-top: 8px; font-size: 13px; }
.anno-toolbar button { border: 1px solid #d8dee4; background: #f6f8fa; border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
.anno-toolbar button:hover { background: #eaeef2; }
#anno-status { color: #57606a; }
.anno-toggle { color: #57606a; cursor: pointer; user-select: none; }
.annotation { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 12px; padding: 10px 12px; background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 8px; }
.anno-item { display: flex; align-items: center; gap: 6px; }
.anno-btn { border: 1px solid #d8dee4; background: #fff; border-radius: 6px; padding: 5px 14px; cursor: pointer; font-size: 13px; font-weight: 600; }
.anno-btn:hover { background: #eef2f6; }
.anno-btn.active[data-val="pass"] { background: #1f883d; color: #fff; border-color: #1f883d; }
.anno-btn.active[data-val="fail"] { background: #cf222e; color: #fff; border-color: #cf222e; }
.anno-note-wrap { flex: 1 1 240px; }
.anno-note { width: 100%; box-sizing: border-box; resize: vertical; border: 1px solid #d8dee4; border-radius: 6px; padding: 6px 8px; font-size: 13px; font-family: inherit; }
@media (max-width: 900px) { .images { grid-template-columns: 1fr 1fr; } }
</style>
</head>
"""

MODAL_HTML = """
<div class="image-modal" id="image-modal" aria-hidden="true">
  <div class="image-modal-tools" aria-label="Image zoom controls">
    <button type="button" id="image-modal-zoom-out" aria-label="Zoom out">-</button>
    <button type="button" id="image-modal-zoom-reset" aria-label="Reset zoom">100%</button>
    <button type="button" id="image-modal-zoom-in" aria-label="Zoom in">+</button>
  </div>
  <button type="button" class="image-modal-close" id="image-modal-close" aria-label="Close image preview">&times;</button>
  <button type="button" class="image-modal-nav image-modal-prev" id="image-modal-prev" aria-label="Previous image">&lsaquo;</button>
  <img id="image-modal-img" alt="" draggable="false">
  <button type="button" class="image-modal-nav image-modal-next" id="image-modal-next" aria-label="Next image">&rsaquo;</button>
</div>
"""

VIEWER_SCRIPT = """<script>
const lazyImages = document.querySelectorAll("img.lazy-img");
const loadImage = (img) => {
  if (!img.dataset.src) return;
  img.src = img.dataset.src;
  img.removeAttribute("data-src");
};

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      loadImage(entry.target);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "900px 0px" });

  lazyImages.forEach((img) => observer.observe(img));
} else {
  lazyImages.forEach(loadImage);
}

const modal = document.getElementById("image-modal");
const modalImg = document.getElementById("image-modal-img");
const modalClose = document.getElementById("image-modal-close");
const modalPrev = document.getElementById("image-modal-prev");
const modalNext = document.getElementById("image-modal-next");
const modalZoomIn = document.getElementById("image-modal-zoom-in");
const modalZoomOut = document.getElementById("image-modal-zoom-out");
const modalZoomReset = document.getElementById("image-modal-zoom-reset");
const zoomableImages = Array.from(document.querySelectorAll("img.zoomable-img"));
const MIN_SCALE = 1;
const MAX_SCALE = 8;
let currentImageIndex = -1;
let modalScale = 1;
let panX = 0;
let panY = 0;
let isDragging = false;
let didDrag = false;
let dragStartX = 0;
let dragStartY = 0;
let dragStartPanX = 0;
let dragStartPanY = 0;

const clampScale = (scale) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));

const clampPan = () => {
  const baseW = modalImg.clientWidth;
  const baseH = modalImg.clientHeight;
  if (!baseW || !baseH) return;
  const maxX = Math.max(0, (baseW * modalScale - window.innerWidth) / 2);
  const maxY = Math.max(0, (baseH * modalScale - window.innerHeight) / 2);
  panX = Math.min(maxX, Math.max(-maxX, panX));
  panY = Math.min(maxY, Math.max(-maxY, panY));
};

const applyZoom = () => {
  clampPan();
  modalImg.style.transform = `translate(${panX}px, ${panY}px) scale(${modalScale})`;
  modalImg.style.cursor = modalScale > 1 ? "grab" : "zoom-in";
};

const setZoom = (scale) => {
  const nextScale = clampScale(scale);
  if (nextScale <= 1) {
    panX = 0;
    panY = 0;
  }
  modalScale = nextScale;
  applyZoom();
};

const zoomAt = (scale, clientX, clientY) => {
  const nextScale = clampScale(scale);
  if (nextScale === modalScale) return;
  const rect = modalImg.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const factor = 1 - nextScale / modalScale;
  panX += (clientX - centerX) * factor;
  panY += (clientY - centerY) * factor;
  modalScale = nextScale;
  if (modalScale <= 1) {
    panX = 0;
    panY = 0;
  }
  applyZoom();
};

const showImage = (index) => {
  if (!zoomableImages.length) return;
  currentImageIndex = (index + zoomableImages.length) % zoomableImages.length;
  const img = zoomableImages[currentImageIndex];
  loadImage(img);
  modalImg.src = img.dataset.fullSrc || img.src;
  modalImg.alt = img.alt;
  setZoom(1);
};

const closeModal = () => {
  modal.classList.remove("is-open");
  modal.setAttribute("aria-hidden", "true");
  modalImg.removeAttribute("src");
  modalImg.alt = "";
  modalImg.style.transform = "";
  currentImageIndex = -1;
  panX = 0;
  panY = 0;
  isDragging = false;
  didDrag = false;
};

const shiftImage = (delta) => {
  if (!modal.classList.contains("is-open")) return;
  showImage(currentImageIndex + delta);
};

zoomableImages.forEach((img, index) => {
  img.addEventListener("click", () => {
    showImage(index);
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
  });
});

modal.addEventListener("click", (event) => {
  if (event.target === modal) closeModal();
});
modalClose.addEventListener("click", closeModal);
modalPrev.addEventListener("click", () => shiftImage(-1));
modalNext.addEventListener("click", () => shiftImage(1));
modalZoomIn.addEventListener("click", () => setZoom(modalScale + 0.25));
modalZoomOut.addEventListener("click", () => setZoom(modalScale - 0.25));
modalZoomReset.addEventListener("click", () => setZoom(1));
modal.addEventListener("wheel", (event) => {
  if (!modal.classList.contains("is-open")) return;
  event.preventDefault();
  const step = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoomAt(modalScale * step, event.clientX, event.clientY);
}, { passive: false });
window.addEventListener("resize", () => {
  if (modal.classList.contains("is-open")) applyZoom();
});
modalImg.addEventListener("dragstart", (event) => event.preventDefault());
modalImg.addEventListener("pointerdown", (event) => {
  if (modalScale <= 1) return;
  event.preventDefault();
  isDragging = true;
  didDrag = false;
  dragStartX = event.clientX;
  dragStartY = event.clientY;
  dragStartPanX = panX;
  dragStartPanY = panY;
  modalImg.setPointerCapture(event.pointerId);
  modalImg.style.cursor = "grabbing";
});
modalImg.addEventListener("pointermove", (event) => {
  if (!isDragging) return;
  const dx = event.clientX - dragStartX;
  const dy = event.clientY - dragStartY;
  didDrag = didDrag || Math.abs(dx) > 3 || Math.abs(dy) > 3;
  panX = dragStartPanX + dx;
  panY = dragStartPanY + dy;
  applyZoom();
});
const stopDragging = () => {
  if (!isDragging) return;
  isDragging = false;
  modalImg.style.cursor = modalScale > 1 ? "grab" : "zoom-in";
};
modalImg.addEventListener("pointerup", stopDragging);
modalImg.addEventListener("pointercancel", stopDragging);
modalImg.addEventListener("click", (event) => {
  if (didDrag) {
    didDrag = false;
    return;
  }
  if (modalScale > 1) {
    setZoom(1);
  } else {
    zoomAt(2, event.clientX, event.clientY);
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && modal.classList.contains("is-open")) closeModal();
  if (event.key === "ArrowLeft") shiftImage(-1);
  if (event.key === "ArrowRight") shiftImage(1);
  if ((event.key === "+" || event.key === "=") && modal.classList.contains("is-open")) setZoom(modalScale + 0.25);
  if ((event.key === "-" || event.key === "_") && modal.classList.contains("is-open")) setZoom(modalScale - 0.25);
  if (event.key === "0" && modal.classList.contains("is-open")) setZoom(1);
});
</script>
"""

# 标注脚本:localStorage 缓存 + 下载 JSON + 只看未标注过滤 + 行高亮
ANNO_SCRIPT = """<script>
(function () {
  const cfg = window.ANNO_CONFIG || {};
  const state = { meta: cfg.meta || {}, annotations: {} };
  const statusEl = document.getElementById("anno-status");
  const progressEl = document.getElementById("anno-progress");
  const setStatus = (t) => { if (statusEl) statusEl.textContent = t; };

  const totalRows = () => document.querySelectorAll(".annotation").length;
  const annotatedCount = () =>
    Object.values(state.annotations).filter((a) => a && a.choice).length;
  const updateProgress = () => {
    if (progressEl) {
      const pass = Object.values(state.annotations).filter((a) => a && a.choice === "pass").length;
      const fail = Object.values(state.annotations).filter((a) => a && a.choice === "fail").length;
      progressEl.textContent = `已标注 ${annotatedCount()} / ${totalRows()}  (pass=${pass} fail=${fail})`;
    }
  };

  let saveTimer = null;
  const scheduleSave = () => {
    setStatus("未保存…");
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(doSave, 500);
  };
  function doSave() {
    state.meta.updated_at = new Date().toISOString();
    const payload = JSON.stringify(state, null, 2);
    try {
      localStorage.setItem(cfg.storageKey, payload);
      setStatus("已存本地缓存 " + new Date().toLocaleTimeString());
    } catch (e) {
      setStatus("本地缓存失败: " + e);
    }
  }

  const getRow = (name) => {
    if (!state.annotations[name]) state.annotations[name] = { choice: "", note: "" };
    return state.annotations[name];
  };

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
  };

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

  const dl = document.getElementById("anno-download");
  if (dl) dl.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (cfg.annotationFile || "annotations.json").split("/").pop();
    a.click();
    URL.revokeObjectURL(url);
    setStatus("已下载 " + new Date().toLocaleTimeString());
  });
  const saveBtn = document.getElementById("anno-save");
  if (saveBtn) saveBtn.addEventListener("click", () => { doSave(); setStatus("已保存 " + new Date().toLocaleTimeString()); });

  const importInput = document.getElementById("anno-import");
  if (importInput) importInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        if (data && data.annotations) {
          state.annotations = Object.assign({}, state.annotations, data.annotations);
          doSave();
          applyState();
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

  const clearBtn = document.getElementById("anno-clear");
  if (clearBtn) clearBtn.addEventListener("click", () => {
    if (!confirm("确定清空当前页面的全部标注吗？此操作不可撤销。")) return;
    state.annotations = {};
    try { localStorage.removeItem(cfg.storageKey); } catch (e) {}
    applyState();
    setStatus("已清空全部标注 " + new Date().toLocaleTimeString());
  });

  const onlyTodo = document.getElementById("anno-only-todo");
  if (onlyTodo) onlyTodo.addEventListener("change", () => {
    const on = onlyTodo.checked;
    document.querySelectorAll("section.row").forEach((sec) => {
      const anno = sec.querySelector(".annotation");
      if (!anno) return;
      const a = state.annotations[anno.dataset.name];
      const done = a && a.choice;
      sec.style.display = on && done ? "none" : "";
    });
  });

  // 快捷键:p = pass, f = fail; 仅在鼠标不在备注框/按钮上时生效
  document.addEventListener("keydown", (event) => {
    if (event.target && (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT")) return;
    if (event.key !== "p" && event.key !== "f") return;
    // 找视口里第一个未被折叠的 row
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

  function init() {
    let loaded = null;
    try { loaded = JSON.parse(localStorage.getItem(cfg.storageKey) || "null"); } catch (e) {}
    if (loaded && loaded.annotations) {
      state.annotations = loaded.annotations;
      if (loaded.meta) state.meta = Object.assign({}, state.meta, loaded.meta);
    }
    applyState();
    setStatus("本地模式:改动存浏览器缓存,记得点\"下载 JSON\"导出");
  }
  init();
})();
</script>
"""


def render_img(src_rel: str, label: str, *, klass: str = "") -> str:
    cls_attr = f' class="{html.escape(klass)}"' if klass else ""
    src_esc = html.escape(src_rel)
    label_esc = html.escape(label)
    return (
        f'<figure{cls_attr}><figcaption>{label_esc}</figcaption>'
        f'<img class="lazy-img zoomable-img" data-src="{src_esc}" '
        f'data-full-src="{src_esc}" alt="{label_esc}" '
        f'title="Click to enlarge" loading="lazy" decoding="async"></figure>'
    )


def render_missing(label: str, path: str) -> str:
    return (
        f'<figure><figcaption>{html.escape(label)}</figcaption>'
        f'<div class="missing">MISSING {html.escape(path)}</div></figure>'
    )


def render_row(global_idx: int, rec: dict, manifest_dir: Path) -> str:
    """一条记录:prompt + ref 图们 + teacher 图。"""
    prompt = rec["prompt"]
    meta = rec["meta"]
    n_refs = meta["n_refs"]
    has_animal = meta["has_animal"]
    subjects = meta["subjects"]
    view_idx = meta["view_idx"]
    seed = meta["seed"]
    template_id = meta["template_id"]
    name = f"task-{global_idx:05d}"

    ref_paths = rec["image_paths"]
    tgt_path = rec["image_tgt_path"]  # 相对 manifest_dir,如 images/000000.jpg

    parts: list[str] = []
    parts.append('<section class="row">')
    # title
    nref_tag = f'<span class="tag tag-{"3ref" if n_refs == 3 else "2ref"}">{n_refs}-ref</span>'
    animal_tag = '<span class="tag tag-animal">animal</span>' if has_animal else '<span class="tag tag-object">object</span>'
    parts.append(
        '<div class="title">'
        '<div class="title-left">'
        f'<span class="idx">#{global_idx:05d}</span>'
        f'{nref_tag}{animal_tag}'
        f'<span>subjects: {html.escape("+".join(subjects))}</span>'
        f'<span>view: {html.escape(",".join(str(v) for v in view_idx))}</span>'
        f'<span>template: {template_id}</span>'
        f'<span>seed: {seed}</span>'
        '</div>'
        '<span class="verdict"></span>'
        '</div>'
    )
    parts.append(f'<div class="prompt"><b>Prompt:</b> {html.escape(prompt)}</div>')
    parts.append('<div class="images">')
    for ref_idx, (ref_rel, subj, vi) in enumerate(zip(ref_paths, subjects, view_idx)):
        label = f"ref{ref_idx+1}: {subj} (view {vi})"
        ref_abs = (manifest_dir / ref_rel).resolve()
        if ref_abs.exists():
            parts.append(render_img(ref_rel, label))
        else:
            parts.append(render_missing(label, ref_rel))
    tgt_abs = (manifest_dir / tgt_path).resolve()
    teacher_label = f"teacher: {tgt_path}"
    if tgt_abs.exists():
        parts.append(render_img(tgt_path, teacher_label, klass="teacher"))
    else:
        parts.append(render_missing(teacher_label, tgt_path))
    parts.append('</div>')

    # 标注面板
    parts.append(
        f'<div class="annotation" data-name="{html.escape(name)}">'
        f'<div class="anno-item">'
        f'<span style="font-weight:700;color:#57606a;">标记:</span>'
        f'<button type="button" class="anno-btn" data-val="pass">pass (主体清晰、构图 OK)</button>'
        f'<button type="button" class="anno-btn" data-val="fail">fail (丢主体/构图崩)</button>'
        f'</div>'
        f'<div class="anno-note-wrap">'
        f'<textarea class="anno-note" placeholder="备注(可选):具体问题如\'cat 丢失\'、\'背景不是 mountain\'等" rows="1"></textarea>'
        f'</div>'
        f'</div>'
    )

    parts.append('</section>')
    return "\n".join(parts)


def write_html(out_path: Path, rows_html: list[str], meta: dict) -> None:
    anno_file = out_path.with_suffix(".annotations.json").name
    anno_storage_key = "anno::" + out_path.name
    with out_path.open("w", encoding="utf-8") as fw:
        fw.write(HEAD_HTML)
        fw.write('<body>\n')
        fw.write(MODAL_HTML)
        fw.write("<header>\n")
        fw.write("<h1>UNO Distill 人工质检面板</h1>\n")
        fw.write(
            '<div class="meta">'
            f'任务数 {meta["n_rows"]} | 来源 {html.escape(meta["source"])} | '
            f'快捷键 <code>p</code>=pass、<code>f</code>=fail(在标注按钮/备注框外时生效)'
            '</div>\n'
        )
        fw.write(
            '<div class="anno-toolbar">\n'
            '  <span id="anno-progress">已标注 0 / 0</span>\n'
            '  <span id="anno-status"></span>\n'
            '  <label class="anno-toggle"><input type="checkbox" id="anno-only-todo"> 只看未标注</label>\n'
            '  <button type="button" id="anno-save">立即保存</button>\n'
            '  <button type="button" id="anno-download">下载 JSON</button>\n'
            '  <label class="anno-toggle" style="border:1px solid #d8dee4;border-radius:6px;padding:5px 10px;">导入 JSON<input type="file" id="anno-import" accept="application/json" style="display:none"></label>\n'
            '  <button type="button" id="anno-clear">清空全部标注</button>\n'
            '</div>\n'
        )
        fw.write("</header>\n")
        for rh in rows_html:
            fw.write(rh)
            fw.write("\n")
        fw.write(VIEWER_SCRIPT)
        anno_config = {
            "annotationFile": anno_file,
            "meta": {
                "source": meta["source"],
                "n_rows": meta["n_rows"],
            },
            "storageKey": anno_storage_key,
        }
        fw.write("<script>window.ANNO_CONFIG = " + json.dumps(anno_config, ensure_ascii=False) + ";</script>\n")
        fw.write(ANNO_SCRIPT)
        fw.write("</body>\n</html>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="为 distill_multiref 生成人工质检 HTML")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="输入 manifest 路径")
    ap.add_argument("--out", type=Path, default=None, help="输出 HTML 路径(默认 <manifest_dir>/inspect.html)")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 条")
    ap.add_argument("--shard", type=int, default=None, help="只取指定 shard 的 1000 条")
    ap.add_argument("--shuffle", action="store_true", help="随机打乱后再截断")
    ap.add_argument("--seed", type=int, default=20260727, help="shuffle 用的种子")
    args = ap.parse_args()

    manifest: list[dict] = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.shard is not None:
        # shard0: 0..999, shard1: 1000..1999 ...
        start = args.shard * 1000
        manifest = manifest[start:start + 1000]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(manifest)
    if args.limit is not None:
        manifest = manifest[:args.limit]

    manifest_dir = args.manifest.parent.resolve()
    out_path = (args.out or manifest_dir / "inspect.html").resolve()
    # 校验 HTML 必须落在 manifest 同目录,这样相对图片路径才是有效的
    if out_path.parent.resolve() != manifest_dir:
        print(f"[ERROR] 输出 HTML 必须在 manifest 同目录下: {manifest_dir}", file=sys.stderr)
        return 1

    source_desc = f"manifest={args.manifest.name}"
    if args.shard is not None:
        source_desc += f" shard={args.shard}"
    if args.shuffle:
        source_desc += f" shuffle(seed={args.seed})"
    if args.limit is not None:
        source_desc += f" limit={args.limit}"

    rows_html = [
        render_row(rec["meta"]["seed"] - 3407000, rec, manifest_dir)
        for rec in manifest
    ]

    meta = {"n_rows": len(manifest), "source": source_desc}
    write_html(out_path, rows_html, meta)
    print(f"[DONE] wrote {out_path}  ({len(manifest)} rows, {out_path.stat().st_size/1024:.1f} KB)")
    print(f"[INFO] 用浏览器打开: file://{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())