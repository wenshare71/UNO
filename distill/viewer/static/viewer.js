// Image viewer: lazy loading + modal zoom/pan/keyboard navigation.
// Call window.initViewer() after new rows are rendered into the DOM.

const modal = document.getElementById("image-modal");
const modalImg = document.getElementById("image-modal-img");
const modalClose = document.getElementById("image-modal-close");
const modalPrev = document.getElementById("image-modal-prev");
const modalNext = document.getElementById("image-modal-next");
const modalZoomIn = document.getElementById("image-modal-zoom-in");
const modalZoomOut = document.getElementById("image-modal-zoom-out");
const modalZoomReset = document.getElementById("image-modal-zoom-reset");

const MIN_SCALE = 1;
const MAX_SCALE = 8;
let zoomableImages = [];
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

const loadImage = (img) => {
  if (!img.dataset.src) return;
  img.src = img.dataset.src;
  img.removeAttribute("data-src");
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

function initLazyLoading() {
  const lazyImages = document.querySelectorAll("img.lazy-img");
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
}

function initZoomableImages() {
  zoomableImages = Array.from(document.querySelectorAll("img.zoomable-img"));
  zoomableImages.forEach((img, index) => {
    img.addEventListener("click", () => {
      showImage(index);
      modal.classList.add("is-open");
      modal.setAttribute("aria-hidden", "false");
    });
  });
}

function initViewer() {
  initLazyLoading();
  initZoomableImages();
}

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

window.initViewer = initViewer;
