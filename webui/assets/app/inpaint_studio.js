// webui/assets/app/inpaint_studio.js
import {
  state,
  el,
  MAX_MASK_HISTORY,
  parseFiniteFloat,
  parseFiniteInt,
  showStudioToast
} from "./state.js";
import { uploadDataUrlToVault } from "./api.js";
import { renderBindingsCard } from "./geometry.js";
import { renderReferenceVault } from "./vault_projects.js";

export function pushMaskHistorySnapshot() {
  const canvas = el("inpaintCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  state.maskUndoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  if (state.maskUndoStack.length > MAX_MASK_HISTORY) {
    state.maskUndoStack.shift();
  }
  state.maskRedoStack.length = 0;
  updateMaskHistoryButtons();
}

export function updateMaskHistoryButtons() {
  const undoBtn = el("undoMaskBtn");
  const redoBtn = el("redoMaskBtn");
  if (undoBtn) undoBtn.disabled = state.maskUndoStack.length === 0;
  if (redoBtn) redoBtn.disabled = state.maskRedoStack.length === 0;
}

export function undoMaskStroke() {
  const canvas = el("inpaintCanvas");
  if (!canvas || state.maskUndoStack.length === 0) return;
  const ctx = canvas.getContext("2d");
  state.maskRedoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  const prev = state.maskUndoStack.pop();
  if (prev.width !== canvas.width || prev.height !== canvas.height) {
    canvas.width = prev.width;
    canvas.height = prev.height;
  }
  ctx.putImageData(prev, 0, 0);
  updateMaskHistoryButtons();
}

export function redoMaskStroke() {
  const canvas = el("inpaintCanvas");
  if (!canvas || state.maskRedoStack.length === 0) return;
  const ctx = canvas.getContext("2d");
  state.maskUndoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  const next = state.maskRedoStack.pop();
  if (next.width !== canvas.width || next.height !== canvas.height) {
    canvas.width = next.width;
    canvas.height = next.height;
  }
  ctx.putImageData(next, 0, 0);
  updateMaskHistoryButtons();
}

export function hasInpaintCanvasContent() {
  const canvas = el("inpaintCanvas");
  if (!canvas) return false;
  const ctx = canvas.getContext("2d");
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < data.length; i += 32) {
    if (data[i] > 4) return true;
  }
  return false;
}

export function loadCanvasFromDataUrl(dataUrl) {
  const canvas = el("inpaintCanvas");
  if (!canvas || !dataUrl) return;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    pushMaskHistorySnapshot();
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = dataUrl;
}

export function projectBoxesToInpaintCanvas() {
  const canvas = el("inpaintCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const featherLat = parseFiniteFloat(
    el("featherInput") ? el("featherInput").value : 1.25,
    1.25,
    0.0,
    6.0
  );
  const featherPx = featherLat * 16.0;

  const enabledBoxes = state.layoutItems.filter(
    (b) => b.enabled !== false && Array.isArray(b.bbox) && b.bbox.length === 4
  );
  const flaggedBoxes = enabledBoxes.filter((b) => Boolean(b.inpaint_box) || Boolean(b.paste_to_canvas));
  const activeBoxes = flaggedBoxes.length > 0 ? flaggedBoxes : enabledBoxes;
  if (!activeBoxes.length) return;

  pushMaskHistorySnapshot();
  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;

  for (let py = 0; py < h; py++) {
    const yy = py + 0.5;
    for (let px = 0; px < w; px++) {
      const xx = px + 0.5;
      let maxVal = 0.0;
      for (let k = 0; k < activeBoxes.length; k++) {
        const [ly0, lx0, ly1, lx1] = activeBoxes[k].bbox;
        const dy = Math.max(0.0, ly0 - yy, yy - ly1);
        const dx = Math.max(0.0, lx0 - xx, xx - lx1);
        const dist = Math.sqrt(dy * dy + dx * dx);
        if (dist <= 0.0) {
          maxVal = 1.0;
          break;
        } else if (featherPx > 0.0 && dist < featherPx) {
          const boxM = 0.5 * (1.0 + Math.cos(Math.PI * (dist / featherPx)));
          if (boxM > maxVal) maxVal = boxM;
        }
      }
      const offset = (py * w + px) * 4;
      const curV = data[offset] / 255.0;
      const outU8 = Math.round(Math.max(curV, maxVal) * 255.0);
      data[offset] = outU8;
      data[offset + 1] = outU8;
      data[offset + 2] = outU8;
      data[offset + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

export function invertInpaintCanvasPixels() {
  const canvas = el("inpaintCanvas");
  if (!canvas) return;
  pushMaskHistorySnapshot();
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const inv = 255 - data[i];
    data[i] = inv;
    data[i + 1] = inv;
    data[i + 2] = inv;
    data[i + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

export function stampRadialDab(ctx, x, y, radius, hardness, isPaint) {
  const innerR = Math.max(0.0, radius * Math.min(0.99, hardness));
  const grad = ctx.createRadialGradient(x, y, innerR, x, y, radius);
  const rgb = isPaint ? "255, 255, 255" : "0, 0, 0";
  grad.addColorStop(0.0, `rgba(${rgb}, 1.0)`);
  grad.addColorStop(1.0, `rgba(${rgb}, 0.0)`);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

export function initInpaintCanvas() {
  const container = el("inpaintStageContainer");
  const canvas = el("inpaintCanvas");
  const cursorEl = el("inpaintBrushCursor");
  if (!container || !canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  function getCanvasCoords(ev) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (ev.clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
      y: (ev.clientY - rect.top) * (canvas.height / Math.max(1, rect.height)),
      localX: ev.clientX - rect.left,
      localY: ev.clientY - rect.top,
      scale: rect.width / Math.max(1, canvas.width)
    };
  }

  function updateCursorRing(ev) {
    if (!cursorEl) return;
    const pt = getCanvasCoords(ev);
    const brushDiam = parseFiniteInt(el("brushSizeInput").value, 56, 6, 220);
    const dispDiam = Math.max(4, brushDiam * pt.scale);
    cursorEl.style.width = `${dispDiam}px`;
    cursorEl.style.height = `${dispDiam}px`;
    cursorEl.style.left = `${pt.localX}px`;
    cursorEl.style.top = `${pt.localY}px`;
    cursorEl.hidden = false;
  }

  function paintSegment(ev) {
    if (!state.isPainting) return;
    const pt = getCanvasCoords(ev);
    const diam = parseFiniteInt(el("brushSizeInput").value, 56, 6, 220);
    const radius = Math.max(2, diam * 0.5);
    const hardness = parseFiniteFloat(el("brushHardnessInput").value, 0.8, 0.05, 1.0);
    const isPaint = state.brushMode === "paint";

    if (!state.lastPaintPt) {
      stampRadialDab(ctx, pt.x, pt.y, radius, hardness, isPaint);
      state.lastPaintPt = { x: pt.x, y: pt.y };
      return;
    }

    const dx = pt.x - state.lastPaintPt.x;
    const dy = pt.y - state.lastPaintPt.y;
    const dist = Math.hypot(dx, dy);
    const step = Math.max(1.5, radius * 0.22);
    const steps = Math.max(1, Math.ceil(dist / step));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      stampRadialDab(
        ctx,
        state.lastPaintPt.x + dx * t,
        state.lastPaintPt.y + dy * t,
        radius,
        hardness,
        isPaint
      );
    }
    state.lastPaintPt = { x: pt.x, y: pt.y };
  }

  container.addEventListener("mouseenter", updateCursorRing);
  container.addEventListener("mousemove", (ev) => {
    updateCursorRing(ev);
    paintSegment(ev);
  });
  container.addEventListener("mouseleave", () => {
    if (cursorEl) cursorEl.hidden = true;
  });

  canvas.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    pushMaskHistorySnapshot();
    state.isPainting = true;
    state.lastPaintPt = null;
    paintSegment(ev);
  });
  window.addEventListener("mouseup", () => {
    state.isPainting = false;
    state.lastPaintPt = null;
  });

  el("brushSizeInput").addEventListener("input", () => {
    el("brushSizeVal").textContent = `${el("brushSizeInput").value}px`;
  });
  el("brushHardnessInput").addEventListener("input", () => {
    const hVal = parseFiniteFloat(el("brushHardnessInput").value, 0.8, 0.05, 1.0);
    el("brushHardnessVal").textContent = `${Math.round(hVal * 100)}%`;
  });

  el("brushModePaintBtn").addEventListener("click", () => {
    state.brushMode = "paint";
    el("brushModePaintBtn").classList.add("active");
    el("brushModeEraseBtn").classList.remove("active");
  });
  el("brushModeEraseBtn").addEventListener("click", () => {
    state.brushMode = "erase";
    el("brushModeEraseBtn").classList.add("active");
    el("brushModePaintBtn").classList.remove("active");
  });

  el("undoMaskBtn").addEventListener("click", undoMaskStroke);
  el("redoMaskBtn").addEventListener("click", redoMaskStroke);
  el("projectBoxesToMaskBtn").addEventListener("click", projectBoxesToInpaintCanvas);
  el("invertMaskCanvasBtn").addEventListener("click", invertInpaintCanvasPixels);

  el("clearMaskCanvasBtn").addEventListener("click", () => {
    pushMaskHistorySnapshot();
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });

  el("commitMaskToVaultBtn").addEventListener("click", async () => {
    try {
      const dataUrl = canvas.toDataURL("image/png");
      const uploaded = await uploadDataUrlToVault("painted_inpaint_mask.png", "inpaint_mask", dataUrl);
      state.bindings.inpaint_mask_path = uploaded.ref_id;
      renderBindingsCard();
      renderReferenceVault();
      showStudioToast(`Inpaint mask staged & bound: ${uploaded.ref_id}`, "success");
    } catch (err) {
      showStudioToast(`Failed staging mask: ${err.message}`, "error");
    }
  });
}