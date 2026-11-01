import {
  state,
  el,
  gcd,
  normalizeBoxLabel,
  isTypographyBox,
  parseFiniteFloat,
  parseFiniteInt,
  findRefMeta,
  computeActiveMaskSummary
} from "./state.js";
import { renderSpatialLayoutEditor } from "./layout_editor.js";

export function snapCoord(val, maxDim, bypassSnap = false) {
  const clamped = Math.max(0, Math.min(maxDim, Math.round(val)));
  const snapEnabled = el("snapGridCheck") ? el("snapGridCheck").checked : true;
  if (!snapEnabled || bypassSnap) return clamped;
  return Math.max(0, Math.min(maxDim, Math.round(clamped / 16) * 16));
}

export function pixelToAbsCentralized(ymin, xmin, ymax, xmax, height, width) {
  const hInt = Math.round(height);
  const wInt = Math.round(width);
  const cy = Math.floor(hInt / 2);
  const cx = Math.floor(wInt / 2);
  const maxY = hInt - cy;
  const maxX = wInt - cx;
  const effYmax = Math.round(ymax) === hInt - 1 ? hInt : ymax;
  const effXmax = Math.round(xmax) === wInt - 1 ? wInt : xmax;
  return [
    Math.max(-cy, Math.min(maxY, Math.round(ymin - cy))),
    Math.max(-cx, Math.min(maxX, Math.round(xmin - cx))),
    Math.max(-cy, Math.min(maxY, Math.round(effYmax - cy))),
    Math.max(-cx, Math.min(maxX, Math.round(effXmax - cx)))
  ];
}

export function canonicalBoxSortKey(item, height, width) {
  const bbox = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox : [0, 0, height, width];
  const hDenom = Math.max(1, height);
  const wDenom = Math.max(1, width);
  const py0 = Math.max(0, Math.min(height - 1, Math.round(Math.min(bbox[0], bbox[2]))));
  const px0 = Math.max(0, Math.min(width - 1, Math.round(Math.min(bbox[1], bbox[3]))));
  const py1 = Math.max(py0 + 1, Math.min(height, Math.round(Math.max(bbox[0], bbox[2]))));
  const px1 = Math.max(px0 + 1, Math.min(width, Math.round(Math.max(bbox[1], bbox[3]))));
  const uy0 = Math.max(0.0, Math.min(1.0, py0 / hDenom));
  const ux0 = Math.max(0.0, Math.min(1.0, px0 / wDenom));
  const uy1 = Math.max(0.0, Math.min(1.0, py1 / hDenom));
  const ux1 = Math.max(0.0, Math.min(1.0, px1 / wDenom));
  const area = Number(((uy1 - uy0) * (ux1 - ux0)).toFixed(6));
  const uyc = Number((0.5 * (uy0 + uy1)).toFixed(6));
  const uxc = Number((0.5 * (ux0 + ux1)).toFixed(6));
  const rawTxt = typeof item.text === "string" ? item.text.trim() : "";
  const hasTxt = Boolean(rawTxt);
  const lbl = normalizeBoxLabel(item.label || item.category, hasTxt);
  const txtStr = hasTxt ? rawTxt : "";
  const capStr = String(item.caption || item.description || "").trim();
  const imgP = String(item.image_path || item.image || item.vizj_path || "").trim();
  return [-area, uyc, uxc, Number(uy0.toFixed(6)), Number(ux0.toFixed(6)), lbl, txtStr, capStr, imgP];
}

export function compareCanonicalKeys(ka, kb) {
  for (let i = 0; i < ka.length; i++) {
    if (ka[i] < kb[i]) return -1;
    if (ka[i] > kb[i]) return 1;
  }
  return 0;
}

export function getCanonicalEnabledBoxIndices(height, width) {
  const keyed = [];
  state.layoutItems.forEach((item, idx) => {
    if (!item || item.enabled === false) return;
    keyed.push({
      idx,
      key: canonicalBoxSortKey(item, height, width)
    });
  });
  keyed.sort((a, b) => compareCanonicalKeys(a.key, b.key));
  return keyed.map((entry) => entry.idx);
}

export function computeBoxPictureBindings() {
  const width = parseFiniteInt(el("widthInput") ? el("widthInput").value : 1024, 1024, 256, 2048);
  const height = parseFiniteInt(el("heightInput") ? el("heightInput").value : 1024, 1024, 256, 2048);
  const globalRole = (el("inputRoleSelect") ? el("inputRoleSelect").value : "photo_ref").toLowerCase();
  const editFormat = (el("editFormatSelect") ? el("editFormatSelect").value : "standard").toLowerCase();
  const globalEmitsVit = globalRole === "photo_ref" && editFormat !== "moss_image_edit";

  let vitPicCounter = globalEmitsVit ? state.bindings.reference_image_paths.length : 0;
  let ditRefCounter = state.bindings.reference_image_paths.length;
  const boxPicMap = {};
  const boxDitMap = {};

  const canonicalIndices = getCanonicalEnabledBoxIndices(height, width);
  for (const idx of canonicalIndices) {
    const b = state.layoutItems[idx];
    if (!b || b.enabled === false) continue;
    if (b.image_path && String(b.image_path).trim() !== "") {
      ditRefCounter += 1;
      boxDitMap[idx] = ditRefCounter;
      const effRole = (b.input_role || globalRole || "photo_ref").toLowerCase();
      if (effRole === "photo_ref") {
        vitPicCounter += 1;
        boxPicMap[idx] = vitPicCounter;
      }
    }
  }
  return { boxPicMap, boxDitMap, canonicalIndices };
}

export function resolveContainerBounds(containerId, fallbackSize = 512) {
  const node = el(containerId);
  if (!node) return { availW: fallbackSize, availH: fallbackSize };
  const rect = node.getBoundingClientRect();
  const availW = rect.width > 64 ? rect.width - 40 : fallbackSize;
  const availH = rect.height > 64 ? rect.height - 40 : fallbackSize;
  return { availW: Math.max(240, availW), availH: Math.max(240, availH) };
}

export function computeStageGeometry(containerId = "spatialStageContainer") {
  const canvasW = parseFiniteInt(el("widthInput") ? el("widthInput").value : 1024, 1024, 256, 2048);
  const canvasH = parseFiniteInt(el("heightInput") ? el("heightInput").value : 1024, 1024, 256, 2048);
  const { availW, availH } = resolveContainerBounds(containerId, 560);
  const fitScale = Math.min(availW / canvasW, availH / canvasH);
  const stageW = Math.max(192, Math.round(canvasW * fitScale));
  const stageH = Math.max(192, Math.round(canvasH * fitScale));

  const div = gcd(canvasW / 16, canvasH / 16);
  const arW = Math.round((canvasW / 16) / div);
  const arH = Math.round((canvasH / 16) / div);
  const badge = el("stageAspectBadge");
  if (badge) {
    badge.textContent = `${canvasW}x${canvasH} (${arW}:${arH})`;
  }
  const hudReadout = el("spatialHudReadout");
  if (hudReadout) {
    hudReadout.textContent = `Latent Grid: ${canvasW / 16}x${canvasH / 16} (16px stride)`;
  }

  return {
    canvasW,
    canvasH,
    stageW,
    stageH,
    scaleX: stageW / canvasW,
    scaleY: stageH / canvasH
  };
}

export function framingModeToCssObjectFit(framingMode) {
  const fm = String(framingMode || "inscribed").toLowerCase();
  if (fm === "contain" || fm === "never_crop") return "contain";
  if (fm === "stretch") return "fill";
  return "cover";
}

export function resolveUnderlayUrl(modeOverride = null) {
  const rawMode = modeOverride || (el("layoutUnderlaySelect") ? el("layoutUnderlaySelect").value : "auto");
  const mode = String(rawMode || "auto").toLowerCase();
  if (mode === "none") return null;

  if (mode === "master") {
    const masterImg = el("masterOutputImage");
    if (masterImg && !masterImg.hidden) {
      const masterSrc = masterImg.getAttribute("src");
      if (masterSrc && masterSrc.trim()) {
        return masterSrc;
      }
    }
    return null;
  }

  let candidates = [];
  if (mode === "init") {
    candidates = [state.bindings.init_image_path];
  } else if (mode === "control") {
    candidates = [state.bindings.control_image_path];
  } else if (mode === "init_ctrl") {
    candidates = [state.bindings.init_image_path, state.bindings.control_image_path];
  } else if (mode === "ref") {
    candidates = [...(state.bindings.reference_image_paths || [])];
  } else {
    candidates = [
      state.bindings.init_image_path,
      state.bindings.control_image_path,
      ...(state.bindings.reference_image_paths || [])
    ];
  }

  for (const refId of candidates) {
    if (!refId) continue;
    const meta = findRefMeta(refId);
    if (meta && !meta.is_vizj && meta.url) {
      return meta.url;
    }
  }
  return null;
}

export function syncUnderlayLayers() {
  const mode = el("layoutUnderlaySelect") ? el("layoutUnderlaySelect").value : "auto";
  const url = resolveUnderlayUrl(mode);
  const compareUrl = mode === "master" ? resolveUnderlayUrl("auto") : url;
  const globalFraming = el("framingModeSelect") ? el("framingModeSelect").value : "inscribed";
  const fitCss = mode === "master" ? "contain" : framingModeToCssObjectFit(globalFraming);
  const compareFitCss = framingModeToCssObjectFit(globalFraming);

  const layoutImg = el("spatialUnderlayImg");
  const inpaintImg = el("inpaintUnderlayImg");
  const viewportFrame = el("viewportStageFrame");
  const viewportUnderlay = el("viewportUnderlayImage");
  const compareBtn = el("toggleCompareSplitBtn");
  const compareSlider = el("viewportCompareSlider");
  const masterImg = el("masterOutputImage");

  const layoutOp = parseFiniteFloat(
    el("layoutUnderlayOpacity") ? el("layoutUnderlayOpacity").value : 0.45,
    0.45,
    0.0,
    1.0
  );
  const inpaintOp = parseFiniteFloat(
    el("inpaintUnderlayOpacity") ? el("inpaintUnderlayOpacity").value : 0.55,
    0.55,
    0.0,
    1.0
  );

  if (layoutImg) {
    if (url && layoutOp > 0.005) {
      if (layoutImg.getAttribute("src") !== url) {
        layoutImg.src = url;
      }
      layoutImg.style.objectFit = fitCss;
      layoutImg.style.opacity = String(layoutOp);
      layoutImg.hidden = false;
    } else {
      layoutImg.hidden = true;
    }
  }

  if (inpaintImg) {
    if (url && inpaintOp > 0.005) {
      if (inpaintImg.getAttribute("src") !== url) {
        inpaintImg.src = url;
      }
      inpaintImg.style.objectFit = fitCss;
      inpaintImg.style.opacity = String(inpaintOp);
      inpaintImg.hidden = false;
    } else {
      inpaintImg.hidden = true;
    }
  }

  if (compareBtn) {
    const hasMaster = Boolean(masterImg && !masterImg.hidden && masterImg.getAttribute("src"));
    compareBtn.hidden = !(hasMaster && compareUrl);
    if (!compareUrl && state.showCompareSplit) {
      state.showCompareSplit = false;
      compareBtn.classList.remove("active");
    }
  }

  if (viewportUnderlay && masterImg && compareSlider) {
    if (state.showCompareSplit && compareUrl && !masterImg.hidden) {
      if (viewportUnderlay.getAttribute("src") !== compareUrl) {
        viewportUnderlay.src = compareUrl;
      }
      viewportUnderlay.style.objectFit = compareFitCss;
      viewportUnderlay.hidden = false;
      compareSlider.hidden = false;
      const pct = parseFiniteFloat(compareSlider.value, 50.0, 0.0, 100.0);
      masterImg.style.clipPath = `inset(0 0 0 ${pct}%)`;
      if (viewportFrame) {
        viewportFrame.classList.add("compare-split-active");
        viewportFrame.style.setProperty("--split-pct", `${pct}%`);
      }
    } else {
      viewportUnderlay.hidden = true;
      compareSlider.hidden = true;
      masterImg.style.clipPath = "";
      if (viewportFrame) {
        viewportFrame.classList.remove("compare-split-active");
        viewportFrame.style.removeProperty("--split-pct");
      }
    }
  }
}

export function renderBindingsCard() {
  const bInit = el("boundInitLabel");
  const bMask = el("boundMaskLabel");
  const bCtrl = el("boundCtrlLabel");
  const bRefs = el("boundRefsLabel");
  if (bInit) bInit.textContent = state.bindings.init_image_path || "<None>";
  if (bMask) bMask.textContent = state.bindings.inpaint_mask_path || "<None>";
  if (bCtrl) bCtrl.textContent = state.bindings.control_image_path || "<None>";
  if (bRefs) {
    bRefs.textContent = state.bindings.reference_image_paths.length > 0
      ? state.bindings.reference_image_paths.join(", ")
      : "<None>";
  }
  const maskStateEl = el("activeMaskStateLabel");
  if (maskStateEl) {
    maskStateEl.textContent = computeActiveMaskSummary();
  }
  syncUnderlayLayers();
}

export function renderInpaintBoxGuides(geomInput) {
  const overlay = el("inpaintBoxesOverlay");
  if (!overlay) return;
  const geom = geomInput || computeStageGeometry("inpaintStageWrap");
  overlay.innerHTML = "";
  state.layoutItems.forEach((item, idx) => {
    if (item.enabled === false || !Array.isArray(item.bbox) || item.bbox.length !== 4) return;
    const [ymin, xmin, ymax, xmax] = item.bbox;
    const isText = isTypographyBox(item);
    const normLbl = normalizeBoxLabel(item.label || item.category, isText);
    const isChar = normLbl === "character";
    const guide = document.createElement("div");
    guide.className = `inpaint-box-guide ${isText ? "text-guide" : (isChar ? "char-guide" : "")}`.trim();
    guide.style.top = `${Math.round(ymin * geom.scaleY)}px`;
    guide.style.left = `${Math.round(xmin * geom.scaleX)}px`;
    guide.style.width = `${Math.max(12, Math.round((xmax - xmin) * geom.scaleX))}px`;
    guide.style.height = `${Math.max(12, Math.round((ymax - ymin) * geom.scaleY))}px`;
    guide.textContent = `#${idx + 1} ${normLbl}`;
    overlay.appendChild(guide);
  });
}

export function renderViewportBoxesHud() {
  const hud = el("viewportBoxesHud");
  if (!hud) return;
  hud.hidden = !state.showViewportHud;
  if (!state.showViewportHud) return;

  const canvasW = parseFiniteInt(el("widthInput") ? el("widthInput").value : 1024, 1024, 256, 2048);
  const canvasH = parseFiniteInt(el("heightInput") ? el("heightInput").value : 1024, 1024, 256, 2048);
  const { boxPicMap } = computeBoxPictureBindings();

  hud.innerHTML = "";
  state.layoutItems.forEach((item, idx) => {
    if (item.enabled === false || !Array.isArray(item.bbox) || item.bbox.length !== 4) return;
    const [ymin, xmin, ymax, xmax] = item.bbox;
    const isText = isTypographyBox(item);
    const normLbl = normalizeBoxLabel(item.label || item.category, isText);
    const isChar = normLbl === "character";
    const boxDiv = document.createElement("div");
    boxDiv.className = `viewport-hud-box ${isText ? "text-hud" : (isChar ? "char-hud" : "")}`.trim();
    boxDiv.style.top = `${((ymin / canvasH) * 100).toFixed(2)}%`;
    boxDiv.style.left = `${((xmin / canvasW) * 100).toFixed(2)}%`;
    boxDiv.style.width = `${(((xmax - xmin) / canvasW) * 100).toFixed(2)}%`;
    boxDiv.style.height = `${(((ymax - ymin) / canvasH) * 100).toFixed(2)}%`;
    const picTag = boxPicMap[idx] ? ` [Pic ${boxPicMap[idx]}]` : "";
    const label = isText ? `T: "${item.text || item.caption || ""}"` : `${normLbl}${picTag}`;
    boxDiv.textContent = `#${idx + 1} ${label}`;
    hud.appendChild(boxDiv);
  });
}

export function syncStageAspectAndGuides() {
  const layoutGeom = computeStageGeometry("spatialStageContainer");
  const spatialStage = el("spatialStage");
  if (spatialStage) {
    spatialStage.style.width = `${layoutGeom.stageW}px`;
    spatialStage.style.height = `${layoutGeom.stageH}px`;
    const cellPxX = Math.max(4, layoutGeom.stageW / (layoutGeom.canvasW / 16));
    const cellPxY = Math.max(4, layoutGeom.stageH / (layoutGeom.canvasH / 16));
    spatialStage.style.backgroundSize = `${cellPxX}px ${cellPxY}px`;
  }

  const inpaintGeom = computeStageGeometry("inpaintStageWrap");
  const inpaintContainer = el("inpaintStageContainer");
  const inpaintCanvas = el("inpaintCanvas");
  if (inpaintContainer && inpaintCanvas) {
    inpaintContainer.style.width = `${inpaintGeom.stageW}px`;
    inpaintContainer.style.height = `${inpaintGeom.stageH}px`;
    if (inpaintCanvas.width !== inpaintGeom.canvasW || inpaintCanvas.height !== inpaintGeom.canvasH) {
      const tmp = document.createElement("canvas");
      tmp.width = inpaintCanvas.width;
      tmp.height = inpaintCanvas.height;
      tmp.getContext("2d").drawImage(inpaintCanvas, 0, 0);
      inpaintCanvas.width = inpaintGeom.canvasW;
      inpaintCanvas.height = inpaintGeom.canvasH;
      const ctx = inpaintCanvas.getContext("2d");
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, inpaintGeom.canvasW, inpaintGeom.canvasH);
      ctx.drawImage(tmp, 0, 0, inpaintGeom.canvasW, inpaintGeom.canvasH);
    }
  }
  renderInpaintBoxGuides(inpaintGeom);
  renderViewportBoxesHud();
}

export function applyCanvasResolutionChange(oldW, oldH, newW, newH) {
  const cleanW = Math.max(256, Math.min(2048, Math.round(newW / 16) * 16));
  const cleanH = Math.max(256, Math.min(2048, Math.round(newH / 16) * 16));
  el("widthInput").value = cleanW;
  el("heightInput").value = cleanH;

  if (oldW > 0 && oldH > 0 && (oldW !== cleanW || oldH !== cleanH)) {
    const sx = cleanW / oldW;
    const sy = cleanH / oldH;
    for (const b of state.layoutItems) {
      if (Array.isArray(b.bbox) && b.bbox.length === 4) {
        const ny0 = snapCoord(b.bbox[0] * sy, cleanH - 16);
        const nx0 = snapCoord(b.bbox[1] * sx, cleanW - 16);
        const ny1 = Math.max(ny0 + 16, snapCoord(b.bbox[2] * sy, cleanH));
        const nx1 = Math.max(nx0 + 16, snapCoord(b.bbox[3] * sx, cleanW));
        b.bbox = [ny0, nx0, ny1, nx1];
      }
    }
  }
  syncStageAspectAndGuides();
  renderSpatialLayoutEditor();
}

export function matchPrimaryAssetAspect() {
  const targetRefId =
    state.bindings.init_image_path ||
    state.bindings.control_image_path ||
    (state.bindings.reference_image_paths && state.bindings.reference_image_paths[0]) ||
    null;
  if (!targetRefId) return;
  const meta = findRefMeta(targetRefId);
  if (!meta || !meta.width || !meta.height) return;

  const oldW = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
  const oldH = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
  const targetArea = 1024 * 1024;
  const scale = Math.sqrt(targetArea / Math.max(1, meta.width * meta.height));
  const newW = Math.max(256, Math.min(2048, Math.round((meta.width * scale) / 16) * 16));
  const newH = Math.max(256, Math.min(2048, Math.round((meta.height * scale) / 16) * 16));
  applyCanvasResolutionChange(oldW, oldH, newW, newH);
}