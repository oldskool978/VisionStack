// webui/assets/app/geometry.js
import {
  state,
  el,
  gcd,
  normalizeBoxLabel,
  isTypographyBox,
  parseFiniteFloat,
  parseFiniteInt,
   parseOptionalFloat,
  resolveEffectiveBoxIsolation,
  findRefMeta,
  computeActiveMaskSummary
} from "./state.js";
import { renderSpatialLayoutEditor } from "./layout_editor.js";

let lastTrackedCanvasW = 1024;
let lastTrackedCanvasH = 1024;

export function getTrackedCanvasDims() {
  return { width: lastTrackedCanvasW, height: lastTrackedCanvasH };
}

export function setTrackedCanvasDims(width, height) {
  lastTrackedCanvasW = Math.max(256, Math.min(2048, Math.round(width / 16) * 16));
  lastTrackedCanvasH = Math.max(256, Math.min(2048, Math.round(height / 16) * 16));
}

export function snapLatticeFloorHalf(val, mult = 16) {
  return Math.ceil(Number(val) / Number(mult) - 0.5 - 1e-9) * Number(mult);
}

export function scaleBboxToCanvas(bbox, oldW, oldH, newW, newH) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return [0, 0, newH, newW];
  }
  const y0 = Number(bbox[0]);
  const x0 = Number(bbox[1]);
  const y1 = Number(bbox[2]);
  const x1 = Number(bbox[3]);
  const sh = Math.max(16, Math.round(oldH));
  const sw = Math.max(16, Math.round(oldW));
  const h = Math.max(16, Math.round(newH));
  const w = Math.max(16, Math.round(newW));
  if (h === sh && w === sw) {
    return [Math.round(y0), Math.round(x0), Math.round(y1), Math.round(x1)];
  }
  const py0 = Math.max(0, Math.min(sh - 1, Math.round(Math.min(y0, y1))));
  const px0 = Math.max(0, Math.min(sw - 1, Math.round(Math.min(x0, x1))));
  const py1 = Math.max(py0 + 1, Math.min(sh, Math.round(Math.max(y0, y1))));
  const px1 = Math.max(px0 + 1, Math.min(sw, Math.round(Math.max(x0, x1))));
  const uy0 = Math.max(0.0, Math.min(1.0, py0 / sh));
  const ux0 = Math.max(0.0, Math.min(1.0, px0 / sw));
  const uy1 = Math.max(0.0, Math.min(1.0, py1 / sh));
  const ux1 = Math.max(0.0, Math.min(1.0, px1 / sw));

  const ny0 =
    uy0 <= 1e-5
      ? 0
      : Math.max(0, Math.min(h - 16, snapLatticeFloorHalf(uy0 * h, 16)));
  const nx0 =
    ux0 <= 1e-5
      ? 0
      : Math.max(0, Math.min(w - 16, snapLatticeFloorHalf(ux0 * w, 16)));
  const ny1 =
    uy1 >= 1.0 - 1e-5
      ? h
      : Math.max(ny0 + 16, Math.min(h, snapLatticeFloorHalf(uy1 * h, 16)));
  const nx1 =
    ux1 >= 1.0 - 1e-5
      ? w
      : Math.max(nx0 + 16, Math.min(w, snapLatticeFloorHalf(ux1 * w, 16)));
  return [ny0, nx0, ny1, nx1];
}

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

export function bboxToNormalizedJs(bbox, height, width) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return [0.0, 0.0, 1.0, 1.0];
  }
  const h = Math.max(1, Math.round(height));
  const w = Math.max(1, Math.round(width));
  const py0 = Math.max(0, Math.min(h - 1, Math.round(Math.min(bbox[0], bbox[2]))));
  const px0 = Math.max(0, Math.min(w - 1, Math.round(Math.min(bbox[1], bbox[3]))));
  const py1 = Math.max(py0 + 1, Math.min(h, Math.round(Math.max(bbox[0], bbox[2]))));
  const px1 = Math.max(px0 + 1, Math.min(w, Math.round(Math.max(bbox[1], bbox[3]))));
  return [
    Math.max(0.0, Math.min(1.0, py0 / h)),
    Math.max(0.0, Math.min(1.0, px0 / w)),
    Math.max(0.0, Math.min(1.0, py1 / h)),
    Math.max(0.0, Math.min(1.0, px1 / w))
  ];
}

function snapNormBoxToLatentGridJs(normBox, hLat, wLat) {
  if (!normBox || normBox.length !== 4) return null;
  const [uy0, ux0, uy1, ux1] = normBox;
  const y0 = Math.max(0, Math.min(hLat - 1, Math.round(uy0 * hLat)));
  const x0 = Math.max(0, Math.min(wLat - 1, Math.round(ux0 * wLat)));
  const y1 = Math.max(y0 + 1, Math.min(hLat, Math.round(uy1 * hLat)));
  const x1 = Math.max(x0 + 1, Math.min(wLat, Math.round(ux1 * wLat)));
  if (y0 === 0 && x0 === 0 && y1 === hLat && x1 === wLat) {
    return null;
  }
  return [
    Number((y0 / hLat).toFixed(6)),
    Number((x0 / wLat).toFixed(6)),
    Number((y1 / hLat).toFixed(6)),
    Number((x1 / wLat).toFixed(6))
  ];
}

function boxesOverlap2dJs(boxA, boxB) {
  if (!boxA || !boxB) return false;
  const interH = Math.min(boxA[2], boxB[2]) - Math.max(boxA[0], boxB[0]);
  const interW = Math.min(boxA[3], boxB[3]) - Math.max(boxA[1], boxB[1]);
  return interH > 1e-5 && interW > 1e-5;
}

function boxesOverlapCoefficientJs(boxA, boxB) {
  if (!boxA || !boxB) return 0.0;
  const interH = Math.min(boxA[2], boxB[2]) - Math.max(boxA[0], boxB[0]);
  const interW = Math.min(boxA[3], boxB[3]) - Math.max(boxA[1], boxB[1]);
  if (interH <= 1e-6 || interW <= 1e-6) return 0.0;
  const interArea = interH * interW;
  const areaA = Math.max(0.0, boxA[2] - boxA[0]) * Math.max(0.0, boxA[3] - boxA[1]);
  const areaB = Math.max(0.0, boxB[2] - boxB[0]) * Math.max(0.0, boxB[3] - boxB[1]);
  const minArea = Math.min(areaA, areaB);
  if (minArea <= 1e-6) return 0.0;
  return interArea / minArea;
}

function normalizeRoleClassJs(role) {
  const r = String(role || "photo_ref").trim().toLowerCase();
  if (["depth_map", "color_map", "structural", "control"].includes(r)) {
    return "structural";
  }
  return "photo_ref";
}

export function computeSpatioVisualBoxGraph(width, height) {
  const hLat = Math.max(1, Math.floor(height / 16));
  const wLat = Math.max(1, Math.floor(width / 16));
  const globalRole = (el("inputRoleSelect") ? el("inputRoleSelect").value : "photo_ref").toLowerCase();
  const globalSplat = Boolean(el("compositeBoxImagesCheck") && el("compositeBoxImagesCheck").checked);

  const boxGraph = {};
  const refBoxEntries = [];
  const splatBoxes = [];
  const splattedIndices = new Set();

  state.layoutItems.forEach((item, idx) => {
    boxGraph[idx] = {
      peerBoxIndices: [],
      coOpBoxIndices: [],
      carvesSplat: false,
      carvesKv: false,
      isCutout: false
    };
    if (!item || item.enabled === false || !Array.isArray(item.bbox) || item.bbox.length !== 4) {
      return;
    }
    const rawImg = String(item.image_path || item.image || item.vizj_path || "").trim();
    const roleStr = String(item.input_role || globalRole || "photo_ref").toLowerCase();
    const normBox = bboxToNormalizedJs(item.bbox, height, width);
    const snappedBox = snapNormBoxToLatentGridJs(normBox, hLat, wLat);
    const zVal = typeof item.z_order === "number" ? item.z_order : idx + 1;

    if (rawImg && roleStr !== "inpaint_mask") {
      const refMeta = findRefMeta(rawImg);
      if (refMeta && refMeta.has_alpha && roleStr === "photo_ref") {
        boxGraph[idx].isCutout = true;
      }
      const isPasted =
        item.paste_to_canvas !== null && item.paste_to_canvas !== undefined
          ? Boolean(item.paste_to_canvas)
          : (globalSplat && roleStr === "photo_ref");
      if (isPasted) {
        splattedIndices.add(idx);
        splatBoxes.push(normBox);
      }
      refBoxEntries.push({
        boxIdx: idx,
        normBox,
        snappedBox,
        roleClass: normalizeRoleClassJs(roleStr),
        zOrder: zVal
      });
    }
  });

  for (let i = 0; i < refBoxEntries.length; i++) {
    const a = refBoxEntries[i];
    if (!a.snappedBox) continue;
    for (let j = i + 1; j < refBoxEntries.length; j++) {
      const b = refBoxEntries[j];
      if (!b.snappedBox) continue;
      if (boxesOverlapCoefficientJs(a.snappedBox, b.snappedBox) >= 0.25) {
        if (a.roleClass !== b.roleClass) {
          boxGraph[a.boxIdx].coOpBoxIndices.push(b.boxIdx);
          boxGraph[b.boxIdx].coOpBoxIndices.push(a.boxIdx);
        } else {
          boxGraph[a.boxIdx].peerBoxIndices.push(b.boxIdx);
          boxGraph[b.boxIdx].peerBoxIndices.push(a.boxIdx);
        }
      }
    }
  }

  const hasGlobalRefs =
    (state.bindings.reference_image_paths && state.bindings.reference_image_paths.length > 0) ||
    Boolean(state.bindings.control_image_path);

  state.layoutItems.forEach((item, idx) => {
    if (!item || item.enabled === false || splattedIndices.has(idx)) return;
    if (!Array.isArray(item.bbox) || item.bbox.length !== 4) return;
    const rawImg = String(item.image_path || item.image || item.vizj_path || "").trim();
    if (rawImg) return;

    const isText = isTypographyBox(item);
    const roleStr = String(item.input_role || "").trim().toLowerCase();
    const isInp = Boolean(item.inpaint_box) || roleStr === "inpaint_mask";
    const expDn = parseOptionalFloat(item.denoise_strength, 0.01, 1.0);
    if (!isText && !isInp && expDn === null) return;

    const ovrBox = bboxToNormalizedJs(item.bbox, height, width);
    const ovrArea = (ovrBox[2] - ovrBox[0]) * (ovrBox[3] - ovrBox[1]);
    if (ovrArea <= 1e-5) return;

    const childDn = expDn !== null ? expDn : (isInp ? 1.0 : (isText ? 0.65 : 0.75));
    const splatCarve = (isText || isInp || childDn >= 0.55) ? 1.0 : 0.0;

    if (splatCarve > 0.0 && splatBoxes.length > 0) {
      const overlapsSplat = splatBoxes.some((sb) => {
        if (!boxesOverlap2dJs(ovrBox, sb)) return false;
        const sbArea = (sb[2] - sb[0]) * (sb[3] - sb[1]);
        return isText || isInp || expDn !== null || sbArea > ovrArea + 1e-5;
      });
      if (overlapsSplat) {
        boxGraph[idx].carvesSplat = true;
      }
    }

    if (isText || isInp) {
      const expIso = parseOptionalFloat(item.isolate_box_attn, 0.0, 1.0);
      const carveIso = expIso !== null ? expIso : 0.85;
      if (carveIso > 1e-6) {
        const childZ = typeof item.z_order === "number" ? item.z_order : null;
        let overlapsRef = false;
        if (hasGlobalRefs && (isText || (childZ !== null && childZ > 1e-4) || ovrArea < 0.98)) {
          overlapsRef = true;
        }
        for (const rEntry of refBoxEntries) {
          if (boxesOverlap2dJs(ovrBox, rEntry.normBox)) {
            const rArea =
              (rEntry.normBox[2] - rEntry.normBox[0]) *
              (rEntry.normBox[3] - rEntry.normBox[1]);
            if (
              isText ||
              (childZ !== null && childZ > rEntry.zOrder + 1e-4) ||
              rArea > ovrArea + 1e-5
            ) {
              overlapsRef = true;
              break;
            }
          }
        }
        if (overlapsRef) {
          boxGraph[idx].carvesKv = true;
        }
      }
    }
  });

  return boxGraph;
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
  const boxGraph = computeSpatioVisualBoxGraph(canvasW, canvasH);
  hud.innerHTML = "";

  state.layoutItems.forEach((item, idx) => {
    if (item.enabled === false || !Array.isArray(item.bbox) || item.bbox.length !== 4) return;
    const [ymin, xmin, ymax, xmax] = item.bbox;
    const isText = isTypographyBox(item);
    const normLbl = normalizeBoxLabel(item.label || item.category, isText);
    const isChar = normLbl === "character";
    const gInfo = boxGraph[idx] || {};
    const boxDiv = document.createElement("div");
    boxDiv.className = `viewport-hud-box ${isText ? "text-hud" : (isChar ? "char-hud" : "")}`.trim();
    boxDiv.style.top = `${((ymin / canvasH) * 100).toFixed(2)}%`;
    boxDiv.style.left = `${((xmin / canvasW) * 100).toFixed(2)}%`;
    boxDiv.style.width = `${(((xmax - xmin) / canvasW) * 100).toFixed(2)}%`;
    boxDiv.style.height = `${(((ymax - ymin) / canvasH) * 100).toFixed(2)}%`;
    const picTag = boxPicMap[idx] ? ` [Pic ${boxPicMap[idx]}]` : "";
    const carveTag = (gInfo.carvesSplat || gInfo.carvesKv) ? " [Carve]" : "";
    const label = isText ? `T: "${item.text || item.caption || ""}"${carveTag}` : `${normLbl}${picTag}${carveTag}`;
    boxDiv.textContent = `#${idx + 1} ${label}`;
    hud.appendChild(boxDiv);
  });
}

export function syncStageAspectAndGuides() {
  const layoutGeom = computeStageGeometry("spatialStageContainer");
  setTrackedCanvasDims(layoutGeom.canvasW, layoutGeom.canvasH);
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
    for (const b of state.layoutItems) {
      if (Array.isArray(b.bbox) && b.bbox.length === 4) {
        b.bbox = scaleBboxToCanvas(b.bbox, oldW, oldH, cleanW, cleanH);
      }
    }
  }
  setTrackedCanvasDims(cleanW, cleanH);
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