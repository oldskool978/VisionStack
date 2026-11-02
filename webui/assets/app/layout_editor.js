// webui/assets/app/layout_editor.js
import {
  state,
  el,
  normalizeBoxLabel,
  isTypographyBox,
  parseOptionalFloat,
  parseFiniteFloat,
  parseFiniteInt,
  parseTriStateBool,
  resolveEffectiveBoxIsolation,
  ensureBoxZOrders,
  findRefMeta,
  resolveThinkingTraceFromDeck
} from "./state.js";
import { apiFetch } from "./api.js";
import {
  snapCoord,
  pixelToAbsCentralized,
  getCanonicalEnabledBoxIndices,
  computeBoxPictureBindings,
  computeSpatioVisualBoxGraph,
  computeStageGeometry,
  framingModeToCssObjectFit,
  syncStageAspectAndGuides,
  syncUnderlayLayers,
  renderBindingsCard,
  renderInpaintBoxGuides,
  renderViewportBoxesHud
} from "./geometry.js";

let previewDebounceTimer = null;
let previewRequestSeq = 0;

function escapeHtmlAttr(raw) {
  return String(raw ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function getReadingOrderTextsJs(height, width) {
  const keyed = [];
  const hDenom = Math.max(1, height);
  const wDenom = Math.max(1, width);
  state.layoutItems.forEach((item, idx) => {
    if (!item || item.enabled === false) return;
    if (!isTypographyBox(item)) return;
    const rawTxt = typeof item.text === "string" ? item.text.trim() : "";
    const rawCap = typeof (item.caption || item.description) === "string"
      ? String(item.caption || item.description).trim()
      : "";
    const txtVal = rawTxt || rawCap;
    if (!txtVal) return;

    const bbox = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox : [0, 0, height, width];
    const py0 = Math.max(0, Math.min(height - 1, Math.round(Math.min(bbox[0], bbox[2]))));
    const px0 = Math.max(0, Math.min(width - 1, Math.round(Math.min(bbox[1], bbox[3]))));
    const py1 = Math.max(py0 + 1, Math.min(height, Math.round(Math.max(bbox[0], bbox[2]))));
    const px1 = Math.max(px0 + 1, Math.min(width, Math.round(Math.max(bbox[1], bbox[3]))));
    const uy0 = Number((py0 / hDenom).toFixed(6));
    const ux0 = Number((px0 / wDenom).toFixed(6));
    const uy1 = Number((py1 / hDenom).toFixed(6));
    const ux1 = Number((px1 / wDenom).toFixed(6));
    keyed.push({
      sortKey: [uy0, ux0, uy1, ux1, idx],
      text: txtVal
    });
  });
  keyed.sort((a, b) => {
    for (let i = 0; i < a.sortKey.length; i++) {
      if (a.sortKey[i] < b.sortKey[i]) return -1;
      if (a.sortKey[i] > b.sortKey[i]) return 1;
    }
    return 0;
  });
  return keyed.map((entry) => entry.text);
}

export function reconcileSummaryQuotesJs(caption, activeTexts) {
  const cleanCap = (caption || "").trim();
  if (!cleanCap || !activeTexts || !activeTexts.length) return cleanCap;
  const quoteRegex = /"([^"\r\n]+)"/g;
  const matches = Array.from(cleanCap.matchAll(quoteRegex));
  if (!matches.length) return cleanCap;

  if (matches.length === activeTexts.length) {
    let out = "";
    let lastIdx = 0;
    matches.forEach((m, i) => {
      out += cleanCap.slice(lastIdx, m.index);
      out += `"${activeTexts[i]}"`;
      lastIdx = m.index + m[0].length;
    });
    out += cleanCap.slice(lastIdx);
    return out;
  }

  const defaultQuote = '"Remember to test SFT & Base permutations @ 1024x1024"';
  if (cleanCap.includes(defaultQuote)) {
    return cleanCap.replace(defaultQuote, `"${activeTexts[0]}"`);
  }
  return cleanCap;
}

export function syncTextEditToPrompt(oldText, newText) {
  if (!el("syncQuotesCheck") || !el("syncQuotesCheck").checked) return;
  if (!newText || !newText.trim()) return;
  const targetOld = (oldText || "").trim();
  const width = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
  const height = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
  const activeTexts = getReadingOrderTextsJs(height, width);

  for (const inputId of ["promptInput", "layoutSummaryInput"]) {
    const field = el(inputId);
    if (!field || !field.value) continue;
    if (targetOld && field.value.includes(`"${targetOld}"`)) {
      field.value = field.value.replace(`"${targetOld}"`, `"${newText}"`);
    } else {
      field.value = reconcileSummaryQuotesJs(field.value, activeTexts);
    }
  }
}

function compileAutoSpatialTraceJs(width, height, reconciledCaption, canonicalIndices, activeTexts, boxPicMap, boxDitMap) {
  const cy = Math.floor(height / 2);
  const cx = Math.floor(width / 2);
  const sceneIntent = reconciledCaption.replace(/\s*"[^"\r\n]+"/g, "").replace(/^[\s:;-]+|[\s:;-]+$/g, "") || reconciledCaption;
  const lines = [
    `1. Canvas Geometry: ${width}x${height} px (latent grid ${Math.floor(width / 16)}x${Math.floor(height / 16)}; abs_centralized yxyx range [${-cy}, ${-cx}, ${height - cy}, ${width - cx}]).`,
    `2. Scene Intent: ${sceneIntent}`
  ];

  const globalRole = (el("inputRoleSelect") ? el("inputRoleSelect").value : "photo_ref").toLowerCase();
  const globalFraming = (el("framingModeSelect") ? el("framingModeSelect").value : "inscribed").toLowerCase();
  const editFormat = (el("editFormatSelect") ? el("editFormatSelect").value : "standard").toLowerCase();
  const globalEmitsVit = globalRole === "photo_ref" && editFormat !== "moss_image_edit";
  const globalSplat = Boolean(el("compositeBoxImagesCheck") && el("compositeBoxImagesCheck").checked);

  const refParts = [];
  let gPic = 0;
  let gDit = 0;
  for (const refPath of state.bindings.reference_image_paths || []) {
    if (!refPath) continue;
    if (globalRole !== "inpaint_mask") gDit += 1;
    if (globalEmitsVit) gPic += 1;
    const tag = globalEmitsVit ? `Picture ${gPic}` : (globalRole !== "inpaint_mask" ? `DiT-Ref ${gDit}` : "Mask-Ref");
    refParts.push(`${tag} (${globalRole}, global_frame=${globalFraming})`);
  }

  for (const idx of canonicalIndices) {
    const item = state.layoutItems[idx];
    if (!item || item.enabled === false) continue;
    const imgP = String(item.image_path || item.image || item.vizj_path || "").trim();
    if (!imgP) continue;
    const role = String(item.input_role || globalRole || "photo_ref").toLowerCase();
    const tag = boxPicMap[idx] ? `Picture ${boxPicMap[idx]}` : (boxDitMap[idx] ? `DiT-Ref ${boxDitMap[idx]}` : "Mask-Ref");
    const bbox = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox : [0, 0, height, width];
    const cB = pixelToAbsCentralized(bbox[0], bbox[1], bbox[2], bbox[3], height, width);
    const iso = resolveEffectiveBoxIsolation(item, height, width);
    const splat = item.paste_to_canvas !== null && item.paste_to_canvas !== undefined
      ? Boolean(item.paste_to_canvas)
      : (globalSplat && role === "photo_ref");
    const refMeta = findRefMeta(imgP);
    const cutoutSfx = refMeta && refMeta.has_alpha && role === "photo_ref" ? ", cutout=True" : "";
    refParts.push(`${tag} (${role}, bbox=[${cB.join(", ")}], iso=${iso.toFixed(2)}, splat=${splat}${cutoutSfx})`);
  }

  const hasBoxRefs = Object.keys(boxDitMap).length > 0;
  const expZeroT = parseTriStateBool(el("refZeroTimestepSelect") ? el("refZeroTimestepSelect").value : "");
  const expIsoKv = parseTriStateBool(el("refIsolatedKvSelect") ? el("refIsolatedKvSelect").value : "");
  const effZeroT = expZeroT !== null
    ? expZeroT
    : (hasBoxRefs || ["moss_image_edit", "ai_toolkit_isolated", "ai_toolkit_full", "control_color_depth"].includes(editFormat));
  const effIsoKv = expIsoKv !== null
    ? expIsoKv
    : (hasBoxRefs || ["moss_image_edit", "ai_toolkit_isolated", "control_color_depth"].includes(editFormat));

  if (refParts.length > 0) {
    lines.push(
      `3. Visual Conditioning: ${refParts.length} reference(s) [t_ref=0: ${effZeroT}, IsoKV: ${effIsoKv}] -> ${refParts.join("; ")}.`
    );
  }

  if (canonicalIndices.length > 0) {
    const boxDescs = [];
    canonicalIndices.forEach((idx, orderIdx) => {
      const item = state.layoutItems[idx];
      const rawTxt = typeof item.text === "string" ? item.text.trim() : "";
      const rawCap = typeof (item.caption || item.description) === "string"
        ? String(item.caption || item.description).trim()
        : "";
      const lbl = normalizeBoxLabel(item.label || item.category, Boolean(rawTxt));
      const txtVal = lbl === "text" ? (rawTxt || rawCap) : "";
      const capVal = lbl === "text"
        ? (!rawCap || rawCap === txtVal ? "Crisp legible typography with clean kerning" : rawCap)
        : (rawCap || (lbl === "character" ? "Character subject" : "Scene object"));
      const bbox = Array.isArray(item.bbox) && item.bbox.length === 4 ? item.bbox : [0, 0, height, width];
      const cB = pixelToAbsCentralized(bbox[0], bbox[1], bbox[2], bbox[3], height, width);
      const areaPct = ((Math.abs(bbox[2] - bbox[0]) / Math.max(1, height)) * (Math.abs(bbox[3] - bbox[1]) / Math.max(1, width)) * 100.0).toFixed(1);
      const refTag = boxPicMap[idx] ? `, ref=Picture ${boxPicMap[idx]}` : "";
      const dnVal = parseOptionalFloat(item.denoise_strength, 0.01, 1.0);
      const dnTag = dnVal !== null ? `, d=${dnVal.toFixed(2)}` : "";
      if (lbl === "text" && txtVal) {
        const styleNote = rawCap && rawCap !== txtVal ? rawCap : "crisp kerning";
        boxDescs.push(`R${orderIdx + 1}[text [${cB.join(", ")}], ${areaPct}%${refTag}${dnTag}: ${styleNote}]`);
      } else {
        boxDescs.push(`R${orderIdx + 1}[${lbl} [${cB.join(", ")}], ${areaPct}%${refTag}${dnTag}: ${capVal}]`);
      }
    });
    const stepNum = refParts.length > 0 ? 4 : 3;
    lines.push(`${stepNum}. Coarse-to-Fine Spatial Hierarchy: ${boxDescs.join("; ")}.`);
  }

  if (activeTexts.length > 0) {
    const stepNum = lines.length + 1;
    const quoted = activeTexts.map((t) => `"${t}"`).join(", ");
    lines.push(`${stepNum}. Typography Lock: Preserve exact character glyph sequence and stroke contrast for ${quoted}.`);
  }

  return lines.join("\n");
}

function scheduleAuthoritativePromptPreview(hasBoxRefs) {
  if (previewDebounceTimer) {
    clearTimeout(previewDebounceTimer);
  }
  const reqSeq = ++previewRequestSeq;
  previewDebounceTimer = setTimeout(async () => {
    try {
      const width = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
      const height = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
      const summaryOverride = el("layoutSummaryInput") ? el("layoutSummaryInput").value.trim() : "";
      const resolvedTrace = resolveThinkingTraceFromDeck();

      const res = await apiFetch("api/v1/prompt/preview", {
        method: "POST",
        body: {
          prompt: el("promptInput").value.trim(),
          prompt_mode: el("promptModeSelect").value,
          layout_summary: summaryOverride || null,
          layout_items: state.layoutItems,
          layout_coords_are_pixels: true,
          width,
          height,
          system_prompt: el("systemPromptInput") ? (el("systemPromptInput").value.trim() || null) : null,
          thinking_trace: resolvedTrace,
          enable_thinking: Boolean(el("enableThinkingCheck") && el("enableThinkingCheck").checked),
          add_vision_id: Boolean(el("addVisionIdCheck") && el("addVisionIdCheck").checked),
          max_sequence_length: parseFiniteInt(
            el("maxSeqLenInput") ? el("maxSeqLenInput").value : 16384,
            16384,
            64,
            32768
          ),
          max_vit_patches: parseFiniteInt(
            el("maxVitPatchesInput") ? el("maxVitPatchesInput").value : 2304,
            2304,
            64,
            9216
          ),
          edit_format: el("editFormatSelect") ? el("editFormatSelect").value : "standard",
          framing_mode: el("framingModeSelect") ? el("framingModeSelect").value : "inscribed",
          input_role: el("inputRoleSelect") ? el("inputRoleSelect").value : "photo_ref",
          box_attn_isolation: parseFiniteFloat(
            el("boxAttnIsolationInput") ? el("boxAttnIsolationInput").value : 0.0,
            0.0,
            0.0,
            1.0
          ),
          composite_box_images_to_init: Boolean(
            el("compositeBoxImagesCheck") && el("compositeBoxImagesCheck").checked
          ),
          uncond_uses_references: parseTriStateBool(
            el("uncondUsesRefsSelect") ? el("uncondUsesRefsSelect").value : ""
          ),
          reference_zero_timestep: parseTriStateBool(
            el("refZeroTimestepSelect") ? el("refZeroTimestepSelect").value : ""
          ),
          reference_isolated_kv: parseTriStateBool(
            el("refIsolatedKvSelect") ? el("refIsolatedKvSelect").value : ""
          ),
          reference_image_paths: state.bindings.reference_image_paths || [],
          control_image_path: state.bindings.control_image_path || null
        }
      });
      if (!res || reqSeq !== previewRequestSeq) return;

      if (res.spatio_visual_graph) {
        state.lastSpatioVisualGraph = res.spatio_visual_graph;
      }

      const modeBadge = el("layoutActiveModeBadge");
      if (modeBadge && res.total_context_tokens !== undefined) {
        const promptMode = el("promptModeSelect").value;
        const baseModeStr = (promptMode === "layout_json" || hasBoxRefs)
          ? (hasBoxRefs ? "LAYOUT JSON + 4D-RoPE" : "LAYOUT JSON")
          : "NATURAL MODE";
        const visTag = res.vision_tokens > 0 ? ` | VIS ${res.vision_tokens}T` : "";
        const sv = res.spatio_visual_graph || {};
        const carveCount = Array.isArray(sv.foreground_carveouts) ? sv.foreground_carveouts.length : 0;
        const carveTag = carveCount > 0 ? ` | CARVE ${carveCount}` : "";
        modeBadge.textContent = `${baseModeStr} (${res.total_context_tokens}/${res.max_sequence_length} TOK${visTag}${carveTag})`;
      }

      const cotWrap = el("compiledCotPreviewWrap");
      const cotPre = el("compiledCotPreview");
      const cotBadge = el("cotActiveModeBadge");
      if (cotWrap && cotPre && cotBadge) {
        if (res.speculative_cot_tokens > 0) {
          cotBadge.textContent = `REANCHOR SPECULATIVE (${res.speculative_cot_tokens} TOK | CTX ${res.total_context_tokens} TOK)`;
        } else if (res.resolved_thinking_trace) {
          const sLower = (resolvedTrace || "").toLowerCase();
          const labelPrefix = ["auto", "spatial", "compile"].includes(sLower)
            ? "AUTO SPATIAL COT"
            : "CUSTOM TRACE";
          cotBadge.textContent = `${labelPrefix} (${res.trace_tokens || 0} TOK)`;
          cotPre.textContent = res.resolved_thinking_trace;
        }
      }
    } catch (_) {}
  }, 160);
}

export function updateCompiledJsonPreview() {
  const width = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
  const height = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
  const summaryOverride = el("layoutSummaryInput") ? el("layoutSummaryInput").value.trim() : "";
  const rawCaption = summaryOverride || (el("promptInput").value || "").trim();
  const { boxPicMap, boxDitMap, canonicalIndices } = computeBoxPictureBindings();

  const items = [];
  const activeTexts = getReadingOrderTextsJs(height, width);
  for (const idx of canonicalIndices) {
    const b = state.layoutItems[idx];
    if (!b || b.enabled === false) continue;
    const rawTxt = typeof b.text === "string" ? b.text.trim() : "";
    const labelVal = normalizeBoxLabel(b.label || b.category, Boolean(rawTxt));
    const rawCap = b.caption || b.description;
    let txtVal = rawTxt;
    let capVal = typeof rawCap === "string" ? rawCap.trim() : "";
    if (labelVal === "text") {
      if (!txtVal && capVal) {
        txtVal = capVal;
      }
      if (!capVal || capVal === txtVal) {
        capVal = "Crisp legible typography with clean kerning";
      }
    } else if (!capVal) {
      capVal = labelVal === "character" ? "Character subject" : "Scene object";
    }

    const entry = { label: labelVal };
    if (Array.isArray(b.bbox) && b.bbox.length === 4) {
      entry.bbox = pixelToAbsCentralized(b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], height, width);
    }
    if (boxPicMap[idx]) {
      entry.reference = `Picture ${boxPicMap[idx]}`;
    }
    if (capVal) {
      entry.caption = capVal;
    }
    if (labelVal === "text" && txtVal) {
      entry.text = txtVal;
    }
    items.push(entry);
  }

  const reconciledCaption = reconcileSummaryQuotesJs(rawCaption, activeTexts);
  const payload = {
    summary: { caption: reconciledCaption },
    items,
    bbox_coord_space: "abs_centralized",
    bbox_format: "yxyx"
  };

  const previewNode = el("compiledJsonPreview");
  if (previewNode) {
    previewNode.textContent = JSON.stringify(payload, null, 2);
  }

  const hasBoxRefs = Object.keys(boxDitMap).length > 0;
  const modeBadge = el("layoutActiveModeBadge");
  if (modeBadge) {
    const promptMode = el("promptModeSelect").value;
    modeBadge.style.cursor = "pointer";
    modeBadge.title = "Click to toggle Prompt Mode between Spatial Layout JSON and Natural Language";
    if (promptMode === "layout_json" || hasBoxRefs) {
      modeBadge.textContent = hasBoxRefs ? "LAYOUT JSON + 4D-RoPE ACTIVE" : "LAYOUT JSON ACTIVE";
    } else {
      modeBadge.textContent = "NATURAL MODE (CLICK TO ACTIVATE JSON)";
    }
  }

  const cotWrap = el("compiledCotPreviewWrap");
  const cotPre = el("compiledCotPreview");
  const cotBadge = el("cotActiveModeBadge");
  if (cotWrap && cotPre && cotBadge) {
    const resolvedTrace = resolveThinkingTraceFromDeck();
    const sLower = (resolvedTrace || "").toLowerCase();
    if (!resolvedTrace) {
      cotWrap.hidden = true;
      cotPre.textContent = "";
    } else if (["auto", "spatial", "compile"].includes(sLower)) {
      cotWrap.hidden = false;
      cotBadge.textContent = "AUTO SPATIAL COT";
      cotPre.textContent = compileAutoSpatialTraceJs(
        width,
        height,
        reconciledCaption,
        canonicalIndices,
        activeTexts,
        boxPicMap,
        boxDitMap
      );
    } else if (sLower === "speculative" || sLower.startsWith("speculative:")) {
      const budget = sLower.includes(":")
        ? parseFiniteInt(sLower.split(":", 2)[1], 64, 4, 512)
        : 64;
      cotWrap.hidden = false;
      cotBadge.textContent = `REANCHOR SPECULATIVE (${budget} TOK)`;
      cotPre.textContent =
        `Live ReAnchor Speculative CoT Active:\n` +
        `  Budget: ${budget} reasoning tokens (block_size=4, max_candidates=4, H_thresh=1.25 bits)\n` +
        `  Drafter: Parallel Shifted MTP / Variable-Order Shannon-Entropy Markov Prefix Survival\n` +
        `  Auto-Closure: Guaranteed \\n</think>\\n\\n termination prior to MOSS-DiT cross-attention`;
    } else {
      cotWrap.hidden = false;
      cotBadge.textContent = "CUSTOM TRACE";
      cotPre.textContent = resolvedTrace;
    }
  }

  scheduleAuthoritativePromptPreview(hasBoxRefs);
  renderBindingsCard();
  renderViewportBoxesHud();
}

export function highlightSelectedBoxInDom() {
  const boxesLayer = el("spatialBoxesLayer");
  const listContainer = el("boxListContainer");
  if (boxesLayer) {
    boxesLayer.querySelectorAll(".spatial-box").forEach((boxEl) => {
      const idx = parseInt(boxEl.getAttribute("data-box-idx"), 10);
      const item = state.layoutItems[idx];
      const baseZ = item && typeof item.z_order === "number" ? item.z_order : idx + 1;
      if (idx === state.selectedBoxIndex) {
        boxEl.classList.add("selected-box");
        boxEl.style.zIndex = String(100 + baseZ);
      } else {
        boxEl.classList.remove("selected-box");
        boxEl.style.zIndex = String(baseZ);
      }
    });
  }
  if (listContainer) {
    listContainer.querySelectorAll(".box-card").forEach((cardEl) => {
      const idx = parseInt(cardEl.getAttribute("data-card-idx"), 10);
      cardEl.classList.toggle("selected-card", idx === state.selectedBoxIndex);
    });
  }
}

export function cycleOverlappingBoxes(canvasX = null, canvasY = null) {
  const enabledIndices = [];
  state.layoutItems.forEach((b, idx) => {
    if (b.enabled === false || !Array.isArray(b.bbox) || b.bbox.length !== 4) return;
    if (canvasX !== null && canvasY !== null) {
      const [y0, x0, y1, x1] = b.bbox;
      if (canvasY >= y0 && canvasY <= y1 && canvasX >= x0 && canvasX <= x1) {
        enabledIndices.push(idx);
      }
    } else {
      enabledIndices.push(idx);
    }
  });
  if (!enabledIndices.length) return;
  const curPos = enabledIndices.indexOf(state.selectedBoxIndex);
  state.selectedBoxIndex = enabledIndices[(curPos + 1) % enabledIndices.length];
  highlightSelectedBoxInDom();
  const listContainer = el("boxListContainer");
  if (listContainer) {
    const targetCard = listContainer.querySelector(`[data-card-idx="${state.selectedBoxIndex}"]`);
    if (targetCard) targetCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

export function shiftBoxZOrder(idx, delta) {
  ensureBoxZOrders();
  const target = state.layoutItems[idx];
  if (!target) return;
  target.z_order = Math.max(1, (target.z_order || 1) + delta);
  state.selectedBoxIndex = idx;
  renderSpatialLayoutEditor();
}

export function renderSpatialLayoutEditor() {
  ensureBoxZOrders();
  const geom = computeStageGeometry("spatialStageContainer");
  syncStageAspectAndGuides();
  syncUnderlayLayers();

  const boxesLayer = el("spatialBoxesLayer");
  const listContainer = el("boxListContainer");
  if (!boxesLayer || !listContainer) return;

  const { boxPicMap, boxDitMap } = computeBoxPictureBindings();
  const boxGraph = computeSpatioVisualBoxGraph(geom.canvasW, geom.canvasH);
  const globalFraming = el("framingModeSelect").value || "inscribed";
  const globalSplat = Boolean(el("compositeBoxImagesCheck").checked);

  boxesLayer.innerHTML = "";
  listContainer.innerHTML = "";

  state.layoutItems.forEach((item, idx) => {
    const isEnabled = item.enabled !== false;
    const [ymin, xmin, ymax, xmax] = item.bbox;
    const isText = isTypographyBox(item);
    const normLbl = normalizeBoxLabel(item.label || item.category, isText);
    const isChar = normLbl === "character";
    const hasBoundRef = Boolean(item.image_path && String(item.image_path).trim());
    const refMeta = hasBoundRef ? findRefMeta(item.image_path) : null;
    const effRole = String(item.input_role || el("inputRoleSelect").value || "photo_ref").toLowerCase();
    const effFraming = item.framing_mode || (globalFraming === "never_crop" ? "inscribed" : globalFraming);
    const effIso = resolveEffectiveBoxIsolation(item, geom.canvasH, geom.canvasW);
    const hasExplicitIso = parseOptionalFloat(item.isolate_box_attn, 0.0, 1.0) !== null;
    const effSplat = item.paste_to_canvas !== null && item.paste_to_canvas !== undefined
      ? Boolean(item.paste_to_canvas)
      : (hasBoundRef && globalSplat && effRole === "photo_ref");
    const isSelected = idx === state.selectedBoxIndex;
    const baseZ = typeof item.z_order === "number" ? item.z_order : idx + 1;
    const pxW = Math.max(1, Math.round(xmax - xmin));
    const pxH = Math.max(1, Math.round(ymax - ymin));
    const gInfo = boxGraph[idx] || {
      peerBoxIndices: [],
      coOpBoxIndices: [],
      carvesSplat: false,
      carvesKv: false,
      isCutout: false
    };

    if (isEnabled) {
      const boxEl = document.createElement("div");
      boxEl.className = `spatial-box ${isText ? "text-box" : ""} ${isChar ? "char-box" : ""} ${hasBoundRef ? "ref-bound-box" : ""} ${isSelected ? "selected-box" : ""}`;
      boxEl.setAttribute("data-box-idx", String(idx));
      boxEl.style.zIndex = String(isSelected ? 100 + baseZ : baseZ);
      boxEl.style.top = `${Math.round(ymin * geom.scaleY)}px`;
      boxEl.style.left = `${Math.round(xmin * geom.scaleX)}px`;
      boxEl.style.width = `${Math.max(12, Math.round((xmax - xmin) * geom.scaleX))}px`;
      boxEl.style.height = `${Math.max(12, Math.round((ymax - ymin) * geom.scaleY))}px`;

      const thumbHtml = refMeta && !refMeta.is_vizj
        ? `<div class="spatial-box-thumb-clip"><img class="spatial-box-thumb" src="${refMeta.url}" alt="" style="object-fit: ${framingModeToCssObjectFit(effFraming)};" /></div>`
        : "";

      const badges = [
        `<span class="box-mini-pill">Z${baseZ}</span>`,
        `<span class="box-mini-pill" data-box-dim-pill="${idx}">${pxW}x${pxH}</span>`
      ];
      if (boxPicMap[idx]) {
        badges.push(`<span class="box-mini-pill">Pic ${boxPicMap[idx]}</span>`);
      } else if (boxDitMap[idx]) {
        badges.push(`<span class="box-mini-pill">4D#${boxDitMap[idx]} (${effRole})</span>`);
      }
      if (hasBoundRef && effIso > 0) {
        badges.push(`<span class="box-mini-pill">Iso ${Number(effIso).toFixed(2)}${hasExplicitIso ? "" : "*"}</span>`);
      }
      if (hasBoundRef && effSplat) {
        badges.push(`<span class="box-mini-pill">Splat</span>`);
      }
      if (gInfo.isCutout) {
        badges.push(`<span class="box-mini-pill pill-cutout" title="Silhouette-conformal RGBA cutout mask">Cutout</span>`);
      }
      if (gInfo.coOpBoxIndices.length > 0) {
        const peerTags = gInfo.coOpBoxIndices.map((p) => `#${p + 1}`).join(",");
        badges.push(`<span class="box-mini-pill pill-coop" title="Multi-role IsoKV peer cooperation with Box ${peerTags}">Co-Op ${peerTags}</span>`);
      } else if (gInfo.peerBoxIndices.length > 0) {
        const peerTags = gInfo.peerBoxIndices.map((p) => `#${p + 1}`).join(",");
        badges.push(`<span class="box-mini-pill pill-coop" title="Szymkiewicz-Simpson IsoKV peer with Box ${peerTags}">Peer ${peerTags}</span>`);
      }
      if (gInfo.carvesSplat || gInfo.carvesKv) {
        const carveDesc = gInfo.carvesSplat && gInfo.carvesKv
          ? "Carves parent splat prior & reference KV attention"
          : (gInfo.carvesSplat ? "Carves parent splat prior" : "Carves parent reference KV attention");
        badges.push(`<span class="box-mini-pill pill-carve" title="${carveDesc}">Carve</span>`);
      }
      if (item.inpaint_box) {
        badges.push(`<span class="box-mini-pill">Inpaint</span>`);
      }

      const handlesHtml = ["nw", "n", "ne", "e", "se", "s", "sw", "w"]
        .map((dir) => `<div class="spatial-box-handle" data-handle="${dir}" data-box-idx="${idx}"></div>`)
        .join("");

      const labelText = isText ? `T: "${item.text || item.caption || ""}"` : `${normLbl}: ${item.caption || ""}`;
      boxEl.innerHTML = `
        ${thumbHtml}
        <div class="spatial-box-tag">[#${idx + 1}] ${escapeHtmlAttr(labelText)}</div>
        <div class="spatial-box-badges">${badges.join("")}</div>
        ${handlesHtml}
      `;

      boxEl.addEventListener("mousedown", (ev) => {
        const handleDir = ev.target.getAttribute("data-handle");
        if (ev.altKey && !handleDir) {
          ev.preventDefault();
          ev.stopPropagation();
          const stageRect = el("spatialStage").getBoundingClientRect();
          const clickX = (ev.clientX - stageRect.left) / geom.scaleX;
          const clickY = (ev.clientY - stageRect.top) / geom.scaleY;
          cycleOverlappingBoxes(clickX, clickY);
          return;
        }

        state.selectedBoxIndex = idx;
        highlightSelectedBoxInDom();
        el("spatialStage").focus();
        ev.preventDefault();
        ev.stopPropagation();

        const startX = ev.clientX;
        const startY = ev.clientY;
        const [origY0, origX0, origY1, origX1] = item.bbox;
        const boxW = origX1 - origX0;
        const boxH = origY1 - origY0;

        function updateDomAndReadouts() {
          const [cy0, cx0, cy1, cx1] = item.bbox;
          const curW = Math.max(1, Math.round(cx1 - cx0));
          const curH = Math.max(1, Math.round(cy1 - cy0));
          boxEl.style.top = `${Math.round(cy0 * geom.scaleY)}px`;
          boxEl.style.left = `${Math.round(cx0 * geom.scaleX)}px`;
          boxEl.style.width = `${Math.max(12, Math.round((cx1 - cx0) * geom.scaleX))}px`;
          boxEl.style.height = `${Math.max(12, Math.round((cy1 - cy0) * geom.scaleY))}px`;

          const dimPill = boxEl.querySelector(`[data-box-dim-pill="${idx}"]`);
          if (dimPill) {
            dimPill.textContent = `${curW}x${curH}`;
          }
          const hdr = listContainer.querySelector(`[data-box-coord-hdr="${idx}"]`);
          if (hdr) {
            const cc = pixelToAbsCentralized(cy0, cx0, cy1, cx1, geom.canvasH, geom.canvasW);
            hdr.textContent = `#${idx + 1} [${cc.join(", ")}] (${curW}x${curH}px)`;
          }
          renderInpaintBoxGuides();
          updateCompiledJsonPreview();
        }

        function onMove(moveEv) {
          const dx = (moveEv.clientX - startX) / geom.scaleX;
          const dy = (moveEv.clientY - startY) / geom.scaleY;
          const freeMotion = moveEv.altKey;
          const minSpan = freeMotion ? 4 : 16;

          if (!handleDir) {
            const nx0 = snapCoord(Math.max(0, Math.min(geom.canvasW - boxW, origX0 + dx)), geom.canvasW - boxW, freeMotion);
            const ny0 = snapCoord(Math.max(0, Math.min(geom.canvasH - boxH, origY0 + dy)), geom.canvasH - boxH, freeMotion);
            item.bbox = [ny0, nx0, ny0 + boxH, nx0 + boxW];
          } else {
            let ny0 = origY0;
            let nx0 = origX0;
            let ny1 = origY1;
            let nx1 = origX1;
            if (handleDir.includes("n")) {
              ny0 = Math.min(ny1 - minSpan, snapCoord(origY0 + dy, geom.canvasH - minSpan, freeMotion));
            }
            if (handleDir.includes("s")) {
              ny1 = Math.max(ny0 + minSpan, snapCoord(origY1 + dy, geom.canvasH, freeMotion));
            }
            if (handleDir.includes("w")) {
              nx0 = Math.min(nx1 - minSpan, snapCoord(origX0 + dx, geom.canvasW - minSpan, freeMotion));
            }
            if (handleDir.includes("e")) {
              nx1 = Math.max(nx0 + minSpan, snapCoord(origX1 + dx, geom.canvasW, freeMotion));
            }
            item.bbox = [ny0, nx0, ny1, nx1];
          }
          updateDomAndReadouts();
        }

        function onUp() {
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
          renderSpatialLayoutEditor();
        }

        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      });

      boxesLayer.appendChild(boxEl);
    }

    const refOptions = [`<option value="">&lt;No Box Image&gt;</option>`]
      .concat(
        state.references.map(
          (r) =>
            `<option value="${escapeHtmlAttr(r.ref_id)}" ${item.image_path === r.ref_id ? "selected" : ""}>${escapeHtmlAttr(r.ref_id)} (${r.width || 1024}x${r.height || 1024})</option>`
        )
      )
      .join("");

    const curRole = item.input_role || "";
    const curFraming = item.framing_mode || "";
    const isoVal = item.isolate_box_attn !== null && item.isolate_box_attn !== undefined ? item.isolate_box_attn : "";
    const dnVal = item.denoise_strength !== null && item.denoise_strength !== undefined ? item.denoise_strength : "";
    const isoPlaceholder = hasBoundRef ? `Attn Iso (Auto: ${effIso.toFixed(2)})` : (isText ? "Carve Iso (Auto: 0.85)" : "Attn Iso (Auto: 0.65)");
    const dnPlaceholder = isText && !hasBoundRef ? "Denoise (Auto: 0.65)" : "Denoise (Auto: 0.75)";
    const cCoords = pixelToAbsCentralized(ymin, xmin, ymax, xmax, geom.canvasH, geom.canvasW);
    const primaryDescVal = isText ? (item.text ?? item.caption ?? "") : (item.caption ?? "");
    const textCapHtml = isText
      ? `<input type="text" class="input-text" data-box-cap="${idx}" value="${escapeHtmlAttr(item.caption || "")}" placeholder="Typography surface / style caption (e.g. Handwritten marker on yellow sticky note)" title="Visual surface, font, and material description for this typography aperture" />`
      : "";

    const card = document.createElement("div");
    card.className = `box-card ${!isEnabled ? "disabled-card" : ""} ${isSelected ? "selected-card" : ""}`;
    card.setAttribute("data-card-idx", String(idx));
    card.innerHTML = `
      <div class="box-card-header">
        <label class="checkbox-inline">
          <input type="checkbox" data-box-enabled="${idx}" ${isEnabled ? "checked" : ""} />
          <strong data-box-coord-hdr="${idx}">#${idx + 1} [${cCoords.join(", ")}] (${pxW}x${pxH}px)</strong>
        </label>
        <div class="box-z-controls">
          <span class="tiny-muted" title="Canonical Qwen3.5 Label &amp; Z-Plane">${normLbl.toUpperCase()} | Z:</span>
          <input type="number" class="z-input" data-box-z="${idx}" value="${baseZ}" min="0" max="99" step="1" title="Explicit Z-order plane (higher Z occludes same-role lower Z and enables foreground child-aperture carve-outs)" />
          <button type="button" class="z-btn" data-z-up="${idx}" title="Bring box forward in Z-plane stack">&uarr;</button>
          <button type="button" class="z-btn" data-z-down="${idx}" title="Send box backward in Z-plane stack">&darr;</button>
          <button type="button" class="btn-link" data-remove-box="${idx}">Remove</button>
        </div>
      </div>
      <div class="field-row two-col">
        <select class="input-select" data-box-cat="${idx}" title="Canonical Qwen3.5 Semantic Box Category">
          <option value="object" ${normLbl === "object" ? "selected" : ""}>Type: object (Prop / Region)</option>
          <option value="character" ${normLbl === "character" ? "selected" : ""}>Type: character (Subject / Person)</option>
          <option value="text" ${normLbl === "text" ? "selected" : ""}>Type: text (Typography Lock)</option>
        </select>
        <select class="input-select" data-box-img="${idx}" title="Bind Reference Asset to Box 4D-RoPE">${refOptions}</select>
      </div>
      <input type="text" class="input-text" data-box-desc="${idx}" value="${escapeHtmlAttr(primaryDescVal)}" placeholder="${isText ? "Exact typography text" : (isChar ? "Character appearance & pose description" : "Object / region description")}" />
      ${textCapHtml}
      <div class="field-row two-col">
        <select class="input-select" data-box-role="${idx}" title="Per-Box Input Role Override">
          <option value="" ${curRole === "" ? "selected" : ""}>Role: Inherit Global</option>
          <option value="photo_ref" ${curRole === "photo_ref" ? "selected" : ""}>Role: photo_ref (Picture k)</option>
          <option value="color_map" ${curRole === "color_map" ? "selected" : ""}>Role: color_map</option>
          <option value="depth_map" ${curRole === "depth_map" ? "selected" : ""}>Role: depth_map</option>
          <option value="inpaint_mask" ${curRole === "inpaint_mask" ? "selected" : ""}>Role: inpaint_mask</option>
        </select>
        <select class="input-select" data-box-framing="${idx}" title="Per-Box Framing Mode Override">
          <option value="" ${curFraming === "" ? "selected" : ""}>Framing: Box Conformal</option>
          <option value="inscribed" ${curFraming === "inscribed" ? "selected" : ""}>Framing: inscribed</option>
          <option value="cover_crop" ${curFraming === "cover_crop" ? "selected" : ""}>Framing: cover_crop</option>
          <option value="contain" ${curFraming === "contain" ? "selected" : ""}>Framing: contain</option>
          <option value="stretch" ${curFraming === "stretch" ? "selected" : ""}>Framing: stretch</option>
          <option value="never_crop" ${curFraming === "never_crop" ? "selected" : ""}>Framing: never_crop</option>
        </select>
      </div>
      <div class="field-row two-col">
        <input type="number" class="input-text" data-box-iso="${idx}" value="${isoVal}" min="0.0" max="1.0" step="0.05" placeholder="${isoPlaceholder}" title="Per-box 2D cosine log-aperture attention isolation [0.0 - 1.0] (Blank = Auto, 0 = Full-Canvas Bleed)" />
        <input type="number" class="input-text" data-box-dn="${idx}" value="${dnVal}" min="0.01" max="1.0" step="0.01" placeholder="${dnPlaceholder}" title="Per-box Tweedie velocity-space splat denoise handoff [0.01 - 1.0]" />
      </div>
      <div class="box-flags-row">
        <label class="checkbox-inline" title="Restrict canvas inpainting to this box aperture">
          <input type="checkbox" data-box-inpaint="${idx}" ${item.inpaint_box ? "checked" : ""} />
          <span>Inpaint Box Only</span>
        </label>
        <label class="checkbox-inline" title="Lend this box's reference latents to the canvas via Tweedie velocity-space prior coupling">
          <input type="checkbox" data-box-paste="${idx}" ${item.paste_to_canvas ? "checked" : ""} />
          <span>Splat to Canvas</span>
        </label>
      </div>
    `;
    card.addEventListener("mouseenter", () => {
      state.selectedBoxIndex = idx;
      highlightSelectedBoxInDom();
    });
    card.addEventListener("click", () => {
      state.selectedBoxIndex = idx;
      highlightSelectedBoxInDom();
    });
    listContainer.appendChild(card);
  });

  listContainer.querySelectorAll("[data-box-z]").forEach((inp) => {
    inp.addEventListener("change", (ev) => {
      ev.stopPropagation();
      const i = parseInt(inp.getAttribute("data-box-z"), 10);
      const target = state.layoutItems[i];
      if (!target) return;
      target.z_order = parseFiniteInt(inp.value, i + 1, 0, 99);
      state.selectedBoxIndex = i;
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-z-up]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      shiftBoxZOrder(parseInt(btn.getAttribute("data-z-up"), 10), 1);
    });
  });

  listContainer.querySelectorAll("[data-z-down]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      shiftBoxZOrder(parseInt(btn.getAttribute("data-z-down"), 10), -1);
    });
  });

  listContainer.querySelectorAll("[data-box-enabled]").forEach((chk) => {
    chk.addEventListener("change", () => {
      const i = parseInt(chk.getAttribute("data-box-enabled"), 10);
      state.layoutItems[i].enabled = chk.checked;
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-remove-box]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const i = parseInt(btn.getAttribute("data-remove-box"), 10);
      state.layoutItems.splice(i, 1);
      state.selectedBoxIndex = Math.max(0, Math.min(state.selectedBoxIndex, state.layoutItems.length - 1));
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-cat]").forEach((sel) => {
    sel.addEventListener("change", () => {
      const i = parseInt(sel.getAttribute("data-box-cat"), 10);
      const target = state.layoutItems[i];
      const norm = normalizeBoxLabel(sel.value, sel.value === "text");
      target.category = norm;
      target.label = norm;
      if (norm === "text") {
        if (target.text === undefined || target.text === "") {
          target.text = target.caption || "SOVEREIGN 2026";
        }
      } else if (target.text !== undefined) {
        if (!target.caption) {
          target.caption = target.text;
        }
        delete target.text;
      }
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-desc]").forEach((inp) => {
    inp.addEventListener("input", () => {
      const i = parseInt(inp.getAttribute("data-box-desc"), 10);
      const target = state.layoutItems[i];
      const isT = isTypographyBox(target);
      if (isT) {
        const prevText = target.text || "";
        target.text = inp.value;
        target.label = "text";
        target.category = "text";
        syncTextEditToPrompt(prevText, inp.value);
      } else {
        target.caption = inp.value;
        const normL = normalizeBoxLabel(target.label || target.category, false);
        target.label = normL;
        target.category = normL;
      }
      const stageBox = boxesLayer.querySelector(`.spatial-box[data-box-idx="${i}"] .spatial-box-tag`);
      if (stageBox) {
        const normL = normalizeBoxLabel(target.label || target.category, isT);
        stageBox.textContent = `[#${i + 1}] ${isT ? `T: "${target.text || ""}"` : `${normL}: ${target.caption || ""}`}`;
      }
      updateCompiledJsonPreview();
    });
  });

  listContainer.querySelectorAll("[data-box-cap]").forEach((inp) => {
    inp.addEventListener("input", () => {
      const i = parseInt(inp.getAttribute("data-box-cap"), 10);
      const target = state.layoutItems[i];
      const val = inp.value;
      if (val && val.trim()) {
        target.caption = val;
      } else {
        delete target.caption;
      }
      updateCompiledJsonPreview();
    });
  });

  listContainer.querySelectorAll("[data-box-img]").forEach((sel) => {
    sel.addEventListener("change", () => {
      const i = parseInt(sel.getAttribute("data-box-img"), 10);
      state.layoutItems[i].image_path = sel.value;
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-role]").forEach((sel) => {
    sel.addEventListener("change", () => {
      const i = parseInt(sel.getAttribute("data-box-role"), 10);
      state.layoutItems[i].input_role = sel.value || "";
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-framing]").forEach((sel) => {
    sel.addEventListener("change", () => {
      const i = parseInt(sel.getAttribute("data-box-framing"), 10);
      state.layoutItems[i].framing_mode = sel.value || "";
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-iso]").forEach((inp) => {
    inp.addEventListener("input", () => {
      const i = parseInt(inp.getAttribute("data-box-iso"), 10);
      state.layoutItems[i].isolate_box_attn = parseOptionalFloat(inp.value, 0.0, 1.0);
      updateCompiledJsonPreview();
    });
  });

  listContainer.querySelectorAll("[data-box-dn]").forEach((inp) => {
    inp.addEventListener("input", () => {
      const i = parseInt(inp.getAttribute("data-box-dn"), 10);
      state.layoutItems[i].denoise_strength = parseOptionalFloat(inp.value, 0.01, 1.0);
      updateCompiledJsonPreview();
    });
  });

  listContainer.querySelectorAll("[data-box-inpaint]").forEach((chk) => {
    chk.addEventListener("change", () => {
      const i = parseInt(chk.getAttribute("data-box-inpaint"), 10);
      state.layoutItems[i].inpaint_box = chk.checked;
      renderSpatialLayoutEditor();
    });
  });

  listContainer.querySelectorAll("[data-box-paste]").forEach((chk) => {
    chk.addEventListener("change", () => {
      const i = parseInt(chk.getAttribute("data-box-paste"), 10);
      state.layoutItems[i].paste_to_canvas = chk.checked;
      renderSpatialLayoutEditor();
    });
  });

  updateCompiledJsonPreview();
}

export function handleStageKeydown(ev) {
  const item = state.layoutItems[state.selectedBoxIndex];
  if (!item || item.enabled === false || !Array.isArray(item.bbox)) return;

  if (ev.key === "Delete" || ev.key === "Backspace") {
    ev.preventDefault();
    state.layoutItems.splice(state.selectedBoxIndex, 1);
    state.selectedBoxIndex = Math.max(0, Math.min(state.selectedBoxIndex, state.layoutItems.length - 1));
    renderSpatialLayoutEditor();
    return;
  }

  const step = ev.shiftKey ? 64 : (ev.altKey ? 1 : 16);
  const canvasW = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
  const canvasH = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
  const [y0, x0, y1, x1] = item.bbox;
  const w = x1 - x0;
  const h = y1 - y0;

  if (ev.key === "ArrowLeft") {
    ev.preventDefault();
    const nx0 = Math.max(0, x0 - step);
    item.bbox = [y0, nx0, y1, nx0 + w];
    renderSpatialLayoutEditor();
  } else if (ev.key === "ArrowRight") {
    ev.preventDefault();
    const nx0 = Math.min(canvasW - w, x0 + step);
    item.bbox = [y0, nx0, y1, nx0 + w];
    renderSpatialLayoutEditor();
  } else if (ev.key === "ArrowUp") {
    ev.preventDefault();
    const ny0 = Math.max(0, y0 - step);
    item.bbox = [ny0, x0, ny0 + h, x1];
    renderSpatialLayoutEditor();
  } else if (ev.key === "ArrowDown") {
    ev.preventDefault();
    const ny0 = Math.min(canvasH - h, y0 + step);
    item.bbox = [ny0, x0, ny0 + h, x1];
    renderSpatialLayoutEditor();
  } else if (ev.key === "Tab") {
    ev.preventDefault();
    cycleOverlappingBoxes();
  }
}