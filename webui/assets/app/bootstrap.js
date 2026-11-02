// webui/assets/app/bootstrap.js
import {
  state,
  el,
  DEFAULT_LAYOUT_ITEMS,
  DEFAULT_SPECULATIVE_COT_TOKENS,
  parseFiniteInt,
  ensureBoxZOrders,
  updateProjectBadge,
  applyThinkingTraceToDeck,
  syncSpecCotInputDisabledState,
  showStudioToast
} from "./state.js";
import {
  showLoginModal,
  authenticateUser,
  applyThemeVariables,
  loadThemeRegistry,
  pollHealthTelemetry,
  uploadDataUrlToVault
} from "./api.js";
import {
  snapCoord,
  getTrackedCanvasDims,
  scaleBboxToCanvas,
  computeStageGeometry,
  syncStageAspectAndGuides,
  syncUnderlayLayers,
  renderBindingsCard,
  renderViewportBoxesHud,
  applyCanvasResolutionChange,
  matchPrimaryAssetAspect
} from "./geometry.js";
import {
  updateCompiledJsonPreview,
  cycleOverlappingBoxes,
  renderSpatialLayoutEditor,
  handleStageKeydown
} from "./layout_editor.js";
import {
  initInpaintCanvas,
  undoMaskStroke,
  redoMaskStroke
} from "./inpaint_studio.js";
import {
  resetToNewProjectDraft,
  saveCurrentProject,
  importProjectFromFile,
  promoteGalleryImageToVault,
  stageDefaultTestSet,
  loadV11SpatioVisualGraphPreset,
  renderReferenceVault
} from "./vault_projects.js";
import {
  triggerSynthesis,
  saveActiveDeckAsEngineDefault
} from "./synthesis.js";

let bootstrapped = false;

export function bindResizeObservers() {
  if (typeof ResizeObserver === "undefined") return;
  let rafId = 0;
  const observer = new ResizeObserver(() => {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      syncStageAspectAndGuides();
      renderSpatialLayoutEditor();
    });
  });
  for (const id of ["spatialStageContainer", "inpaintStageWrap", "viewportCanvasWrap"]) {
    const node = el(id);
    if (node) observer.observe(node);
  }
}

export function bindEventListeners() {
  el("authBtn").addEventListener("click", showLoginModal);
  el("loginSubmitBtn").addEventListener("click", () => {
    authenticateUser(el("loginUsernameInput").value);
  });
  el("loginUsernameInput").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") authenticateUser(el("loginUsernameInput").value);
  });

  el("themeSelect").addEventListener("change", (ev) => {
    applyThemeVariables(ev.target.value);
  });

  const modeBadge = el("layoutActiveModeBadge");
  if (modeBadge) {
    modeBadge.addEventListener("click", () => {
      const sel = el("promptModeSelect");
      sel.value = sel.value === "layout_json" ? "natural" : "layout_json";
      updateCompiledJsonPreview();
    });
  }

  const traceModeSel = el("thinkingTraceModeSelect");
  const specTokInput = el("specCotTokensInput");
  const traceRawInput = el("thinkingTraceInput");
  if (traceModeSel) {
    traceModeSel.addEventListener("change", () => {
      const mode = traceModeSel.value;
      if (mode === "") {
        if (traceRawInput) traceRawInput.value = "";
      } else if (mode === "auto") {
        if (traceRawInput) traceRawInput.value = "auto";
        el("enableThinkingCheck").checked = true;
      } else if (mode === "speculative") {
        const budget = parseFiniteInt(
          specTokInput ? specTokInput.value : DEFAULT_SPECULATIVE_COT_TOKENS,
          DEFAULT_SPECULATIVE_COT_TOKENS,
          4,
          512
        );
        if (traceRawInput) traceRawInput.value = `speculative:${budget}`;
        el("enableThinkingCheck").checked = true;
      }
      syncSpecCotInputDisabledState();
      updateCompiledJsonPreview();
    });
  }
  if (specTokInput) {
    specTokInput.addEventListener("input", () => {
      const budget = parseFiniteInt(
        specTokInput.value,
        DEFAULT_SPECULATIVE_COT_TOKENS,
        4,
        512
      );
      if (traceModeSel && traceModeSel.value === "speculative") {
        if (traceRawInput) traceRawInput.value = `speculative:${budget}`;
        updateCompiledJsonPreview();
      }
    });
  }
  if (traceRawInput) {
    traceRawInput.addEventListener("input", () => {
      applyThinkingTraceToDeck(traceRawInput.value);
      updateCompiledJsonPreview();
    });
  }

  el("newProjectBtn").addEventListener("click", resetToNewProjectDraft);
  el("saveProjectBtn").addEventListener("click", () => saveCurrentProject(false));
  el("vaultSaveProjectBtn").addEventListener("click", () => saveCurrentProject(false));
  el("exportProjectBtn").addEventListener("click", () => saveCurrentProject(true));
  el("importProjectInput").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    try {
      await importProjectFromFile(file);
    } finally {
      ev.target.value = "";
    }
  });

  el("toggleViewportHudBtn").addEventListener("click", () => {
    state.showViewportHud = !state.showViewportHud;
    el("toggleViewportHudBtn").classList.toggle("active", state.showViewportHud);
    renderViewportBoxesHud();
  });
  el("toggleCompareSplitBtn").addEventListener("click", () => {
    state.showCompareSplit = !state.showCompareSplit;
    el("toggleCompareSplitBtn").classList.toggle("active", state.showCompareSplit);
    syncUnderlayLayers();
  });
  el("viewportCompareSlider").addEventListener("input", syncUnderlayLayers);
  el("promoteViewportInitBtn").addEventListener("click", async () => {
    if (state.activeImageId) {
      await promoteGalleryImageToVault(state.activeImageId, "init", state.activeVariantIndex);
    }
  });
  el("promoteViewportRefBtn").addEventListener("click", async () => {
    if (state.activeImageId) {
      await promoteGalleryImageToVault(state.activeImageId, "ref", state.activeVariantIndex);
    }
  });

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = el(btn.getAttribute("data-tab"));
      if (target) target.classList.add("active");
      syncStageAspectAndGuides();
      renderSpatialLayoutEditor();
      syncUnderlayLayers();
    });
  });

  const vaultPanel = document.querySelector(".vault-panel");
  document.querySelectorAll("[data-vault-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-vault-view]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (vaultPanel) {
        vaultPanel.setAttribute("data-vault-mode", btn.getAttribute("data-vault-view"));
      }
    });
  });

  document.querySelectorAll("[data-aspect]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const [wStr, hStr] = btn.getAttribute("data-aspect").split("x");
      const prev = getTrackedCanvasDims();
      applyCanvasResolutionChange(prev.width, prev.height, parseInt(wStr, 10), parseInt(hStr, 10));
    });
  });
  el("matchAssetAspectBtn").addEventListener("click", matchPrimaryAssetAspect);

  el("synthesizeBtn").addEventListener("click", triggerSynthesis);
  el("saveDefaultPresetBtn").addEventListener("click", saveActiveDeckAsEngineDefault);
  el("stageDefaultRefsBtn").addEventListener("click", stageDefaultTestSet);

  el("clearBindingsBtn").addEventListener("click", () => {
    state.bindings.init_image_path = null;
    state.bindings.inpaint_mask_path = null;
    state.bindings.control_image_path = null;
    state.bindings.reference_image_paths = [];
    renderBindingsCard();
    renderReferenceVault();
    renderSpatialLayoutEditor();
  });

  el("addObjectBoxBtn").addEventListener("click", () => {
    const geom = computeStageGeometry("spatialStageContainer");
    const y0 = snapCoord(geom.canvasH * 0.25, geom.canvasH);
    const x0 = snapCoord(geom.canvasW * 0.25, geom.canvasW);
    const y1 = snapCoord(geom.canvasH * 0.75, geom.canvasH);
    const x1 = snapCoord(geom.canvasW * 0.75, geom.canvasW);
    const maxZ = state.layoutItems.reduce((m, b) => Math.max(m, b.z_order || 1), 0);
    state.layoutItems.push({
      label: "object",
      category: "object",
      caption: "New spatial object region",
      bbox: [y0, x0, y1, x1],
      enabled: true,
      z_order: maxZ + 1,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    });
    state.selectedBoxIndex = state.layoutItems.length - 1;
    renderSpatialLayoutEditor();
  });

  el("addCharacterBoxBtn").addEventListener("click", () => {
    const geom = computeStageGeometry("spatialStageContainer");
    const y0 = snapCoord(geom.canvasH * 0.15, geom.canvasH);
    const x0 = snapCoord(geom.canvasW * 0.28, geom.canvasW);
    const y1 = snapCoord(geom.canvasH * 0.88, geom.canvasH);
    const x1 = snapCoord(geom.canvasW * 0.72, geom.canvasW);
    const maxZ = state.layoutItems.reduce((m, b) => Math.max(m, b.z_order || 1), 0);
    state.layoutItems.push({
      label: "character",
      category: "character",
      caption: "Character subject in scene",
      bbox: [y0, x0, y1, x1],
      enabled: true,
      z_order: maxZ + 1,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    });
    state.selectedBoxIndex = state.layoutItems.length - 1;
    renderSpatialLayoutEditor();
  });

  el("addTextBoxBtn").addEventListener("click", () => {
    const geom = computeStageGeometry("spatialStageContainer");
    const y0 = snapCoord(geom.canvasH * 0.12, geom.canvasH);
    const x0 = snapCoord(geom.canvasW * 0.25, geom.canvasW);
    const y1 = snapCoord(geom.canvasH * 0.28, geom.canvasH);
    const x1 = snapCoord(geom.canvasW * 0.75, geom.canvasW);
    const maxZ = state.layoutItems.reduce((m, b) => Math.max(m, b.z_order || 1), 0);
    state.layoutItems.push({
      label: "text",
      category: "text",
      text: "SOVEREIGN 2026",
      bbox: [y0, x0, y1, x1],
      enabled: true,
      z_order: maxZ + 1,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    });
    state.selectedBoxIndex = state.layoutItems.length - 1;
    renderSpatialLayoutEditor();
  });

  if (el("loadV11GraphBtn")) {
    el("loadV11GraphBtn").addEventListener("click", loadV11SpatioVisualGraphPreset);
  }
  el("cycleOverlapBtn").addEventListener("click", () => cycleOverlappingBoxes());

  el("clearAllBoxesBtn").addEventListener("click", () => {
    state.layoutItems = [];
    state.selectedBoxIndex = 0;
    renderSpatialLayoutEditor();
  });

  el("resetDefaultBoxesBtn").addEventListener("click", () => {
    const geom = computeStageGeometry("spatialStageContainer");
    const cloned = JSON.parse(JSON.stringify(DEFAULT_LAYOUT_ITEMS));
    for (const b of cloned) {
      if (Array.isArray(b.bbox) && b.bbox.length === 4) {
        b.bbox = scaleBboxToCanvas(b.bbox, 1024, 1024, geom.canvasW, geom.canvasH);
      }
    }
    state.layoutItems = cloned;
    ensureBoxZOrders();
    state.selectedBoxIndex = 0;
    renderSpatialLayoutEditor();
  });

  el("spatialStage").addEventListener("keydown", handleStageKeydown);

  window.addEventListener("keydown", (ev) => {
    const inpaintActive = el("inpaintTab") && el("inpaintTab").classList.contains("active");
    if (!inpaintActive) return;
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      if (ev.shiftKey) redoMaskStroke();
      else undoMaskStroke();
    } else if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "y") {
      ev.preventDefault();
      redoMaskStroke();
    }
  });

  el("promptInput").addEventListener("input", updateCompiledJsonPreview);
  el("layoutSummaryInput").addEventListener("input", updateCompiledJsonPreview);
  el("promptModeSelect").addEventListener("change", updateCompiledJsonPreview);
  el("systemPromptInput").addEventListener("input", updateCompiledJsonPreview);
  el("enableThinkingCheck").addEventListener("change", updateCompiledJsonPreview);
  el("addVisionIdCheck").addEventListener("change", updateCompiledJsonPreview);
  el("maxSeqLenInput").addEventListener("change", updateCompiledJsonPreview);
  el("maxVitPatchesInput").addEventListener("change", updateCompiledJsonPreview);
  el("editFormatSelect").addEventListener("change", renderSpatialLayoutEditor);
  el("inputRoleSelect").addEventListener("change", renderSpatialLayoutEditor);
  el("framingModeSelect").addEventListener("change", renderSpatialLayoutEditor);
  el("boxAttnIsolationInput").addEventListener("input", renderSpatialLayoutEditor);
  el("compositeBoxImagesCheck").addEventListener("change", renderSpatialLayoutEditor);
  el("preserveOutsideBoxesCheck").addEventListener("change", renderBindingsCard);
  el("maskBlendModeSelect").addEventListener("change", renderBindingsCard);
  el("invertMaskCheck").addEventListener("change", renderBindingsCard);
  if (el("refZeroTimestepSelect")) {
    el("refZeroTimestepSelect").addEventListener("change", updateCompiledJsonPreview);
  }
  if (el("refIsolatedKvSelect")) {
    el("refIsolatedKvSelect").addEventListener("change", updateCompiledJsonPreview);
  }
  el("layoutUnderlaySelect").addEventListener("change", syncUnderlayLayers);
  el("layoutUnderlayOpacity").addEventListener("input", syncUnderlayLayers);
  el("inpaintUnderlayOpacity").addEventListener("input", syncUnderlayLayers);

  el("widthInput").addEventListener("change", () => {
    const prev = getTrackedCanvasDims();
    const newW = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
    const newH = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
    applyCanvasResolutionChange(prev.width, prev.height, newW, newH);
  });
  el("heightInput").addEventListener("change", () => {
    const prev = getTrackedCanvasDims();
    const newW = parseFiniteInt(el("widthInput").value, 1024, 256, 2048);
    const newH = parseFiniteInt(el("heightInput").value, 1024, 256, 2048);
    applyCanvasResolutionChange(prev.width, prev.height, newW, newH);
  });

  el("referenceFileInput").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const uploaded = await uploadDataUrlToVault(
          file.name,
          el("inputRoleSelect").value,
          reader.result
        );
        showStudioToast(`Uploaded reference: ${uploaded.ref_id}`, "success");
      } catch (err) {
        showStudioToast(`Upload failed: ${err.message}`, "error");
      } finally {
        ev.target.value = "";
      }
    };
    reader.readAsDataURL(file);
  });
}

export async function bootstrapStudio() {
  if (bootstrapped) return;
  bootstrapped = true;

  ensureBoxZOrders();
  bindEventListeners();
  syncSpecCotInputDisabledState();
  initInpaintCanvas();
  bindResizeObservers();
  updateProjectBadge();

  await loadThemeRegistry();
  await pollHealthTelemetry(true);
  syncStageAspectAndGuides();
  renderBindingsCard();
  renderSpatialLayoutEditor();

  const savedUser = localStorage.getItem("vs_last_username") || "administrator";
  el("loginUsernameInput").value = savedUser;
  await authenticateUser(savedUser);

  setInterval(() => pollHealthTelemetry(false), 8000);
}