// webui/assets/app/vault_projects.js
import {
  state,
  el,
  DEFAULT_PROMPT_TEXT,
  DEFAULT_LAYOUT_ITEMS,
  V11_SPATIOVISUAL_BOXES,
  ensureBoxZOrders,
  updateProjectBadge,
  applyThinkingTraceToDeck,
  showStudioToast
} from "./state.js";
import { apiFetch, showLoginModal } from "./api.js";
import {
  scaleBboxToCanvas,
  computeStageGeometry,
  renderBindingsCard,
  renderViewportBoxesHud,
  syncUnderlayLayers
} from "./geometry.js";
import {
  syncTextEditToPrompt,
  renderSpatialLayoutEditor
} from "./layout_editor.js";
import {
  updateMaskHistoryButtons,
  hasInpaintCanvasContent,
  loadCanvasFromDataUrl
} from "./inpaint_studio.js";
import { collectDeckPayload } from "./synthesis.js";

export async function refreshReferences() {
  if (!state.token) return;
  try {
    const data = await apiFetch("api/v1/references");
    state.references = (data && Array.isArray(data.references)) ? data.references : [];
    renderReferenceVault();
    renderBindingsCard();
    renderSpatialLayoutEditor();
  } catch (_) {}
}

export async function refreshProjects() {
  if (!state.token) return;
  try {
    const data = await apiFetch("api/v1/projects");
    state.projects = (data && Array.isArray(data.projects)) ? data.projects : [];
    renderProjectVault();
  } catch (_) {}
}

export function renderReferenceVault() {
  const container = el("referenceList");
  const badge = el("assetCountBadge");
  if (badge) badge.textContent = String(state.references.length);
  if (!container) return;
  if (!state.references.length) {
    container.innerHTML = `<div class="muted-note">No reference assets staged in your user vault.</div>`;
    return;
  }

  const selIdx = Math.max(0, Math.min(state.selectedBoxIndex, Math.max(0, state.layoutItems.length - 1)));
  const activeBox = state.layoutItems[selIdx] || null;
  const boxBtnLabel = activeBox ? `Box #${selIdx + 1}` : "Box";

  container.innerHTML = "";
  for (const ref of state.references) {
    const card = document.createElement("div");
    card.className = "ref-card";
    const sizeKb = (ref.size_bytes / 1024).toFixed(1);
    const isGlobalRef = state.bindings.reference_image_paths.includes(ref.ref_id);
    const isSelBoxBound = Boolean(activeBox && activeBox.image_path === ref.ref_id);

    const boundBoxNumbers = [];
    state.layoutItems.forEach((b, idx) => {
      if (b && b.enabled !== false && b.image_path === ref.ref_id) {
        boundBoxNumbers.push(`#${idx + 1}`);
      }
    });

    const dimsLabel = ref.width && ref.height ? `${ref.width}x${ref.height}` : "1024x1024";
    const alphaBadge = ref.has_alpha ? `<span class="meta-tag alpha-tag" title="Contains translucent alpha channel">ALPHA</span>` : "";
    const vizjBadge = ref.is_vizj ? `<span class="meta-tag vizj-tag" title="Tier-2 Zero-Copy .VIZJ Container">VIZJ</span>` : "";
    const boxBadge = boundBoxNumbers.length > 0
      ? `<span class="meta-tag vizj-tag" title="Scoped to Spatial Layout Box(es) ${boundBoxNumbers.join(", ")}">BOX ${boundBoxNumbers.join(",")}</span>`
      : "";

    card.innerHTML = `
      <div class="ref-card-top">
        <span title="${ref.ref_id}">${ref.ref_id}</span>
        <button type="button" class="btn-link" data-del-ref="${ref.ref_id}">Delete</button>
      </div>
      <div class="ref-meta-badges">
        <span class="meta-tag">${dimsLabel}</span>
        <span class="meta-tag">${sizeKb} KB</span>
        ${alphaBadge}
        ${vizjBadge}
        ${boxBadge}
      </div>
      <div class="ref-bind-btns">
        <button type="button" class="btn btn-secondary compact-btn ${isGlobalRef ? "active" : ""}" data-bind-role="ref" data-ref-id="${ref.ref_id}" title="Bind as full-canvas Global 4D Reference">4D Ref</button>
        <button type="button" class="btn btn-secondary compact-btn ${isSelBoxBound ? "active" : ""}" data-bind-role="box" data-ref-id="${ref.ref_id}" ${activeBox ? "" : "disabled"} title="Bind/unbind strictly to selected Spatial Layout ${boxBtnLabel} (never renders as full-stage background)">${boxBtnLabel}</button>
        <button type="button" class="btn btn-secondary compact-btn ${state.bindings.init_image_path === ref.ref_id ? "active" : ""}" data-bind-role="init" data-ref-id="${ref.ref_id}" title="Bind as full-canvas Init Image">Init Img</button>
        <button type="button" class="btn btn-secondary compact-btn ${state.bindings.inpaint_mask_path === ref.ref_id ? "active" : ""}" data-bind-role="mask" data-ref-id="${ref.ref_id}" title="Bind as full-canvas Inpaint Mask">Mask</button>
        <button type="button" class="btn btn-secondary compact-btn ${state.bindings.control_image_path === ref.ref_id ? "active" : ""}" data-bind-role="ctrl" data-ref-id="${ref.ref_id}" title="Bind as full-canvas Control Map">Control</button>
      </div>
    `;
    container.appendChild(card);
  }

  container.querySelectorAll("[data-del-ref]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const refId = btn.getAttribute("data-del-ref");
      await apiFetch(`api/v1/references/${encodeURIComponent(refId)}`, { method: "DELETE" });
      state.bindings.reference_image_paths = state.bindings.reference_image_paths.filter((id) => id !== refId);
      if (state.bindings.init_image_path === refId) state.bindings.init_image_path = null;
      if (state.bindings.inpaint_mask_path === refId) state.bindings.inpaint_mask_path = null;
      if (state.bindings.control_image_path === refId) state.bindings.control_image_path = null;
      for (const b of state.layoutItems) {
        if (b.image_path === refId) b.image_path = "";
      }
      renderBindingsCard();
      await refreshReferences();
    });
  });

  container.querySelectorAll("[data-bind-role]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const role = btn.getAttribute("data-bind-role");
      const refId = btn.getAttribute("data-ref-id");
      if (role === "ref") {
        const idx = state.bindings.reference_image_paths.indexOf(refId);
        if (idx >= 0) {
          state.bindings.reference_image_paths.splice(idx, 1);
        } else {
          state.bindings.reference_image_paths.push(refId);
        }
      } else if (role === "box") {
        const targetIdx = Math.max(0, Math.min(state.selectedBoxIndex, state.layoutItems.length - 1));
        const targetBox = state.layoutItems[targetIdx];
        if (targetBox) {
          targetBox.image_path = targetBox.image_path === refId ? "" : refId;
          targetBox.enabled = true;
        }
      } else if (role === "init") {
        state.bindings.init_image_path = state.bindings.init_image_path === refId ? null : refId;
      } else if (role === "mask") {
        state.bindings.inpaint_mask_path = state.bindings.inpaint_mask_path === refId ? null : refId;
      } else if (role === "ctrl") {
        state.bindings.control_image_path = state.bindings.control_image_path === refId ? null : refId;
      }
      renderBindingsCard();
      renderReferenceVault();
      renderSpatialLayoutEditor();
    });
  });
}

export function renderProjectVault() {
  const container = el("projectList");
  const badge = el("projectCountBadge");
  if (badge) badge.textContent = String(state.projects.length);
  if (!container) return;
  if (!state.projects.length) {
    container.innerHTML = `<div class="muted-note">Save a project to bundle your layout boxes, prompt, mask canvas, and reference assets.</div>`;
    return;
  }
  container.innerHTML = "";
  for (const proj of state.projects) {
    const card = document.createElement("div");
    const isCurrent = state.activeProjectId === proj.project_id;
    card.className = `project-card ${isCurrent ? "active-project" : ""}`;
    const sizeKb = ((proj.size_bytes || 0) / 1024).toFixed(1);
    const dims = `${proj.width || 1024}x${proj.height || 1024}`;
    const dlHref = `api/v1/projects/${encodeURIComponent(proj.project_id)}/download?token=${encodeURIComponent(state.token)}`;

    card.innerHTML = `
      <div class="project-card-top">
        <span class="project-title-text" title="${proj.title || proj.project_id}">${proj.title || proj.project_id}</span>
        <button type="button" class="btn-link" data-del-proj="${proj.project_id}">Delete</button>
      </div>
      <div class="ref-meta-badges">
        <span class="meta-tag">${(proj.model_variant || "sft").toUpperCase()}</span>
        <span class="meta-tag">${dims}</span>
        <span class="meta-tag">${proj.asset_count || 0} assets</span>
        <span class="meta-tag">${sizeKb} KB</span>
      </div>
      <div class="project-actions">
        <button type="button" class="btn btn-secondary compact-btn" data-load-proj="${proj.project_id}">Load Project</button>
        <a class="btn btn-secondary compact-btn" href="${dlHref}" download="${proj.project_id}.vsproj">Export .VSPROJ</a>
      </div>
    `;
    container.appendChild(card);
  }

  container.querySelectorAll("[data-load-proj]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const projId = btn.getAttribute("data-load-proj");
      await loadProjectFromVault(projId);
    });
  });

  container.querySelectorAll("[data-del-proj]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const projId = btn.getAttribute("data-del-proj");
      await apiFetch(`api/v1/projects/${encodeURIComponent(projId)}`, { method: "DELETE" });
      if (state.activeProjectId === projId) {
        state.activeProjectId = null;
        state.activeProjectTitle = null;
        updateProjectBadge();
      }
      await refreshProjects();
    });
  });
}

export function resetToNewProjectDraft() {
  state.activeProjectId = null;
  state.activeProjectTitle = null;
  state.activeImageId = null;
  state.lastTelemetry = null;
  state.lastSpatioVisualGraph = null;
  el("titleInput").value = "Untitled Vision Master";
  el("promptInput").value = DEFAULT_PROMPT_TEXT;
  el("negativePromptInput").value = "";
  el("systemPromptInput").value = "";
  applyThinkingTraceToDeck("");
  el("layoutSummaryInput").value = "";
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
  state.bindings.init_image_path = null;
  state.bindings.inpaint_mask_path = null;
  state.bindings.control_image_path = null;
  state.bindings.reference_image_paths = [];
  if (el("refZeroTimestepSelect")) el("refZeroTimestepSelect").value = "";
  if (el("refIsolatedKvSelect")) el("refIsolatedKvSelect").value = "";

  const canvas = el("inpaintCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    state.maskUndoStack.length = 0;
    state.maskRedoStack.length = 0;
    updateMaskHistoryButtons();
  }
  updateProjectBadge();
  renderBindingsCard();
  renderReferenceVault();
  renderProjectVault();
  renderSpatialLayoutEditor();
}

export async function loadV11SpatioVisualGraphPreset() {
  const hasRefA = state.references.some((r) => r.ref_id === "ref_subject_a.png");
  const hasCutC = state.references.some((r) => r.ref_id === "ref_cutout_c.png");
  const hasCtrl = state.references.some((r) => r.ref_id === "control_map.png");
  if (state.token && (!hasRefA || !hasCutC || !hasCtrl)) {
    try {
      await apiFetch("api/v1/references/stage-defaults", { method: "POST" });
      await refreshReferences();
    } catch (_) {}
  }

  const geom = computeStageGeometry("spatialStageContainer");
  const v11Boxes = JSON.parse(JSON.stringify(V11_SPATIOVISUAL_BOXES));
  for (const b of v11Boxes) {
    if (Array.isArray(b.bbox) && b.bbox.length === 4) {
      b.bbox = scaleBboxToCanvas(b.bbox, 1024, 1024, geom.canvasW, geom.canvasH);
    }
  }
  state.layoutItems = v11Boxes;
  ensureBoxZOrders();
  state.selectedBoxIndex = 0;
  el("promptModeSelect").value = "layout_json";
  el("boxAttnIsolationInput").value = "0.85";
  el("compositeBoxImagesCheck").checked = true;
  el("enableThinkingCheck").checked = true;
  applyThinkingTraceToDeck("auto");
  syncTextEditToPrompt(
    "Remember to test SFT & Base permutations @ 1024x1024",
    "SOVEREIGN V11 CONVERGED"
  );
  renderBindingsCard();
  renderReferenceVault();
  renderSpatialLayoutEditor();
  showStudioToast(
    "Loaded V11 Spatio-Visual Graph (Co-Op Depth + Cutout + Carved Typography)",
    "success"
  );
}

export async function saveCurrentProject(downloadAfterSave = false) {
  if (!state.token) {
    showLoginModal();
    return;
  }
  const saveBtn = el("saveProjectBtn");
  const expBtn = el("exportProjectBtn");
  saveBtn.disabled = true;
  expBtn.disabled = true;
  try {
    const draft = collectDeckPayload();
    const canvas = el("inpaintCanvas");
    const maskDataUrl = (canvas && hasInpaintCanvasContent()) ? canvas.toDataURL("image/png") : null;
    const res = await apiFetch("api/v1/projects/save", {
      method: "POST",
      body: {
        project_id: state.activeProjectId || null,
        title: draft.title,
        working_draft: draft,
        telemetry: state.lastTelemetry,
        active_image_id: state.activeImageId,
        inpaint_canvas_data_url: maskDataUrl,
        include_all_vault_refs: false
      }
    });
    if (res && res.project) {
      state.activeProjectId = res.project.project_id;
      state.activeProjectTitle = res.project.title;
      updateProjectBadge();
      await refreshProjects();
      showStudioToast(`Saved project: ${res.project.title}`, "success");
      if (downloadAfterSave) {
        const dlUrl = `api/v1/projects/${encodeURIComponent(res.project.project_id)}/download?token=${encodeURIComponent(state.token)}`;
        const a = document.createElement("a");
        a.href = dlUrl;
        a.download = `${res.project.project_id}.vsproj`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
    }
  } catch (err) {
    showStudioToast(`Project save failed: ${err.message}`, "error");
  } finally {
    saveBtn.disabled = false;
    expBtn.disabled = false;
  }
}

export async function loadProjectFromVault(projectId) {
  if (!state.token) {
    showLoginModal();
    return;
  }
  try {
    const res = await apiFetch(`api/v1/projects/${encodeURIComponent(projectId)}/load`, {
      method: "POST"
    });
    if (!res) return;
    state.activeProjectId = res.project_id;
    state.activeProjectTitle = res.title;
    updateProjectBadge();
    await Promise.all([refreshReferences(), refreshProjects()]);
    restoreDraftToStudio({
      title: res.title,
      working_draft: res.working_draft,
      telemetry: res.telemetry,
      image_url: res.preview_url
    });
    if (res.inpaint_canvas_data_url) {
      loadCanvasFromDataUrl(res.inpaint_canvas_data_url);
    }
    showStudioToast(`Loaded project: ${res.title || res.project_id}`, "success");
  } catch (err) {
    showStudioToast(`Failed loading project: ${err.message}`, "error");
  }
}

export async function importProjectFromFile(file) {
  if (!state.token) {
    showLoginModal();
    return;
  }
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const res = await apiFetch("api/v1/projects/import", {
        method: "POST",
        body: {
          filename: file.name,
          data_url: reader.result
        }
      });
      if (!res) return;
      state.activeProjectId = res.project_id;
      state.activeProjectTitle = res.title;
      updateProjectBadge();
      await Promise.all([refreshReferences(), refreshProjects()]);
      restoreDraftToStudio({
        title: res.title,
        working_draft: res.working_draft,
        telemetry: res.telemetry,
        image_url: res.preview_url
      });
      if (res.inpaint_canvas_data_url) {
        loadCanvasFromDataUrl(res.inpaint_canvas_data_url);
      }
      showStudioToast(`Imported .vsproj: ${res.title || res.project_id}`, "success");
    } catch (err) {
      showStudioToast(`Failed importing .vsproj package: ${err.message}`, "error");
    }
  };
  reader.readAsDataURL(file);
}

export async function promoteGalleryImageToVault(imageId, bindMode = "ref", variantIndex = 1) {
  if (!state.token || !imageId) return;
  try {
    const meta = await apiFetch(`api/v1/references/from-image/${encodeURIComponent(imageId)}`, {
      method: "POST",
      body: {
        role: "photo_ref",
        variant_index: variantIndex
      }
    });
    if (!meta || !meta.ref_id) return;
    await refreshReferences();
    if (bindMode === "init") {
      state.bindings.init_image_path = meta.ref_id;
      showStudioToast(`Bound ${meta.ref_id} as Init Image`, "success");
    } else if (bindMode === "box") {
      const targetIdx = Math.max(0, Math.min(state.selectedBoxIndex, state.layoutItems.length - 1));
      if (state.layoutItems[targetIdx]) {
        state.layoutItems[targetIdx].image_path = meta.ref_id;
        state.layoutItems[targetIdx].enabled = true;
        showStudioToast(`Bound ${meta.ref_id} to Box #${targetIdx + 1}`, "success");
      }
    } else {
      if (!state.bindings.reference_image_paths.includes(meta.ref_id)) {
        state.bindings.reference_image_paths.push(meta.ref_id);
      }
      showStudioToast(`Bound ${meta.ref_id} as 4D Reference`, "success");
    }
    renderBindingsCard();
    renderReferenceVault();
    renderSpatialLayoutEditor();
  } catch (err) {
    showStudioToast(`Failed promoting image to vault: ${err.message}`, "error");
  }
}

export async function stageDefaultTestSet() {
  if (!state.token) {
    showLoginModal();
    return;
  }
  const btn = el("stageDefaultRefsBtn");
  btn.disabled = true;
  try {
    const res = await apiFetch("api/v1/references/stage-defaults", { method: "POST" });
    await refreshReferences();
    const count = res && Array.isArray(res.staged_files) ? res.staged_files.length : 6;
    showStudioToast(`Staged ${count} deterministic test assets in Vault`, "success");
  } catch (err) {
    showStudioToast(`Failed staging test set: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

export function updateViewportAndTelemetry(jobStatus) {
  const frame = el("viewportStageFrame");
  const img = el("masterOutputImage");
  const dlPng = el("downloadPngBtn");
  const strip = el("batchThumbStrip");
  const promoInitBtn = el("promoteViewportInitBtn");
  const promoRefBtn = el("promoteViewportRefBtn");

  const resolvedId = jobStatus.job_id || jobStatus.image_id || null;
  if (resolvedId) {
    state.activeImageId = resolvedId;
    state.activeVariantIndex = 1;
  }

  if (jobStatus.image_url) {
    const allUrls = [jobStatus.image_url].concat(jobStatus.additional_image_urls || []);
    img.src = `${allUrls[0]}?t=${Date.now()}`;
    img.hidden = false;
    if (frame) frame.hidden = false;
    el("emptyViewportState").hidden = true;
    dlPng.href = allUrls[0];
    dlPng.hidden = false;
    if (promoInitBtn) promoInitBtn.hidden = !state.activeImageId;
    if (promoRefBtn) promoRefBtn.hidden = !state.activeImageId;

    if (strip) {
      if (allUrls.length > 1) {
        strip.innerHTML = "";
        allUrls.forEach((u, idx) => {
          const th = document.createElement("img");
          th.className = `batch-thumb ${idx === 0 ? "active" : ""}`;
          th.src = `${u}?t=${Date.now()}`;
          th.alt = `Variation ${idx + 1}`;
          th.addEventListener("click", () => {
            img.src = `${u}?t=${Date.now()}`;
            dlPng.href = u;
            state.activeVariantIndex = idx + 1;
            strip.querySelectorAll(".batch-thumb").forEach((elThumb) => elThumb.classList.remove("active"));
            th.classList.add("active");
            syncUnderlayLayers();
          });
          strip.appendChild(th);
        });
        strip.hidden = false;
      } else {
        strip.hidden = true;
      }
    }
    renderViewportBoxesHud();
    syncUnderlayLayers();
  }

  const dlVizj = el("downloadVizjBtn");
  if (jobStatus.vizj_url) {
    dlVizj.href = jobStatus.vizj_url;
    dlVizj.hidden = false;
  } else {
    dlVizj.hidden = true;
  }

  const t = jobStatus.telemetry;
  if (t) {
    state.lastTelemetry = t;
    el("tSolverVal").textContent = `${(t.early_solver || "heun").toUpperCase()} -> ${(t.late_solver || "ipndm").toUpperCase()} (${t.total_nfe} NFE)`;
    el("tLatencyVal").textContent = `${Number(t.generation_time_seconds).toFixed(2)}s (${Number(t.step_latency_ms).toFixed(0)}ms/st)`;
    const contig = t.wddm_contiguous_avail_gb !== undefined ? ` | ${Number(t.wddm_contiguous_avail_gb).toFixed(1)}G free` : "";
    el("tVramVal").textContent = `${Number(t.peak_vram_gb).toFixed(2)}G${contig}`;
    const boxIso = Number(t.box_attn_isolation || 0);
    const splatTag = t.composite_box_images ? " | Splat" : "";
    const isoTag = boxIso > 0 ? ` | Iso=${boxIso.toFixed(2)}` : (t.reference_isolated_kv ? " | IsoKV" : "");
    const sv = t.spatio_visual_graph || {};
    const carveCnt = Array.isArray(sv.foreground_carveouts) ? sv.foreground_carveouts.length : 0;
    const carveTag = carveCnt > 0 ? ` | Carve=${carveCnt}` : "";
    el("tRoutingVal").textContent = `${t.edit_format || "standard"}${isoTag}${splatTag}${carveTag}`;
    el("tSignalVal").textContent = `${Number(t.shannon_entropy_bits).toFixed(2)}b / ${Number(t.laplacian_sharpness).toFixed(3)}`;
    el("tLumaVal").textContent = `${Number(t.mean_luminance).toFixed(3)} / ${Number(t.rms_contrast).toFixed(3)}`;
    let cotSuffix = "";
    if (t.speculative_cot_tokens && Number(t.speculative_cot_tokens) > 0) {
      const rt = t.reanchor_telemetry;
      const accStr = rt && rt.mean_acceptance_length
        ? `, acc=${Number(rt.mean_acceptance_length).toFixed(2)}`
        : "";
      cotSuffix = ` | SpecCoT ${t.speculative_cot_tokens}t${accStr}`;
    } else if (typeof t.thinking_trace_mode === "string" && ["auto", "spatial", "compile"].includes(t.thinking_trace_mode.toLowerCase())) {
      cotSuffix = " | AutoCoT";
    }
    el("tCacheVal").textContent = `Hit=${Boolean(t.prompt_cache_hit)} (Pfx ${t.common_prefix_tokens || 0}t)${cotSuffix}`;
  }
}

export function restoreDraftToStudio(item) {
  const draft = item.working_draft || item.recipe || {};
  if (draft.title || item.title) el("titleInput").value = draft.title || item.title;
  if (draft.model_variant) el("modelVariantSelect").value = draft.model_variant;
  if (draft.prompt_mode) el("promptModeSelect").value = draft.prompt_mode;
  if (draft.prompt !== undefined && draft.prompt !== null) el("promptInput").value = draft.prompt;
  if (draft.layout_summary !== undefined && el("layoutSummaryInput")) {
    const sumVal = draft.layout_summary || "";
    el("layoutSummaryInput").value = sumVal === draft.prompt ? "" : sumVal;
  }
  if (draft.negative_prompt !== undefined && draft.negative_prompt !== null) {
    el("negativePromptInput").value = draft.negative_prompt;
  }
  if (draft.system_prompt !== undefined) el("systemPromptInput").value = draft.system_prompt || "";
  if (draft.thinking_trace !== undefined) {
    applyThinkingTraceToDeck(draft.thinking_trace || "");
  }
  if (draft.add_vision_id !== undefined && draft.add_vision_id !== null) {
    el("addVisionIdCheck").checked = Boolean(draft.add_vision_id);
  }
  if (draft.enable_thinking !== undefined && draft.enable_thinking !== null) {
    el("enableThinkingCheck").checked = Boolean(draft.enable_thinking);
  }
  if (draft.max_sequence_length !== undefined && draft.max_sequence_length !== null) {
    el("maxSeqLenInput").value = draft.max_sequence_length;
  }
  if (draft.max_vit_patches !== undefined && draft.max_vit_patches !== null) {
    el("maxVitPatchesInput").value = draft.max_vit_patches;
  }
  if (draft.width !== undefined && draft.width !== null) el("widthInput").value = draft.width;
  if (draft.height !== undefined && draft.height !== null) el("heightInput").value = draft.height;
  if (draft.num_inference_steps !== undefined && draft.num_inference_steps !== null) {
    el("stepsInput").value = draft.num_inference_steps;
  }
  if (draft.num_images_per_prompt !== undefined && draft.num_images_per_prompt !== null) {
    el("numImagesInput").value = draft.num_images_per_prompt;
  }
  if (draft.early_solver) el("earlySolverSelect").value = draft.early_solver;
  if (draft.late_solver) el("lateSolverSelect").value = draft.late_solver;
  if (draft.handoff_threshold !== undefined && draft.handoff_threshold !== null) {
    el("handoffInput").value = draft.handoff_threshold;
  }
  if (draft.early_cfg !== undefined && draft.early_cfg !== null) el("earlyCfgInput").value = draft.early_cfg;
  if (draft.late_cfg !== undefined && draft.late_cfg !== null) el("lateCfgInput").value = draft.late_cfg;
  if (draft.shift !== undefined && draft.shift !== null) el("shiftInput").value = draft.shift;
  if (Array.isArray(draft.uncond_skip_layers)) {
    el("skipLayersInput").value = draft.uncond_skip_layers.join(", ");
  }
  if (Array.isArray(draft.slg_layers)) {
    el("slgLayersInput").value = draft.slg_layers.join(", ");
  }
  if (draft.slg_scale !== undefined && draft.slg_scale !== null) el("slgScaleInput").value = draft.slg_scale;
  if (draft.slg_start !== undefined && draft.slg_start !== null) el("slgStartInput").value = draft.slg_start;
  if (draft.slg_stop !== undefined && draft.slg_stop !== null) el("slgStopInput").value = draft.slg_stop;
  if (draft.eta !== undefined && draft.eta !== null) el("etaInput").value = draft.eta;
  if (draft.s_noise !== undefined && draft.s_noise !== null) el("sNoiseInput").value = draft.s_noise;
  if (draft.seed !== undefined && draft.seed !== null) el("seedInput").value = draft.seed;

  if (draft.use_v2_grid !== undefined && draft.use_v2_grid !== null) {
    el("useV2GridCheck").checked = Boolean(draft.use_v2_grid);
  }
  if (draft.preserve_canvas_outside_boxes !== undefined && draft.preserve_canvas_outside_boxes !== null) {
    el("preserveOutsideBoxesCheck").checked = Boolean(draft.preserve_canvas_outside_boxes);
  }
  if (draft.composite_box_images_to_init !== undefined && draft.composite_box_images_to_init !== null) {
    el("compositeBoxImagesCheck").checked = Boolean(draft.composite_box_images_to_init);
  }
  if (draft.box_attn_isolation !== undefined && draft.box_attn_isolation !== null) {
    el("boxAttnIsolationInput").value = draft.box_attn_isolation;
  }
  if (draft.uncond_uses_references === true) {
    el("uncondUsesRefsSelect").value = "true";
  } else if (draft.uncond_uses_references === false) {
    el("uncondUsesRefsSelect").value = "false";
  } else {
    el("uncondUsesRefsSelect").value = "";
  }
  if (el("refZeroTimestepSelect")) {
    if (draft.reference_zero_timestep === true) {
      el("refZeroTimestepSelect").value = "true";
    } else if (draft.reference_zero_timestep === false) {
      el("refZeroTimestepSelect").value = "false";
    } else {
      el("refZeroTimestepSelect").value = "";
    }
  }
  if (el("refIsolatedKvSelect")) {
    if (draft.reference_isolated_kv === true) {
      el("refIsolatedKvSelect").value = "true";
    } else if (draft.reference_isolated_kv === false) {
      el("refIsolatedKvSelect").value = "false";
    } else {
      el("refIsolatedKvSelect").value = "";
    }
  }
  if (draft.mask_blend_mode) el("maskBlendModeSelect").value = draft.mask_blend_mode;
  if (draft.invert_inpaint_mask !== undefined && draft.invert_inpaint_mask !== null) {
    el("invertMaskCheck").checked = Boolean(draft.invert_inpaint_mask);
  }
  if (draft.edit_format) el("editFormatSelect").value = draft.edit_format;
  if (draft.framing_mode) el("framingModeSelect").value = draft.framing_mode;
  if (draft.input_role) el("inputRoleSelect").value = draft.input_role;
  if (draft.denoise_strength !== undefined && draft.denoise_strength !== null) {
    el("denoiseInput").value = draft.denoise_strength;
  }
  if (draft.mtf_boost !== undefined && draft.mtf_boost !== null) el("mtfBoostInput").value = draft.mtf_boost;
  if (draft.soft_knee_tau !== undefined && draft.soft_knee_tau !== null) {
    el("softKneeTauInput").value = draft.soft_knee_tau;
  }
  if (draft.mask_feather_radius !== undefined && draft.mask_feather_radius !== null) {
    el("featherInput").value = draft.mask_feather_radius;
  }
  if (draft.latent_clamp_sigma !== undefined && draft.latent_clamp_sigma !== null) {
    el("clampSigmaInput").value = draft.latent_clamp_sigma;
  }
  if (draft.deband_8bit !== undefined && draft.deband_8bit !== null) {
    el("deband8bitCheck").checked = Boolean(draft.deband_8bit);
  }
  if (draft.apply_bayer_dither !== undefined && draft.apply_bayer_dither !== null) {
    el("applyBayerDitherCheck").checked = Boolean(draft.apply_bayer_dither);
  }
  if (draft.use_reference_cache !== undefined && draft.use_reference_cache !== null) {
    el("useRefCacheCheck").checked = Boolean(draft.use_reference_cache);
  }
  if (draft.prefer_vulkan !== undefined && draft.prefer_vulkan !== null) {
    el("preferVulkanCheck").checked = Boolean(draft.prefer_vulkan);
  }
  if (draft.cpu_offload !== undefined && draft.cpu_offload !== null) {
    el("cpuOffloadCheck").checked = Boolean(draft.cpu_offload);
  }
  if (el("prepassWarmupCheck") && draft.prepass_warmup !== undefined && draft.prepass_warmup !== null) {
    el("prepassWarmupCheck").checked = Boolean(draft.prepass_warmup);
  }
  if (draft.export_vizj !== undefined && draft.export_vizj !== null) {
    el("exportVizjCheck").checked = Boolean(draft.export_vizj);
  }
  if (draft.edit_lora_path !== undefined) el("editLoraSelect").value = draft.edit_lora_path || "";
  if (draft.edit_lora_scale !== undefined && draft.edit_lora_scale !== null) {
    el("loraScaleInput").value = draft.edit_lora_scale;
  }

  if (Array.isArray(draft.layout_items)) {
    state.layoutItems = JSON.parse(JSON.stringify(draft.layout_items));
    ensureBoxZOrders();
    state.selectedBoxIndex = 0;
  }

  state.bindings.init_image_path = draft.init_image_path || null;
  state.bindings.inpaint_mask_path = draft.inpaint_mask_path || null;
  state.bindings.control_image_path = draft.control_image_path || null;
  state.bindings.reference_image_paths = Array.isArray(draft.reference_image_paths)
    ? [...draft.reference_image_paths]
    : [];

  renderBindingsCard();
  renderReferenceVault();
  renderSpatialLayoutEditor();
  updateViewportAndTelemetry(item);
}

export function renderGallery() {
  const container = el("galleryList");
  el("galleryCountBadge").textContent = `${state.gallery.length} items`;
  if (!state.gallery.length) {
    container.innerHTML = `<div class="muted-note">Completed syntheses appear here with full recipe &amp; draft restoration.</div>`;
    return;
  }
  container.innerHTML = "";

  for (const item of state.gallery) {
    const card = document.createElement("div");
    card.className = "gallery-card";
    const t = item.telemetry || {};
    const solverTag = `${(t.early_solver || "heun").toUpperCase()}->${(t.late_solver || "ipndm").toUpperCase()}`;
    const batchCount = 1 + ((item.additional_image_urls && item.additional_image_urls.length) || 0);
    const batchBadge = batchCount > 1 ? ` (${batchCount}x)` : "";
    card.innerHTML = `
      <div class="gallery-thumb-wrap" data-view-id="${item.image_id}">
        <img src="${item.image_url}" alt="${item.title || "Master"}" loading="lazy" />
      </div>
      <div class="gallery-meta">
        <div class="gallery-title" title="${item.title || "Untitled Vision Master"}">${item.title || "Untitled Vision Master"}${batchBadge}</div>
        <div class="gallery-sub">${(item.model_variant || "sft").toUpperCase()} // ${item.width}x${item.height} // ${solverTag} // Seed ${item.seed}</div>
        <div class="gallery-actions">
          <button type="button" class="btn btn-secondary compact-btn" data-restore-id="${item.image_id}">Restore</button>
          <button type="button" class="btn btn-secondary compact-btn" data-promo-init="${item.image_id}" title="Promote image to Vault and bind as Init Image">-&gt; Init</button>
          <button type="button" class="btn btn-secondary compact-btn" data-promo-ref="${item.image_id}" title="Promote image to Vault and bind as 4D Reference">-&gt; 4D Ref</button>
          <button type="button" class="btn btn-secondary compact-btn" data-promo-box="${item.image_id}" title="Promote image to Vault and bind strictly to the active Spatial Layout Box">-&gt; Box</button>
          <button type="button" class="btn btn-secondary compact-btn" data-del-img="${item.image_id}">Delete</button>
        </div>
      </div>
    `;
    container.appendChild(card);
  }

  container.querySelectorAll("[data-view-id]").forEach((thumb) => {
    thumb.addEventListener("click", () => {
      const id = thumb.getAttribute("data-view-id");
      const found = state.gallery.find((g) => g.image_id === id);
      if (found) updateViewportAndTelemetry(found);
    });
  });

  container.querySelectorAll("[data-restore-id]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-restore-id");
      const found = state.gallery.find((g) => g.image_id === id);
      if (found) {
        restoreDraftToStudio(found);
        showStudioToast(`Restored recipe: ${found.title || found.image_id}`, "info");
      }
    });
  });

  container.querySelectorAll("[data-promo-init]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-promo-init");
      await promoteGalleryImageToVault(id, "init", 1);
    });
  });

  container.querySelectorAll("[data-promo-ref]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-promo-ref");
      await promoteGalleryImageToVault(id, "ref", 1);
    });
  });

  container.querySelectorAll("[data-promo-box]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-promo-box");
      await promoteGalleryImageToVault(id, "box", 1);
    });
  });

  container.querySelectorAll("[data-del-img]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-del-img");
      await apiFetch(`api/v1/images/${encodeURIComponent(id)}`, { method: "DELETE" });
      state.gallery = state.gallery.filter((g) => g.image_id !== id);
      renderGallery();
    });
  });
}