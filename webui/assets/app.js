// webui/assets/app.js
(() => {
  "use strict";

  const DEFAULT_PROMPT_TEXT =
    'A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter with a ceramic bowl of fresh citrus fruit. A small yellow sticky note on the refrigerator door is handwritten in crisp black marker: "Remember to test SFT & Base permutations @ 1024x1024"';

  const DEFAULT_LAYOUT_ITEMS = [
    {
      category: "refrigerator",
      caption: "A sleek brushed-steel refrigerator door with a small yellow sticky note",
      bbox: [32, 48, 992, 592],
      enabled: true,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    },
    {
      category: "text",
      text: "Remember to test SFT & Base permutations @ 1024x1024",
      bbox: [288, 144, 544, 480],
      enabled: true,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    },
    {
      category: "bowl",
      caption: "A ceramic bowl of fresh citrus fruit on a warm walnut kitchen counter",
      bbox: [592, 608, 976, 976],
      enabled: true,
      inpaint_box: false,
      paste_to_canvas: false,
      image_path: "",
      input_role: "",
      framing_mode: "",
      isolate_box_attn: null,
      denoise_strength: null
    }
  ];

  const state = {
    token: localStorage.getItem("vs_session_token") || "",
    user: null,
    themes: {},
    engineDefaults: {},
    layoutItems: JSON.parse(JSON.stringify(DEFAULT_LAYOUT_ITEMS)),
    references: [],
    projects: [],
    gallery: [],
    bindings: {
      init_image_path: null,
      inpaint_mask_path: null,
      control_image_path: null,
      reference_image_paths: []
    },
    activeProjectId: null,
    activeProjectTitle: null,
    activeImageId: null,
    activeVariantIndex: 1,
    lastTelemetry: null,
    activeJobId: null,
    activeEventSource: null,
    brushMode: "paint",
    isPainting: false
  };

  const el = (id) => document.getElementById(id);

  function parseIntList(raw) {
    if (!raw || typeof raw !== "string") return [];
    const s = raw.trim().replace(/^\[|\]$/g, "");
    if (!s || ["none", "off", "null"].includes(s.toLowerCase())) return [];
    const set = new Set();
    for (const part of s.split(/[,;]+/)) {
      const tok = part.trim();
      if (!tok) continue;
      if (tok.includes("-")) {
        const [a, b] = tok.split("-", 2).map((n) => parseInt(n.trim(), 10));
        if (!Number.isNaN(a) && !Number.isNaN(b)) {
          for (let i = Math.min(a, b); i <= Math.max(a, b); i++) {
            if (i >= 0 && i < 42) set.add(i);
          }
        }
      } else {
        const v = parseInt(tok, 10);
        if (!Number.isNaN(v) && v >= 0 && v < 42) set.add(v);
      }
    }
    return Array.from(set).sort((a, b) => a - b);
  }

  function parseOptionalFloat(raw, minVal, maxVal) {
    if (raw === null || raw === undefined) return null;
    const s = String(raw).trim();
    if (!s) return null;
    const val = parseFloat(s);
    if (Number.isNaN(val)) return null;
    return Math.max(minVal, Math.min(maxVal, val));
  }

  function parseTriStateBool(raw) {
    if (raw === "true" || raw === true) return true;
    if (raw === "false" || raw === false) return false;
    return null;
  }

  function gcd(a, b) {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y) {
      const t = y;
      y = x % y;
      x = t;
    }
    return x || 1;
  }

  function updateProjectBadge() {
    const badge = el("activeProjectBadge");
    if (!badge) return;
    if (state.activeProjectId) {
      const label = state.activeProjectTitle || state.activeProjectId;
      badge.textContent = `Project: ${label}`;
    } else {
      badge.textContent = "Project: Unsaved Draft";
    }
  }

  function computeStageGeometry() {
    const canvasW = Math.max(256, Math.min(2048, parseInt(el("widthInput").value, 10) || 1024));
    const canvasH = Math.max(256, Math.min(2048, parseInt(el("heightInput").value, 10) || 1024));
    const maxStagePx = 512;
    let stageW = maxStagePx;
    let stageH = maxStagePx;
    if (canvasW >= canvasH) {
      stageW = maxStagePx;
      stageH = Math.max(128, Math.round((maxStagePx * canvasH) / canvasW));
    } else {
      stageH = maxStagePx;
      stageW = Math.max(128, Math.round((maxStagePx * canvasW) / canvasH));
    }
    const div = gcd(canvasW / 16, canvasH / 16);
    const arW = Math.round((canvasW / 16) / div);
    const arH = Math.round((canvasH / 16) / div);
    const badge = el("stageAspectBadge");
    if (badge) {
      badge.textContent = `${canvasW}x${canvasH} (${arW}:${arH})`;
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

  function findRefMeta(refId) {
    if (!refId) return null;
    return state.references.find((r) => r.ref_id === refId) || null;
  }

  function resolveUnderlayUrl() {
    const mode = el("layoutUnderlaySelect") ? el("layoutUnderlaySelect").value : "auto";
    if (mode === "none") return null;
    const candidates = [
      state.bindings.init_image_path,
      state.bindings.control_image_path,
      ...(state.bindings.reference_image_paths || [])
    ];
    for (const refId of candidates) {
      const meta = findRefMeta(refId);
      if (meta && !meta.is_vizj && meta.url) {
        return meta.url;
      }
    }
    return null;
  }

  function syncUnderlayLayers() {
    const url = resolveUnderlayUrl();
    const layoutImg = el("spatialUnderlayImg");
    const inpaintImg = el("inpaintUnderlayImg");
    const layoutOp = parseFloat(el("layoutUnderlayOpacity").value) || 0.45;
    const inpaintOp = parseFloat(el("inpaintUnderlayOpacity").value) || 0.55;

    if (layoutImg) {
      if (url) {
        layoutImg.src = url;
        layoutImg.style.opacity = String(layoutOp);
        layoutImg.hidden = false;
      } else {
        layoutImg.hidden = true;
      }
    }
    if (inpaintImg) {
      if (url && inpaintOp > 0.01) {
        inpaintImg.src = url;
        inpaintImg.style.opacity = String(inpaintOp);
        inpaintImg.hidden = false;
      } else {
        inpaintImg.hidden = true;
      }
    }
  }

  function framingModeToCssObjectFit(framingMode) {
    const fm = String(framingMode || "inscribed").toLowerCase();
    if (fm === "contain" || fm === "never_crop") return "contain";
    if (fm === "stretch") return "fill";
    return "cover";
  }

  function reconcileSummaryQuotesJs(caption, activeTexts) {
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

  function syncTextEditToPrompt(oldText, newText) {
    if (!el("syncQuotesCheck") || !el("syncQuotesCheck").checked) return;
    if (!newText || !newText.trim()) return;
    const targetOld = (oldText || "").trim();
    for (const inputId of ["promptInput", "layoutSummaryInput"]) {
      const field = el(inputId);
      if (!field || !field.value) continue;
      if (targetOld && field.value.includes(`"${targetOld}"`)) {
        field.value = field.value.replace(`"${targetOld}"`, `"${newText}"`);
      } else {
        const activeTexts = state.layoutItems
          .filter((b) => b.enabled !== false && (b.category === "text" || b.text !== undefined) && String(b.text || "").trim())
          .map((b) => String(b.text).trim());
        field.value = reconcileSummaryQuotesJs(field.value, activeTexts);
      }
    }
  }

  async function apiFetch(endpoint, options = {}) {
    const cleanEndpoint = endpoint.replace(/^\/+/, "");
    const headers = Object.assign({}, options.headers || {});
    if (state.token) {
      headers["Authorization"] = `Bearer ${state.token}`;
    }
    if (options.body && typeof options.body === "object" && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.body);
    }
    const resp = await fetch(cleanEndpoint, Object.assign({}, options, { headers }));
    if (resp.status === 401) {
      showLoginModal();
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || "Session authentication required.");
    }
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `Request failed (${resp.status})`);
    }
    if (resp.status === 204 || resp.status === 304) return null;
    return resp.json();
  }

  function applyThemeVariables(themeId) {
    const themeObj = state.themes[themeId];
    if (!themeObj || !themeObj.variables) return;
    const root = document.documentElement;
    for (const [k, v] of Object.entries(themeObj.variables)) {
      root.style.setProperty(k, v);
    }
    const sel = el("themeSelect");
    if (sel && sel.value !== themeId) {
      sel.value = themeId;
    }
    localStorage.setItem("vs_active_theme", themeId);
  }

  async function loadThemeRegistry() {
    try {
      const data = await apiFetch("api/v1/themes/registry");
      if (data && data.themes) {
        state.themes = data.themes;
        const saved = localStorage.getItem("vs_active_theme") || "obsidian_gold";
        if (state.themes[saved]) {
          applyThemeVariables(saved);
        }
      }
    } catch (_) {}
  }

  function applyEngineDefaultsToDeck(defaults) {
    if (!defaults || typeof defaults !== "object") return;
    state.engineDefaults = defaults;
    if (defaults.model_variant) el("modelVariantSelect").value = defaults.model_variant;
    if (defaults.prompt_mode) el("promptModeSelect").value = defaults.prompt_mode;
    if (defaults.system_prompt !== undefined) el("systemPromptInput").value = defaults.system_prompt || "";
    if (defaults.thinking_trace !== undefined) el("thinkingTraceInput").value = defaults.thinking_trace || "";
    if (defaults.add_vision_id !== undefined) el("addVisionIdCheck").checked = Boolean(defaults.add_vision_id);
    if (defaults.max_sequence_length !== undefined) el("maxSeqLenInput").value = defaults.max_sequence_length;
    if (defaults.max_vit_patches !== undefined) el("maxVitPatchesInput").value = defaults.max_vit_patches;
    if (defaults.width) el("widthInput").value = defaults.width;
    if (defaults.height) el("heightInput").value = defaults.height;
    if (defaults.num_inference_steps) el("stepsInput").value = defaults.num_inference_steps;
    if (defaults.num_images_per_prompt) el("numImagesInput").value = defaults.num_images_per_prompt;
    if (defaults.early_solver) el("earlySolverSelect").value = defaults.early_solver;
    if (defaults.late_solver) el("lateSolverSelect").value = defaults.late_solver;
    if (defaults.handoff_threshold !== undefined) el("handoffInput").value = defaults.handoff_threshold;
    if (defaults.early_cfg !== undefined) el("earlyCfgInput").value = defaults.early_cfg;
    if (defaults.late_cfg !== undefined) el("lateCfgInput").value = defaults.late_cfg;
    if (defaults.shift !== undefined) el("shiftInput").value = defaults.shift;
    if (Array.isArray(defaults.uncond_skip_layers)) {
      el("skipLayersInput").value = defaults.uncond_skip_layers.join(", ");
    }
    if (Array.isArray(defaults.slg_layers)) {
      el("slgLayersInput").value = defaults.slg_layers.join(", ");
    }
    if (defaults.slg_scale !== undefined) el("slgScaleInput").value = defaults.slg_scale;
    if (defaults.slg_start !== undefined) el("slgStartInput").value = defaults.slg_start;
    if (defaults.slg_stop !== undefined) el("slgStopInput").value = defaults.slg_stop;
    if (defaults.eta !== undefined) el("etaInput").value = defaults.eta;
    if (defaults.s_noise !== undefined) el("sNoiseInput").value = defaults.s_noise;
    if (defaults.enable_thinking !== undefined) el("enableThinkingCheck").checked = Boolean(defaults.enable_thinking);
    if (defaults.use_v2_grid !== undefined) el("useV2GridCheck").checked = Boolean(defaults.use_v2_grid);
    if (defaults.preserve_canvas_outside_boxes !== undefined) {
      el("preserveOutsideBoxesCheck").checked = Boolean(defaults.preserve_canvas_outside_boxes);
    }
    if (defaults.composite_box_images_to_init !== undefined) {
      el("compositeBoxImagesCheck").checked = Boolean(defaults.composite_box_images_to_init);
    }
    if (defaults.box_attn_isolation !== undefined && defaults.box_attn_isolation !== null) {
      el("boxAttnIsolationInput").value = defaults.box_attn_isolation;
    }
    if (defaults.uncond_uses_references === true) {
      el("uncondUsesRefsSelect").value = "true";
    } else if (defaults.uncond_uses_references === false) {
      el("uncondUsesRefsSelect").value = "false";
    } else {
      el("uncondUsesRefsSelect").value = "";
    }
    if (defaults.mask_blend_mode) el("maskBlendModeSelect").value = defaults.mask_blend_mode;
    if (defaults.invert_inpaint_mask !== undefined) {
      el("invertMaskCheck").checked = Boolean(defaults.invert_inpaint_mask);
    }
    if (defaults.edit_format) el("editFormatSelect").value = defaults.edit_format;
    if (defaults.framing_mode) el("framingModeSelect").value = defaults.framing_mode;
    if (defaults.input_role) el("inputRoleSelect").value = defaults.input_role;
    if (defaults.denoise_strength !== undefined) el("denoiseInput").value = defaults.denoise_strength;
    if (defaults.mtf_boost !== undefined) el("mtfBoostInput").value = defaults.mtf_boost;
    if (defaults.soft_knee_tau !== undefined) el("softKneeTauInput").value = defaults.soft_knee_tau;
    if (defaults.mask_feather_radius !== undefined) el("featherInput").value = defaults.mask_feather_radius;
    if (defaults.latent_clamp_sigma !== undefined) el("clampSigmaInput").value = defaults.latent_clamp_sigma;
    if (defaults.deband_8bit !== undefined) el("deband8bitCheck").checked = Boolean(defaults.deband_8bit);
    if (defaults.apply_bayer_dither !== undefined) el("applyBayerDitherCheck").checked = Boolean(defaults.apply_bayer_dither);
    if (defaults.use_reference_cache !== undefined) el("useRefCacheCheck").checked = Boolean(defaults.use_reference_cache);
    if (defaults.prefer_vulkan !== undefined) el("preferVulkanCheck").checked = Boolean(defaults.prefer_vulkan);
    if (defaults.cpu_offload !== undefined) el("cpuOffloadCheck").checked = Boolean(defaults.cpu_offload);
    if (defaults.edit_lora_path) {
      el("editLoraSelect").value = defaults.edit_lora_path;
    } else {
      el("editLoraSelect").value = "";
    }
    if (defaults.edit_lora_scale !== undefined) el("loraScaleInput").value = defaults.edit_lora_scale;
    renderBindingsCard();
  }

  async function pollHealthTelemetry(applyDefaults = false) {
    try {
      const data = await apiFetch("api/v1/health");
      if (!data) return;
      const hw = data.hardware || {};
      const adapter = hw.adapter_name || data.engine_device.toUpperCase();
      const vizBackend = hw.vizinjector_backend || "Vulkan";
      el("hwStatusLine").textContent = `${adapter} // ${vizBackend}`;

      if (hw.dedicated_total_gb > 0) {
        const contig = hw.contiguous_avail_gb !== undefined ? ` | ${hw.contiguous_avail_gb.toFixed(1)}G contig` : "";
        const spillStr = hw.shared_delta_gb > 0.05 ? ` (+${hw.shared_delta_gb.toFixed(2)}G spill)` : "";
        el("vramPillText").textContent = `${hw.dedicated_used_gb.toFixed(1)}/${hw.dedicated_total_gb.toFixed(1)}G${contig}${spillStr}`;
      } else {
        el("vramPillText").textContent = `${data.engine_device.toUpperCase()} READY`;
      }

      const bp = data.baseline_parameters || {};
      const eSolv = (bp.early_solver || "heun").toUpperCase();
      const lSolv = (bp.late_solver || "ipndm").toUpperCase();
      const presetTag = data.engine_preset_active ? "DEFAULT.JSON" : "V03 BASELINE";
      el("presetPillText").textContent = `${presetTag} (${eSolv}->${lSolv})`;

      const qDepth = data.queue_depth || 0;
      el("queuePillText").textContent = data.active_job ? `ACTIVE (${qDepth} queued)` : `IDLE (${qDepth})`;

      if (applyDefaults && bp) {
        applyEngineDefaultsToDeck(bp);
      }
    } catch (err) {
      el("hwStatusLine").textContent = `Daemon Offline: ${err.message}`;
    }
  }

  function showLoginModal() {
    el("loginModal").hidden = false;
    el("loginErrorMsg").hidden = true;
    el("loginUsernameInput").focus();
  }

  function hideLoginModal() {
    el("loginModal").hidden = true;
  }

  async function authenticateUser(username) {
    const cleanUser = (username || "").trim();
    if (!cleanUser) return;
    el("loginErrorMsg").hidden = true;
    try {
      const data = await apiFetch("api/v1/auth/login", {
        method: "POST",
        body: { username: cleanUser }
      });
      state.token = data.token;
      state.user = data.user;
      localStorage.setItem("vs_session_token", data.token);
      localStorage.setItem("vs_last_username", cleanUser);

      el("authBtn").textContent = `${data.user.display_name} (${data.user.slug})`;
      el("quotaBadge").textContent = data.user.is_admin
        ? "Quota: UNLIMITED"
        : `Quota: ${data.user.tokens_remaining}/${data.user.daily_quota}`;
      el("saveDefaultPresetBtn").hidden = !data.user.is_admin;

      if (data.theme && data.theme.theme_id) {
        if (data.theme.manifest) {
          state.themes[data.theme.theme_id] = data.theme.manifest;
        }
        applyThemeVariables(data.theme.theme_id);
      }
      if (data.engine_defaults && Object.keys(state.engineDefaults).length === 0) {
        applyEngineDefaultsToDeck(data.engine_defaults);
      }
      if (Array.isArray(data.default_layout_items) && data.default_layout_items.length > 0 && state.layoutItems.length === 0) {
        state.layoutItems = JSON.parse(JSON.stringify(data.default_layout_items));
      }
      state.gallery = Array.isArray(data.images) ? data.images : [];
      renderGallery();
      await Promise.all([refreshReferences(), refreshProjects()]);
      hideLoginModal();
    } catch (err) {
      const msgBox = el("loginErrorMsg");
      msgBox.textContent = err.message;
      msgBox.hidden = false;
    }
  }

  function computeActiveMaskSummary() {
    const hasInit = Boolean(state.bindings.init_image_path);
    const initMeta = findRefMeta(state.bindings.init_image_path);
    const hasInitAlpha = Boolean(initMeta && initMeta.has_alpha);
    const hasPaintMask = Boolean(state.bindings.inpaint_mask_path);
    const globalSplat = Boolean(el("compositeBoxImagesCheck") && el("compositeBoxImagesCheck").checked);
    const preserveOutside = Boolean(el("preserveOutsideBoxesCheck") && el("preserveOutsideBoxesCheck").checked);
    const invert = Boolean(el("invertMaskCheck") && el("invertMaskCheck").checked);
    const blendMode = el("maskBlendModeSelect") ? el("maskBlendModeSelect").value : "auto";

    const enabledBoxes = state.layoutItems.filter((b) => b.enabled !== false);
    const hasSplatBoxes = enabledBoxes.some(
      (b) => Boolean(b.image_path && String(b.image_path).trim()) && (Boolean(b.paste_to_canvas) || globalSplat)
    );
    const hasFlaggedBoxes = enabledBoxes.some(
      (b) => Boolean(b.inpaint_box) || (hasInit && Boolean(b.paste_to_canvas))
    );
    const hasBoxAperture = (hasInit || hasSplatBoxes) && (preserveOutside || blendMode === "boxes_only" || hasFlaggedBoxes);

    if (!hasInit && !hasSplatBoxes) {
      return hasPaintMask ? "Mask Staged (Bind Init Img or Splat Box to Activate)" : "Unmasked Full-Canvas Flow";
    }

    const activeSources = [];
    if (hasPaintMask && blendMode !== "boxes_only" && blendMode !== "alpha_only") {
      activeSources.push("PaintMask");
    }
    if (hasBoxAperture && blendMode !== "paint_only" && blendMode !== "alpha_only") {
      activeSources.push(preserveOutside ? "AllBoxApertures" : "FlaggedBoxes");
    }
    if (hasInitAlpha && (blendMode === "alpha_only" || blendMode === "union" || (!hasPaintMask && !hasBoxAperture))) {
      activeSources.push("InitAlphaMatte");
    }

    if (!activeSources.length) {
      return hasSplatBoxes && !hasInit ? "Box Splatting (Soft Feather Alpha)" : "Img2Img Full-Canvas Denoise";
    }
    const opStr = activeSources.length > 1 ? ` [${blendMode.toUpperCase()}]` : "";
    const invStr = invert ? " (INVERTED)" : "";
    return `${activeSources.join(" + ")}${opStr}${invStr}`;
  }

  function renderBindingsCard() {
    el("boundInitLabel").textContent = state.bindings.init_image_path || "<None>";
    el("boundMaskLabel").textContent = state.bindings.inpaint_mask_path || "<None>";
    el("boundCtrlLabel").textContent = state.bindings.control_image_path || "<None>";
    el("boundRefsLabel").textContent = state.bindings.reference_image_paths.length > 0
      ? state.bindings.reference_image_paths.join(", ")
      : "<None>";
    const maskStateEl = el("activeMaskStateLabel");
    if (maskStateEl) {
      maskStateEl.textContent = computeActiveMaskSummary();
    }
    syncUnderlayLayers();
  }

  async function refreshReferences() {
    if (!state.token) return;
    try {
      const data = await apiFetch("api/v1/references");
      state.references = (data && Array.isArray(data.references)) ? data.references : [];
      renderReferenceVault();
      renderBindingsCard();
      renderSpatialLayoutEditor();
    } catch (_) {}
  }

  async function refreshProjects() {
    if (!state.token) return;
    try {
      const data = await apiFetch("api/v1/projects");
      state.projects = (data && Array.isArray(data.projects)) ? data.projects : [];
      renderProjectVault();
    } catch (_) {}
  }

  function renderReferenceVault() {
    const container = el("referenceList");
    const badge = el("assetCountBadge");
    if (badge) badge.textContent = String(state.references.length);

    if (!state.references.length) {
      container.innerHTML = `<div class="muted-note">No reference assets staged in your user vault.</div>`;
      return;
    }
    container.innerHTML = "";
    for (const ref of state.references) {
      const card = document.createElement("div");
      card.className = "ref-card";
      const sizeKb = (ref.size_bytes / 1024).toFixed(1);
      const isGlobalRef = state.bindings.reference_image_paths.includes(ref.ref_id);
      const dimsLabel = ref.width && ref.height ? `${ref.width}x${ref.height}` : "1024x1024";
      const alphaBadge = ref.has_alpha ? `<span class="meta-tag alpha-tag" title="Contains translucent alpha channel">ALPHA</span>` : "";
      const vizjBadge = ref.is_vizj ? `<span class="meta-tag vizj-tag" title="Tier-2 Zero-Copy .VIZJ Container">VIZJ</span>` : "";

      card.innerHTML = `
        <div class="ref-card-top">
          <span title="${ref.ref_id}">${ref.ref_id}</span>
          <button class="btn-link" data-del-ref="${ref.ref_id}">Delete</button>
        </div>
        <div class="ref-meta-badges">
          <span class="meta-tag">${dimsLabel}</span>
          <span class="meta-tag">${sizeKb} KB</span>
          ${alphaBadge}
          ${vizjBadge}
        </div>
        <div class="ref-bind-btns">
          <button class="btn btn-secondary compact-btn ${isGlobalRef ? "active" : ""}" data-bind-role="ref" data-ref-id="${ref.ref_id}">4D Ref</button>
          <button class="btn btn-secondary compact-btn ${state.bindings.init_image_path === ref.ref_id ? "active" : ""}" data-bind-role="init" data-ref-id="${ref.ref_id}">Init Img</button>
          <button class="btn btn-secondary compact-btn ${state.bindings.inpaint_mask_path === ref.ref_id ? "active" : ""}" data-bind-role="mask" data-ref-id="${ref.ref_id}">Mask</button>
          <button class="btn btn-secondary compact-btn ${state.bindings.control_image_path === ref.ref_id ? "active" : ""}" data-bind-role="ctrl" data-ref-id="${ref.ref_id}">Control</button>
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
            if (el("editFormatSelect").value === "standard") {
              el("editFormatSelect").value = "ai_toolkit_isolated";
            }
          }
        } else if (role === "init") {
          state.bindings.init_image_path = state.bindings.init_image_path === refId ? null : refId;
          if (state.bindings.init_image_path && parseFloat(el("denoiseInput").value) >= 1.0) {
            el("denoiseInput").value = "0.75";
          }
        } else if (role === "mask") {
          state.bindings.inpaint_mask_path = state.bindings.inpaint_mask_path === refId ? null : refId;
        } else if (role === "ctrl") {
          state.bindings.control_image_path = state.bindings.control_image_path === refId ? null : refId;
          if (state.bindings.control_image_path) {
            el("editFormatSelect").value = "control_color_depth";
            el("inputRoleSelect").value = "color_map";
          }
        }
        renderBindingsCard();
        renderReferenceVault();
        renderSpatialLayoutEditor();
      });
    });
  }

  function renderProjectVault() {
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
          <button class="btn-link" data-del-proj="${proj.project_id}">Delete</button>
        </div>
        <div class="ref-meta-badges">
          <span class="meta-tag">${(proj.model_variant || "sft").toUpperCase()}</span>
          <span class="meta-tag">${dims}</span>
          <span class="meta-tag">${proj.asset_count || 0} assets</span>
          <span class="meta-tag">${sizeKb} KB</span>
        </div>
        <div class="project-actions">
          <button class="btn btn-secondary compact-btn" data-load-proj="${proj.project_id}">Load Project</button>
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

  function hasInpaintCanvasContent() {
    const canvas = el("inpaintCanvas");
    if (!canvas) return false;
    const ctx = canvas.getContext("2d");
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < data.length; i += 16) {
      if (data[i] > 4) return true;
    }
    return false;
  }

  function loadCanvasFromDataUrl(dataUrl) {
    const canvas = el("inpaintCanvas");
    if (!canvas || !dataUrl) return;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = dataUrl;
  }

  function resetToNewProjectDraft() {
    state.activeProjectId = null;
    state.activeProjectTitle = null;
    state.activeImageId = null;
    state.lastTelemetry = null;
    el("titleInput").value = "Untitled Vision Master";
    el("promptInput").value = DEFAULT_PROMPT_TEXT;
    el("negativePromptInput").value = "";
    el("layoutSummaryInput").value = "";
    state.layoutItems = JSON.parse(JSON.stringify(DEFAULT_LAYOUT_ITEMS));
    state.bindings.init_image_path = null;
    state.bindings.inpaint_mask_path = null;
    state.bindings.control_image_path = null;
    state.bindings.reference_image_paths = [];
    const canvas = el("inpaintCanvas");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    updateProjectBadge();
    renderBindingsCard();
    renderReferenceVault();
    renderProjectVault();
    renderSpatialLayoutEditor();
  }

  async function saveCurrentProject(downloadAfterSave = false) {
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
      alert(`Project save failed: ${err.message}`);
    } finally {
      saveBtn.disabled = false;
      expBtn.disabled = false;
    }
  }

  async function loadProjectFromVault(projectId) {
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
    } catch (err) {
      alert(`Failed loading project: ${err.message}`);
    }
  }

  async function importProjectFromFile(file) {
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
      } catch (err) {
        alert(`Failed importing .vsproj package: ${err.message}`);
      }
    };
    reader.readAsDataURL(file);
  }

  async function promoteGalleryImageToVault(imageId, bindMode = "ref", variantIndex = 1) {
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
        if (parseFloat(el("denoiseInput").value) >= 1.0) {
          el("denoiseInput").value = "0.75";
        }
      } else {
        if (!state.bindings.reference_image_paths.includes(meta.ref_id)) {
          state.bindings.reference_image_paths.push(meta.ref_id);
        }
        if (el("editFormatSelect").value === "standard") {
          el("editFormatSelect").value = "ai_toolkit_isolated";
        }
      }
      renderBindingsCard();
      renderReferenceVault();
      renderSpatialLayoutEditor();
    } catch (err) {
      alert(`Failed promoting image to vault: ${err.message}`);
    }
  }

  function matchPrimaryAssetAspect() {
    let targetRefId =
      state.bindings.init_image_path ||
      state.bindings.control_image_path ||
      (state.bindings.reference_image_paths && state.bindings.reference_image_paths[0]) ||
      null;
    if (!targetRefId) {
      const boxWithImg = state.layoutItems.find((b) => b.enabled !== false && b.image_path);
      if (boxWithImg) targetRefId = boxWithImg.image_path;
    }
    if (!targetRefId && state.references.length > 0) {
      targetRefId = state.references[0].ref_id;
    }
    const meta = findRefMeta(targetRefId);
    if (!meta || !meta.width || !meta.height) return;

    const oldW = parseInt(el("widthInput").value, 10) || 1024;
    const oldH = parseInt(el("heightInput").value, 10) || 1024;
    const targetArea = 1024 * 1024;
    const scale = Math.sqrt(targetArea / Math.max(1, meta.width * meta.height));
    const newW = Math.max(256, Math.min(2048, Math.round((meta.width * scale) / 16) * 16));
    const newH = Math.max(256, Math.min(2048, Math.round((meta.height * scale) / 16) * 16));
    applyCanvasResolutionChange(oldW, oldH, newW, newH);
  }

  function applyCanvasResolutionChange(oldW, oldH, newW, newH) {
    const cleanW = Math.max(256, Math.min(2048, Math.round(newW / 16) * 16));
    const cleanH = Math.max(256, Math.min(2048, Math.round(newH / 16) * 16));
    el("widthInput").value = cleanW;
    el("heightInput").value = cleanH;

    if (oldW > 0 && oldH > 0 && (oldW !== cleanW || oldH !== cleanH)) {
      const sx = cleanW / oldW;
      const sy = cleanH / oldH;
      for (const b of state.layoutItems) {
        if (Array.isArray(b.bbox) && b.bbox.length === 4) {
          const ny0 = Math.max(0, Math.min(cleanH - 16, Math.round(b.bbox[0] * sy)));
          const nx0 = Math.max(0, Math.min(cleanW - 16, Math.round(b.bbox[1] * sx)));
          const ny1 = Math.max(ny0 + 16, Math.min(cleanH, Math.round(b.bbox[2] * sy)));
          const nx1 = Math.max(nx0 + 16, Math.min(cleanW, Math.round(b.bbox[3] * sx)));
          b.bbox = [ny0, nx0, ny1, nx1];
        }
      }
    }
    syncStageAspectAndGuides();
    renderSpatialLayoutEditor();
  }

  function syncStageAspectAndGuides() {
    const geom = computeStageGeometry();
    const spatialStage = el("spatialStage");
    if (spatialStage) {
      spatialStage.style.width = `${geom.stageW}px`;
      spatialStage.style.height = `${geom.stageH}px`;
    }

    const inpaintContainer = el("inpaintStageContainer");
    const inpaintCanvas = el("inpaintCanvas");
    if (inpaintContainer && inpaintCanvas) {
      inpaintContainer.style.width = `${geom.stageW}px`;
      inpaintContainer.style.height = `${geom.stageH}px`;
      if (inpaintCanvas.width !== geom.stageW || inpaintCanvas.height !== geom.stageH) {
        const tmp = document.createElement("canvas");
        tmp.width = inpaintCanvas.width;
        tmp.height = inpaintCanvas.height;
        tmp.getContext("2d").drawImage(inpaintCanvas, 0, 0);
        inpaintCanvas.width = geom.stageW;
        inpaintCanvas.height = geom.stageH;
        const ctx = inpaintCanvas.getContext("2d");
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, geom.stageW, geom.stageH);
        ctx.drawImage(tmp, 0, 0, geom.stageW, geom.stageH);
      }
    }
    renderInpaintBoxGuides(geom);
  }

  function renderInpaintBoxGuides(geomInput) {
    const overlay = el("inpaintBoxesOverlay");
    if (!overlay) return;
    const geom = geomInput || computeStageGeometry();
    overlay.innerHTML = "";
    state.layoutItems.forEach((item, idx) => {
      if (item.enabled === false || !Array.isArray(item.bbox) || item.bbox.length !== 4) return;
      const [ymin, xmin, ymax, xmax] = item.bbox;
      const guide = document.createElement("div");
      guide.className = "inpaint-box-guide";
      guide.style.top = `${Math.round(ymin * geom.scaleY)}px`;
      guide.style.left = `${Math.round(xmin * geom.scaleX)}px`;
      guide.style.width = `${Math.max(16, Math.round((xmax - xmin) * geom.scaleX))}px`;
      guide.style.height = `${Math.max(16, Math.round((ymax - ymin) * geom.scaleY))}px`;
      guide.textContent = `#${idx + 1} ${item.category || "box"}`;
      overlay.appendChild(guide);
    });
  }

  async function uploadDataUrlToVault(filename, role, dataUrl) {
    const res = await apiFetch("api/v1/references/upload", {
      method: "POST",
      body: { filename, role, data_url: dataUrl }
    });
    await refreshReferences();
    return res;
  }

  function pixelToAbsCentralized(ymin, xmin, ymax, xmax, height, width) {
    const cy = (height - 1) / 2.0;
    const cx = (width - 1) / 2.0;
    return [
      Math.round(ymin - cy),
      Math.round(xmin - cx),
      Math.round(ymax - cy),
      Math.round(xmax - cx)
    ];
  }

  function computeBoxPictureBindings() {
    const globalRole = (el("inputRoleSelect").value || "photo_ref").toLowerCase();
    const editFormat = (el("editFormatSelect").value || "standard").toLowerCase();
    const globalEmitsVit = globalRole === "photo_ref" && editFormat !== "moss_image_edit";

    let vitPicCounter = globalEmitsVit ? state.bindings.reference_image_paths.length : 0;
    let ditRefCounter = state.bindings.reference_image_paths.length;
    const boxPicMap = {};
    const boxDitMap = {};

    state.layoutItems.forEach((b, idx) => {
      if (b.enabled === false) return;
      if (b.image_path && String(b.image_path).trim() !== "") {
        ditRefCounter += 1;
        boxDitMap[idx] = ditRefCounter;
        const effRole = (b.input_role || globalRole || "photo_ref").toLowerCase();
        if (effRole === "photo_ref") {
          vitPicCounter += 1;
          boxPicMap[idx] = vitPicCounter;
        }
      }
    });
    return { boxPicMap, boxDitMap };
  }

  function updateCompiledJsonPreview() {
    const width = parseInt(el("widthInput").value, 10) || 1024;
    const height = parseInt(el("heightInput").value, 10) || 1024;
    const summaryOverride = el("layoutSummaryInput") ? el("layoutSummaryInput").value.trim() : "";
    const rawCaption = summaryOverride || (el("promptInput").value || "").trim();
    const { boxPicMap, boxDitMap } = computeBoxPictureBindings();

    const items = [];
    const activeTexts = [];

    state.layoutItems.forEach((b, idx) => {
      if (b.enabled === false) return;
      const entry = { category: b.category || (b.text !== undefined ? "text" : "region") };
      if (b.text !== undefined && String(b.text).trim() !== "") {
        entry.text = b.text;
        if (entry.category === "text") {
          activeTexts.push(String(b.text).trim());
        }
      } else if (b.caption !== undefined && String(b.caption).trim() !== "") {
        entry.caption = b.caption;
      }
      if (boxPicMap[idx]) {
        entry.reference = `Picture ${boxPicMap[idx]}`;
      }
      if (Array.isArray(b.bbox) && b.bbox.length === 4) {
        entry.bbox = pixelToAbsCentralized(b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], height, width);
      }
      items.push(entry);
    });

    const reconciledCaption = reconcileSummaryQuotesJs(rawCaption, activeTexts);
    const payload = {
      bbox_format: "yxyx",
      bbox_coord_space: "abs_centralized",
      summary: { caption: reconciledCaption },
      items
    };
    el("compiledJsonPreview").textContent = JSON.stringify(payload, null, 2);

    const modeBadge = el("layoutActiveModeBadge");
    if (modeBadge) {
      const promptMode = el("promptModeSelect").value;
      const hasBoxRefs = Object.keys(boxDitMap).length > 0;
      if (promptMode === "layout_json" || hasBoxRefs) {
        modeBadge.textContent = hasBoxRefs ? "LAYOUT JSON + 4D-RoPE ACTIVE" : "LAYOUT JSON ACTIVE";
      } else {
        modeBadge.textContent = "NATURAL MODE (JSON INACTIVE)";
      }
    }
    renderBindingsCard();
  }

  function renderSpatialLayoutEditor() {
    const geom = computeStageGeometry();
    syncStageAspectAndGuides();
    syncUnderlayLayers();

    const boxesLayer = el("spatialBoxesLayer");
    const listContainer = el("boxListContainer");
    if (!boxesLayer || !listContainer) return;

    const { boxPicMap, boxDitMap } = computeBoxPictureBindings();
    const globalFraming = el("framingModeSelect").value || "inscribed";
    const globalIso = parseFloat(el("boxAttnIsolationInput").value) || 0.0;
    const globalSplat = Boolean(el("compositeBoxImagesCheck").checked);

    boxesLayer.innerHTML = "";
    listContainer.innerHTML = "";

    state.layoutItems.forEach((item, idx) => {
      const isEnabled = item.enabled !== false;
      const [ymin, xmin, ymax, xmax] = item.bbox;
      const isText = item.category === "text" || item.text !== undefined;
      const hasBoundRef = Boolean(item.image_path && String(item.image_path).trim());
      const refMeta = hasBoundRef ? findRefMeta(item.image_path) : null;
      const effFraming = item.framing_mode || (globalFraming === "never_crop" ? "inscribed" : globalFraming);
      const effIso = item.isolate_box_attn !== null && item.isolate_box_attn !== undefined ? item.isolate_box_attn : (hasBoundRef ? globalIso : 0.0);
      const effSplat = item.paste_to_canvas !== null && item.paste_to_canvas !== undefined ? Boolean(item.paste_to_canvas) : (hasBoundRef && globalSplat);

      if (isEnabled) {
        const boxEl = document.createElement("div");
        boxEl.className = `spatial-box ${isText ? "text-box" : ""} ${hasBoundRef ? "ref-bound-box" : ""}`;
        boxEl.style.top = `${Math.round(ymin * geom.scaleY)}px`;
        boxEl.style.left = `${Math.round(xmin * geom.scaleX)}px`;
        boxEl.style.width = `${Math.max(24, Math.round((xmax - xmin) * geom.scaleX))}px`;
        boxEl.style.height = `${Math.max(24, Math.round((ymax - ymin) * geom.scaleY))}px`;

        const thumbHtml = refMeta && !refMeta.is_vizj
          ? `<img class="spatial-box-thumb" src="${refMeta.url}" alt="" style="object-fit: ${framingModeToCssObjectFit(effFraming)};" />`
          : "";

        const badges = [];
        if (boxPicMap[idx]) {
          badges.push(`<span class="box-mini-pill">Pic ${boxPicMap[idx]}</span>`);
        } else if (boxDitMap[idx]) {
          badges.push(`<span class="box-mini-pill">4D#${boxDitMap[idx]} (${item.input_role || el("inputRoleSelect").value})</span>`);
        }
        if (hasBoundRef && effIso > 0) {
          badges.push(`<span class="box-mini-pill">Iso ${Number(effIso).toFixed(2)}</span>`);
        }
        if (hasBoundRef && effSplat) {
          badges.push(`<span class="box-mini-pill">Splat</span>`);
        }
        if (item.inpaint_box) {
          badges.push(`<span class="box-mini-pill">Inpaint</span>`);
        }

        const labelText = isText ? `T: "${item.text || ""}"` : `${item.category}: ${item.caption || ""}`;
        boxEl.innerHTML = `
          ${thumbHtml}
          <div class="spatial-box-tag">[#${idx + 1}] ${labelText}</div>
          <div class="spatial-box-badges">${badges.join("")}</div>
          <div class="spatial-box-resize" data-resize-idx="${idx}"></div>
        `;

        boxEl.addEventListener("mousedown", (ev) => {
          if (ev.target.hasAttribute("data-resize-idx")) return;
          ev.preventDefault();
          const startX = ev.clientX;
          const startY = ev.clientY;
          const [origY0, origX0, origY1, origX1] = item.bbox;
          const boxW = origX1 - origX0;
          const boxH = origY1 - origY0;

          function onMove(moveEv) {
            const dx = Math.round((moveEv.clientX - startX) / geom.scaleX);
            const dy = Math.round((moveEv.clientY - startY) / geom.scaleY);
            const nx0 = Math.max(0, Math.min(geom.canvasW - boxW, origX0 + dx));
            const ny0 = Math.max(0, Math.min(geom.canvasH - boxH, origY0 + dy));
            item.bbox = [ny0, nx0, ny0 + boxH, nx0 + boxW];
            boxEl.style.top = `${Math.round(ny0 * geom.scaleY)}px`;
            boxEl.style.left = `${Math.round(nx0 * geom.scaleX)}px`;
            const hdr = listContainer.querySelector(`[data-box-coord-hdr="${idx}"]`);
            if (hdr) {
              const cc = pixelToAbsCentralized(item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3], geom.canvasH, geom.canvasW);
              hdr.textContent = `#${idx + 1} [${cc.join(", ")}]`;
            }
            renderInpaintBoxGuides(geom);
            updateCompiledJsonPreview();
          }

          function onUp() {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
          }

          window.addEventListener("mousemove", onMove);
          window.addEventListener("mouseup", onUp);
        });

        const resizeHandle = boxEl.querySelector("[data-resize-idx]");
        resizeHandle.addEventListener("mousedown", (ev) => {
          ev.stopPropagation();
          ev.preventDefault();
          const startX = ev.clientX;
          const startY = ev.clientY;
          const [origY0, origX0, origY1, origX1] = item.bbox;

          function onMove(moveEv) {
            const dx = Math.round((moveEv.clientX - startX) / geom.scaleX);
            const dy = Math.round((moveEv.clientY - startY) / geom.scaleY);
            const nx1 = Math.max(origX0 + 32, Math.min(geom.canvasW, origX1 + dx));
            const ny1 = Math.max(origY0 + 32, Math.min(geom.canvasH, origY1 + dy));
            item.bbox = [origY0, origX0, ny1, nx1];
            boxEl.style.width = `${Math.max(24, Math.round((nx1 - origX0) * geom.scaleX))}px`;
            boxEl.style.height = `${Math.max(24, Math.round((ny1 - origY0) * geom.scaleY))}px`;
            const hdr = listContainer.querySelector(`[data-box-coord-hdr="${idx}"]`);
            if (hdr) {
              const cc = pixelToAbsCentralized(item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3], geom.canvasH, geom.canvasW);
              hdr.textContent = `#${idx + 1} [${cc.join(", ")}]`;
            }
            renderInpaintBoxGuides(geom);
            updateCompiledJsonPreview();
          }

          function onUp() {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
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
              `<option value="${r.ref_id}" ${item.image_path === r.ref_id ? "selected" : ""}>${r.ref_id} (${r.width || 1024}x${r.height || 1024})</option>`
          )
        )
        .join("");

      const curRole = item.input_role || "";
      const curFraming = item.framing_mode || "";
      const isoVal = item.isolate_box_attn !== null && item.isolate_box_attn !== undefined ? item.isolate_box_attn : "";
      const dnVal = item.denoise_strength !== null && item.denoise_strength !== undefined ? item.denoise_strength : "";

      const cCoords = pixelToAbsCentralized(ymin, xmin, ymax, xmax, geom.canvasH, geom.canvasW);
      const card = document.createElement("div");
      card.className = `box-card ${!isEnabled ? "disabled-card" : ""}`;
      card.innerHTML = `
        <div class="box-card-header">
          <label class="checkbox-inline">
            <input type="checkbox" data-box-enabled="${idx}" ${isEnabled ? "checked" : ""} />
            <strong data-box-coord-hdr="${idx}">#${idx + 1} [${cCoords.join(", ")}]</strong>
          </label>
          <button class="btn-link" data-remove-box="${idx}">Remove</button>
        </div>
        <div class="field-row two-col">
          <input type="text" class="input-text" data-box-cat="${idx}" value="${item.category || ""}" placeholder="Category" />
          <select class="input-select" data-box-img="${idx}" title="Bind Reference Asset to Box 4D-RoPE">${refOptions}</select>
        </div>
        <input type="text" class="input-text" data-box-desc="${idx}" value="${(isText ? item.text : item.caption) || ""}" placeholder="${isText ? "Exact typography text" : "Region description"}" />
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
          <input type="number" class="input-text" data-box-iso="${idx}" value="${isoVal}" min="0.0" max="1.0" step="0.05" placeholder="Attn Iso (Inherit)" title="Per-box 2D cosine log-aperture attention isolation [0.0 - 1.0]" />
          <input type="number" class="input-text" data-box-dn="${idx}" value="${dnVal}" min="0.05" max="1.0" step="0.05" placeholder="Denoise (Inherit)" title="Per-box latent splat denoise strength [0.05 - 1.0]" />
        </div>
        <div class="box-flags-row">
          <label class="checkbox-inline" title="Restrict canvas inpainting strictly to this box aperture">
            <input type="checkbox" data-box-inpaint="${idx}" ${item.inpaint_box ? "checked" : ""} />
            <span>Inpaint Box Only</span>
          </label>
          <label class="checkbox-inline" title="Splat this box's reference latents onto the initial canvas">
            <input type="checkbox" data-box-paste="${idx}" ${item.paste_to_canvas ? "checked" : ""} />
            <span>Splat to Canvas</span>
          </label>
        </div>
      `;
      listContainer.appendChild(card);
    });

    listContainer.querySelectorAll("[data-box-enabled]").forEach((chk) => {
      chk.addEventListener("change", () => {
        const i = parseInt(chk.getAttribute("data-box-enabled"), 10);
        state.layoutItems[i].enabled = chk.checked;
        renderSpatialLayoutEditor();
      });
    });

    listContainer.querySelectorAll("[data-remove-box]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const i = parseInt(btn.getAttribute("data-remove-box"), 10);
        state.layoutItems.splice(i, 1);
        renderSpatialLayoutEditor();
      });
    });

    listContainer.querySelectorAll("[data-box-cat]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const i = parseInt(inp.getAttribute("data-box-cat"), 10);
        state.layoutItems[i].category = inp.value;
        updateCompiledJsonPreview();
      });
    });

    listContainer.querySelectorAll("[data-box-desc]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const i = parseInt(inp.getAttribute("data-box-desc"), 10);
        const target = state.layoutItems[i];
        if (target.category === "text" || target.text !== undefined) {
          const prevText = target.text || "";
          target.text = inp.value;
          delete target.caption;
          syncTextEditToPrompt(prevText, inp.value);
        } else {
          target.caption = inp.value;
        }
        const stageTags = boxesLayer.querySelectorAll(".spatial-box-tag");
        if (stageTags[i]) {
          const isT = target.category === "text" || target.text !== undefined;
          stageTags[i].textContent = `[#${i + 1}] ${isT ? `T: "${target.text || ""}"` : `${target.category}: ${target.caption || ""}`}`;
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
        renderBindingsCard();
      });
    });

    listContainer.querySelectorAll("[data-box-dn]").forEach((inp) => {
      inp.addEventListener("input", () => {
        const i = parseInt(inp.getAttribute("data-box-dn"), 10);
        state.layoutItems[i].denoise_strength = parseOptionalFloat(inp.value, 0.05, 1.0);
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

  function projectBoxesToInpaintCanvas() {
    const canvas = el("inpaintCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const geom = computeStageGeometry();
    const w = canvas.width;
    const h = canvas.height;
    const featherLat = Math.max(0.25, parseFloat(el("featherInput").value) || 1.25);
    const featherPx = Math.max(1.0, featherLat * 16.0 * ((geom.scaleX + geom.scaleY) * 0.5));

    const enabledBoxes = state.layoutItems.filter(
      (b) => b.enabled !== false && Array.isArray(b.bbox) && b.bbox.length === 4
    );
    const flaggedBoxes = enabledBoxes.filter((b) => Boolean(b.inpaint_box) || Boolean(b.paste_to_canvas));
    const activeBoxes = flaggedBoxes.length > 0 ? flaggedBoxes : enabledBoxes;
    if (!activeBoxes.length) return;

    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    const scaledBoxes = activeBoxes.map((b) => [
      b.bbox[0] * geom.scaleY,
      b.bbox[1] * geom.scaleX,
      b.bbox[2] * geom.scaleY,
      b.bbox[3] * geom.scaleX
    ]);

    for (let py = 0; py < h; py++) {
      const yy = py + 0.5;
      for (let px = 0; px < w; px++) {
        const xx = px + 0.5;
        let maxVal = 0.0;
        for (let k = 0; k < scaledBoxes.length; k++) {
          const [ly0, lx0, ly1, lx1] = scaledBoxes[k];
          const dy = Math.max(0.0, ly0 - yy, yy - ly1);
          const dx = Math.max(0.0, lx0 - xx, xx - lx1);
          const dist = Math.sqrt(dy * dy + dx * dx);
          if (dist <= 0.0) {
            maxVal = 1.0;
            break;
          } else if (dist < featherPx) {
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

  function invertInpaintCanvasPixels() {
    const canvas = el("inpaintCanvas");
    if (!canvas) return;
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

  function initInpaintCanvas() {
    const canvas = el("inpaintCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function paintAt(ev) {
      if (!state.isPainting) return;
      const rect = canvas.getBoundingClientRect();
      const x = (ev.clientX - rect.left) * (canvas.width / rect.width);
      const y = (ev.clientY - rect.top) * (canvas.height / rect.height);
      const rad = parseInt(el("brushSizeInput").value, 10) || 40;
      ctx.fillStyle = state.brushMode === "paint" ? "#ffffff" : "#000000";
      ctx.beginPath();
      ctx.arc(x, y, rad / 2, 0, Math.PI * 2);
      ctx.fill();
    }

    canvas.addEventListener("mousedown", (ev) => {
      state.isPainting = true;
      paintAt(ev);
    });
    window.addEventListener("mousemove", paintAt);
    window.addEventListener("mouseup", () => {
      state.isPainting = false;
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

    el("projectBoxesToMaskBtn").addEventListener("click", projectBoxesToInpaintCanvas);
    el("invertMaskCanvasBtn").addEventListener("click", invertInpaintCanvasPixels);

    el("clearMaskCanvasBtn").addEventListener("click", () => {
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
      } catch (err) {
        alert(`Failed staging mask: ${err.message}`);
      }
    });
  }

  function solvePowInWorker(challenge, difficulty) {
    return new Promise((resolve) => {
      const workerCode = `
        function rightRotate(value, amount) {
          return (value >>> amount) | (value << (32 - amount));
        }
        function sha256Hex(ascii) {
          const mathPow = Math.pow;
          const maxWord = mathPow(2, 32);
          let result = "";
          const words = [];
          const asciiBitLength = ascii.length * 8;
          let hash = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
          ];
          const k = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
          ];
          ascii += "\\x80";
          while ((ascii.length % 64) - 56) ascii += "\\x00";
          for (let i = 0; i < ascii.length; i++) {
            const j = ascii.charCodeAt(i);
            words[i >> 2] |= j << ((3 - (i % 4)) * 8);
          }
          words[words.length] = (asciiBitLength / maxWord) | 0;
          words[words.length] = asciiBitLength;
          for (let j = 0; j < words.length; ) {
            const w = words.slice(j, (j += 16));
            const oldHash = hash.slice(0);
            for (let i = 0; i < 64; i++) {
              const w15 = w[i - 15], w2 = w[i - 2];
              const a = hash[0], e = hash[4];
              const temp1 =
                hash[7] +
                (rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25)) +
                ((e & hash[5]) ^ (~e & hash[6])) +
                k[i] +
                (w[i] =
                  i < 16
                    ? w[i]
                    : (w[i - 16] +
                        (rightRotate(w15, 7) ^ rightRotate(w15, 18) ^ (w15 >>> 3)) +
                        w[i - 7] +
                        (rightRotate(w2, 17) ^ rightRotate(w2, 19) ^ (w2 >>> 10))) |
                      0);
              const temp2 =
                (rightRotate(a, 2) ^ rightRotate(a, 13) ^ rightRotate(a, 22)) +
                ((a & hash[1]) ^ (a & hash[2]) ^ (hash[1] & hash[2]));
              hash = [(temp1 + temp2) | 0].concat(hash);
              hash[4] = (hash[4] + temp1) | 0;
              hash.pop();
            }
            for (let i = 0; i < 8; i++) hash[i] = (hash[i] + oldHash[i]) | 0;
          }
          for (let i = 0; i < 8; i++) {
            for (let j = 3; j + 1; j--) {
              const b = (hash[i] >> (j * 8)) & 255;
              result += (b < 16 ? 0 : "") + b.toString(16);
            }
          }
          return result;
        }
        self.onmessage = (e) => {
          const { challenge, difficulty } = e.data;
          const target = "0".repeat(difficulty);
          let n = 0;
          while (true) {
            const cand = n.toString(16);
            const h = sha256Hex(challenge + ":" + cand);
            if (h.startsWith(target)) {
              self.postMessage(cand);
              break;
            }
            n++;
          }
        };
      `;
      const blob = new Blob([workerCode], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      const worker = new Worker(url);
      worker.onmessage = (e) => {
        URL.revokeObjectURL(url);
        worker.terminate();
        resolve(e.data);
      };
      worker.postMessage({ challenge, difficulty });
    });
  }

  function updateViewportAndTelemetry(jobStatus) {
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
            });
            strip.appendChild(th);
          });
          strip.hidden = false;
        } else {
          strip.hidden = true;
        }
      }
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
      el("tRoutingVal").textContent = `${t.edit_format || "standard"}${isoTag}${splatTag}`;
      el("tSignalVal").textContent = `${Number(t.shannon_entropy_bits).toFixed(2)}b / ${Number(t.laplacian_sharpness).toFixed(3)}`;
      el("tLumaVal").textContent = `${Number(t.mean_luminance).toFixed(3)} / ${Number(t.rms_contrast).toFixed(3)}`;
      el("tCacheVal").textContent = `Hit=${Boolean(t.prompt_cache_hit)} (Pfx ${t.common_prefix_tokens || 0}t)`;
    }
  }

  function restoreDraftToStudio(item) {
    const draft = item.working_draft || item.recipe || {};
    if (draft.title || item.title) el("titleInput").value = draft.title || item.title;
    if (draft.model_variant) el("modelVariantSelect").value = draft.model_variant;
    if (draft.prompt_mode) el("promptModeSelect").value = draft.prompt_mode;
    if (draft.prompt !== undefined) el("promptInput").value = draft.prompt;
    if (draft.layout_summary !== undefined && el("layoutSummaryInput")) {
      const sumVal = draft.layout_summary || "";
      el("layoutSummaryInput").value = sumVal === draft.prompt ? "" : sumVal;
    }
    if (draft.negative_prompt !== undefined) el("negativePromptInput").value = draft.negative_prompt;
    if (draft.system_prompt !== undefined) el("systemPromptInput").value = draft.system_prompt || "";
    if (draft.thinking_trace !== undefined) el("thinkingTraceInput").value = draft.thinking_trace || "";
    if (draft.add_vision_id !== undefined && draft.add_vision_id !== null) {
      el("addVisionIdCheck").checked = Boolean(draft.add_vision_id);
    }
    if (draft.enable_thinking !== undefined && draft.enable_thinking !== null) {
      el("enableThinkingCheck").checked = Boolean(draft.enable_thinking);
    }
    if (draft.width) el("widthInput").value = draft.width;
    if (draft.height) el("heightInput").value = draft.height;
    if (draft.num_inference_steps) el("stepsInput").value = draft.num_inference_steps;
    if (draft.num_images_per_prompt) el("numImagesInput").value = draft.num_images_per_prompt;
    if (draft.early_solver) el("earlySolverSelect").value = draft.early_solver;
    if (draft.late_solver) el("lateSolverSelect").value = draft.late_solver;
    if (draft.handoff_threshold !== undefined) el("handoffInput").value = draft.handoff_threshold;
    if (draft.early_cfg !== undefined) el("earlyCfgInput").value = draft.early_cfg;
    if (draft.late_cfg !== undefined) el("lateCfgInput").value = draft.late_cfg;
    if (draft.shift !== undefined) el("shiftInput").value = draft.shift;
    if (Array.isArray(draft.uncond_skip_layers)) {
      el("skipLayersInput").value = draft.uncond_skip_layers.join(", ");
    }
    if (Array.isArray(draft.slg_layers)) {
      el("slgLayersInput").value = draft.slg_layers.join(", ");
    }
    if (draft.slg_scale !== undefined) el("slgScaleInput").value = draft.slg_scale;
    if (draft.slg_start !== undefined) el("slgStartInput").value = draft.slg_start;
    if (draft.slg_stop !== undefined) el("slgStopInput").value = draft.slg_stop;
    if (draft.eta !== undefined) el("etaInput").value = draft.eta;
    if (draft.s_noise !== undefined) el("sNoiseInput").value = draft.s_noise;
    if (draft.seed !== undefined) el("seedInput").value = draft.seed;
    if (draft.use_v2_grid !== undefined) el("useV2GridCheck").checked = Boolean(draft.use_v2_grid);
    if (draft.preserve_canvas_outside_boxes !== undefined) {
      el("preserveOutsideBoxesCheck").checked = Boolean(draft.preserve_canvas_outside_boxes);
    }
    if (draft.composite_box_images_to_init !== undefined) {
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
    if (draft.mask_blend_mode) el("maskBlendModeSelect").value = draft.mask_blend_mode;
    if (draft.invert_inpaint_mask !== undefined) {
      el("invertMaskCheck").checked = Boolean(draft.invert_inpaint_mask);
    }
    if (draft.edit_format) el("editFormatSelect").value = draft.edit_format;
    if (draft.framing_mode) el("framingModeSelect").value = draft.framing_mode;
    if (draft.input_role) el("inputRoleSelect").value = draft.input_role;
    if (draft.denoise_strength !== undefined) el("denoiseInput").value = draft.denoise_strength;
    if (draft.mtf_boost !== undefined) el("mtfBoostInput").value = draft.mtf_boost;
    if (draft.mask_feather_radius !== undefined) el("featherInput").value = draft.mask_feather_radius;
    if (draft.latent_clamp_sigma !== undefined) el("clampSigmaInput").value = draft.latent_clamp_sigma;
    if (draft.edit_lora_path !== undefined) el("editLoraSelect").value = draft.edit_lora_path || "";
    if (draft.edit_lora_scale !== undefined) el("loraScaleInput").value = draft.edit_lora_scale;

    if (Array.isArray(draft.layout_items)) {
      state.layoutItems = JSON.parse(JSON.stringify(draft.layout_items));
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

  function renderGallery() {
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
            <button class="btn btn-secondary compact-btn" data-restore-id="${item.image_id}">Restore</button>
            <button class="btn btn-secondary compact-btn" data-promo-init="${item.image_id}" title="Promote image to Vault and bind as Init Image">-&gt; Init</button>
            <button class="btn btn-secondary compact-btn" data-promo-ref="${item.image_id}" title="Promote image to Vault and bind as 4D Reference">-&gt; 4D Ref</button>
            <button class="btn btn-secondary compact-btn" data-del-img="${item.image_id}">Delete</button>
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
        if (found) restoreDraftToStudio(found);
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

    container.querySelectorAll("[data-del-img]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = btn.getAttribute("data-del-img");
        await apiFetch(`api/v1/images/${encodeURIComponent(id)}`, { method: "DELETE" });
        state.gallery = state.gallery.filter((g) => g.image_id !== id);
        renderGallery();
      });
    });
  }

  function monitorJobProgress(jobId) {
    if (state.activeEventSource) {
      state.activeEventSource.close();
      state.activeEventSource = null;
    }
    state.activeJobId = jobId;
    const wrap = el("progressWrap");
    const fill = el("progressBarFill");
    const stageTxt = el("progressStageText");
    wrap.hidden = false;

    const streamUrl = `api/v1/jobs/${encodeURIComponent(jobId)}/stream?token=${encodeURIComponent(state.token)}`;
    const es = new EventSource(streamUrl);
    state.activeEventSource = es;

    es.onmessage = async (ev) => {
      try {
        const status = JSON.parse(ev.data);
        fill.style.width = `${status.progress_pct || 0}%`;
        stageTxt.textContent = `[${status.progress_pct || 0}%] ${status.stage || status.status}`;

        if (status.status === "COMPLETED") {
          es.close();
          state.activeEventSource = null;
          el("synthesizeBtn").disabled = false;
          updateViewportAndTelemetry(status);
          if (state.user) {
            await authenticateUser(state.user.slug);
          }
          await pollHealthTelemetry(false);
        } else if (status.status === "FAILED") {
          es.close();
          state.activeEventSource = null;
          el("synthesizeBtn").disabled = false;
          stageTxt.textContent = `Synthesis Error: ${status.error || "Unknown failure"}`;
        }
      } catch (_) {}
    };

    es.onerror = () => {
      es.close();
      state.activeEventSource = null;
      el("synthesizeBtn").disabled = false;
    };
  }

  function collectDeckPayload() {
    const promptMode = el("promptModeSelect").value;
    const rawSeed = el("seedInput").value.trim();
    const loraVal = el("editLoraSelect").value.trim();
    const sysPrompt = el("systemPromptInput").value.trim();
    const thinkTrace = el("thinkingTraceInput").value.trim();
    const explicitLayoutSummary = el("layoutSummaryInput") ? el("layoutSummaryInput").value.trim() : "";

    return {
      title: el("titleInput").value.trim() || "Untitled Vision Master",
      model_variant: el("modelVariantSelect").value,
      prompt: el("promptInput").value.trim(),
      negative_prompt: el("negativePromptInput").value.trim(),
      system_prompt: sysPrompt || null,
      thinking_trace: thinkTrace || null,
      add_vision_id: el("addVisionIdCheck").checked,
      prompt_mode: promptMode,
      layout_summary: explicitLayoutSummary || el("promptInput").value.trim(),
      layout_items: state.layoutItems,
      layout_coords_are_pixels: true,
      preserve_canvas_outside_boxes: el("preserveOutsideBoxesCheck").checked,
      box_attn_isolation: parseOptionalFloat(el("boxAttnIsolationInput").value, 0.0, 1.0) ?? 0.0,
      uncond_uses_references: parseTriStateBool(el("uncondUsesRefsSelect").value),
      composite_box_images_to_init: el("compositeBoxImagesCheck").checked,
      mask_blend_mode: el("maskBlendModeSelect").value || "auto",
      invert_inpaint_mask: el("invertMaskCheck").checked,
      width: parseInt(el("widthInput").value, 10) || 1024,
      height: parseInt(el("heightInput").value, 10) || 1024,
      num_inference_steps: parseInt(el("stepsInput").value, 10) || 50,
      num_images_per_prompt: Math.max(1, Math.min(4, parseInt(el("numImagesInput").value, 10) || 1)),
      shift: parseFloat(el("shiftInput").value) || 5.0,
      use_v2_grid: el("useV2GridCheck").checked,
      early_solver: el("earlySolverSelect").value,
      late_solver: el("lateSolverSelect").value,
      handoff_threshold: parseFloat(el("handoffInput").value),
      early_cfg: parseFloat(el("earlyCfgInput").value),
      late_cfg: parseFloat(el("lateCfgInput").value),
      uncond_skip_layers: parseIntList(el("skipLayersInput").value),
      slg_layers: parseIntList(el("slgLayersInput").value),
      slg_scale: parseFloat(el("slgScaleInput").value) || 0.0,
      slg_start: parseOptionalFloat(el("slgStartInput").value, 0.0, 1.0) ?? 0.20,
      slg_stop: parseOptionalFloat(el("slgStopInput").value, 0.0, 1.0) ?? 0.70,
      eta: parseFloat(el("etaInput").value) || 0.0,
      s_noise: parseFloat(el("sNoiseInput").value) || 1.0,
      max_sequence_length: parseInt(el("maxSeqLenInput").value, 10) || 4096,
      max_vit_patches: parseInt(el("maxVitPatchesInput").value, 10) || 2304,
      enable_thinking: el("enableThinkingCheck").checked,
      init_image_path: state.bindings.init_image_path,
      denoise_strength: parseFloat(el("denoiseInput").value) || 1.0,
      inpaint_mask_path: state.bindings.inpaint_mask_path,
      control_image_path: state.bindings.control_image_path,
      reference_image_paths: state.bindings.reference_image_paths,
      edit_format: el("editFormatSelect").value,
      framing_mode: el("framingModeSelect").value,
      input_role: el("inputRoleSelect").value,
      mtf_boost: parseFloat(el("mtfBoostInput").value) || 0.08,
      soft_knee_tau: parseFloat(el("softKneeTauInput").value) || 0.015,
      mask_feather_radius: parseFloat(el("featherInput").value) || 1.25,
      latent_clamp_sigma: parseFloat(el("clampSigmaInput").value) || 4.5,
      deband_8bit: el("deband8bitCheck").checked,
      apply_bayer_dither: el("applyBayerDitherCheck").checked,
      use_reference_cache: el("useRefCacheCheck").checked,
      prefer_vulkan: el("preferVulkanCheck").checked,
      cpu_offload: el("cpuOffloadCheck").checked,
      edit_lora_path: loraVal || null,
      edit_lora_scale: parseFloat(el("loraScaleInput").value) || 1.0,
      export_vizj: el("exportVizjCheck").checked,
      export_vizj_tier: 2,
      seed: rawSeed !== "" ? parseInt(rawSeed, 10) : null
    };
  }

  async function triggerSynthesis() {
    if (!state.token) {
      showLoginModal();
      return;
    }
    const btn = el("synthesizeBtn");
    const wrap = el("progressWrap");
    const fill = el("progressBarFill");
    const stageTxt = el("progressStageText");

    btn.disabled = true;
    wrap.hidden = false;
    fill.style.width = "2%";
    stageTxt.textContent = "Acquiring Cryptographic PoW Challenge...";

    try {
      const ch = await apiFetch("api/v1/auth/challenge");
      stageTxt.textContent = `Mining SHA-256 Proof-of-Work (Difficulty ${ch.difficulty})...`;
      const solutionNonce = await solvePowInWorker(ch.challenge, ch.difficulty);

      const payload = collectDeckPayload();
      payload.pow = {
        challenge: ch.challenge,
        signature: ch.signature,
        solution_nonce: solutionNonce
      };

      stageTxt.textContent = "Submitting Job to Sovereign Compute Queue...";
      const jobStatus = await apiFetch("api/v1/synthesize", {
        method: "POST",
        body: payload
      });
      monitorJobProgress(jobStatus.job_id);
    } catch (err) {
      btn.disabled = false;
      stageTxt.textContent = `Dispatch Error: ${err.message}`;
    }
  }

  async function saveActiveDeckAsEngineDefault() {
    try {
      const payload = collectDeckPayload();
      const res = await apiFetch("api/v1/presets/default", {
        method: "POST",
        body: { engine_defaults: payload }
      });
      if (res && res.baseline_parameters) {
        applyEngineDefaultsToDeck(res.baseline_parameters);
      }
      await pollHealthTelemetry(false);
    } catch (err) {
      alert(`Failed updating default.json: ${err.message}`);
    }
  }

  async function stageDefaultTestSet() {
    if (!state.token) {
      showLoginModal();
      return;
    }
    const btn = el("stageDefaultRefsBtn");
    btn.disabled = true;
    try {
      await apiFetch("api/v1/references/stage-defaults", { method: "POST" });
      await refreshReferences();
    } catch (err) {
      alert(`Failed staging test set: ${err.message}`);
    } finally {
      btn.disabled = false;
    }
  }

  function bindEventListeners() {
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
        const oldW = parseInt(el("widthInput").value, 10) || 1024;
        const oldH = parseInt(el("heightInput").value, 10) || 1024;
        applyCanvasResolutionChange(oldW, oldH, parseInt(wStr, 10), parseInt(hStr, 10));
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
      const geom = computeStageGeometry();
      const y0 = Math.round(geom.canvasH * 0.25);
      const x0 = Math.round(geom.canvasW * 0.25);
      const y1 = Math.round(geom.canvasH * 0.75);
      const x1 = Math.round(geom.canvasW * 0.75);
      state.layoutItems.push({
        category: "object",
        caption: "New spatial subject region",
        bbox: [y0, x0, y1, x1],
        enabled: true,
        inpaint_box: false,
        paste_to_canvas: false,
        image_path: "",
        input_role: "",
        framing_mode: "",
        isolate_box_attn: null,
        denoise_strength: null
      });
      el("promptModeSelect").value = "layout_json";
      renderSpatialLayoutEditor();
    });

    el("addTextBoxBtn").addEventListener("click", () => {
      const geom = computeStageGeometry();
      const y0 = Math.round(geom.canvasH * 0.12);
      const x0 = Math.round(geom.canvasW * 0.25);
      const y1 = Math.round(geom.canvasH * 0.28);
      const x1 = Math.round(geom.canvasW * 0.75);
      state.layoutItems.push({
        category: "text",
        text: "SOVEREIGN 2026",
        bbox: [y0, x0, y1, x1],
        enabled: true,
        inpaint_box: false,
        paste_to_canvas: false,
        image_path: "",
        input_role: "",
        framing_mode: "",
        isolate_box_attn: null,
        denoise_strength: null
      });
      el("promptModeSelect").value = "layout_json";
      renderSpatialLayoutEditor();
    });

    el("clearAllBoxesBtn").addEventListener("click", () => {
      state.layoutItems = [];
      renderSpatialLayoutEditor();
    });

    el("resetDefaultBoxesBtn").addEventListener("click", () => {
      state.layoutItems = JSON.parse(JSON.stringify(DEFAULT_LAYOUT_ITEMS));
      renderSpatialLayoutEditor();
    });

    el("promptInput").addEventListener("input", updateCompiledJsonPreview);
    el("layoutSummaryInput").addEventListener("input", updateCompiledJsonPreview);
    el("promptModeSelect").addEventListener("change", updateCompiledJsonPreview);
    el("editFormatSelect").addEventListener("change", renderSpatialLayoutEditor);
    el("inputRoleSelect").addEventListener("change", renderSpatialLayoutEditor);
    el("framingModeSelect").addEventListener("change", renderSpatialLayoutEditor);
    el("boxAttnIsolationInput").addEventListener("input", renderSpatialLayoutEditor);
    el("compositeBoxImagesCheck").addEventListener("change", renderSpatialLayoutEditor);
    el("preserveOutsideBoxesCheck").addEventListener("change", renderBindingsCard);
    el("maskBlendModeSelect").addEventListener("change", renderBindingsCard);
    el("invertMaskCheck").addEventListener("change", renderBindingsCard);

    el("layoutUnderlaySelect").addEventListener("change", syncUnderlayLayers);
    el("layoutUnderlayOpacity").addEventListener("input", syncUnderlayLayers);
    el("inpaintUnderlayOpacity").addEventListener("input", syncUnderlayLayers);

    el("widthInput").addEventListener("change", () => {
      syncStageAspectAndGuides();
      renderSpatialLayoutEditor();
    });
    el("heightInput").addEventListener("change", () => {
      syncStageAspectAndGuides();
      renderSpatialLayoutEditor();
    });

    el("referenceFileInput").addEventListener("change", async (ev) => {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          await uploadDataUrlToVault(file.name, el("inputRoleSelect").value, reader.result);
        } catch (err) {
          alert(`Upload failed: ${err.message}`);
        } finally {
          ev.target.value = "";
        }
      };
      reader.readAsDataURL(file);
    });
  }

  async function bootstrapStudio() {
    bindEventListeners();
    initInpaintCanvas();
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

  document.addEventListener("DOMContentLoaded", bootstrapStudio);
})();