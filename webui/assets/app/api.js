// webui/assets/app/api.js
import {
  state,
  el,
  ensureBoxZOrders,
  applyThinkingTraceToDeck
} from "./state.js";
import { renderBindingsCard } from "./geometry.js";
import { renderGallery, refreshReferences, refreshProjects } from "./vault_projects.js";

export function showLoginModal() {
  el("loginModal").hidden = false;
  el("loginErrorMsg").hidden = true;
  el("loginUsernameInput").focus();
}

export function hideLoginModal() {
  el("loginModal").hidden = true;
}

export async function apiFetch(endpoint, options = {}) {
  const cleanEndpoint = endpoint.replace(/^\/+/, "");
  const headers = Object.assign(
    {
      "Cache-Control": "no-cache",
      "Pragma": "no-cache"
    },
    options.headers || {}
  );
  if (state.token) {
    headers["Authorization"] = `Bearer ${state.token}`;
  }
  const fetchOpts = Object.assign({ cache: "no-store" }, options, { headers });
  if (fetchOpts.body && typeof fetchOpts.body === "object" && !(fetchOpts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    fetchOpts.body = JSON.stringify(fetchOpts.body);
  }
  const resp = await fetch(cleanEndpoint, fetchOpts);
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

export function applyThemeVariables(themeId) {
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
  if (state.user && state.user.slug) {
    localStorage.setItem("vs_theme_user", state.user.slug);
  }
}

export async function loadThemeRegistry() {
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

export function updateAvailableModelsDropdown(availableModels) {
  if (!Array.isArray(availableModels) || !availableModels.length) return;
  state.availableModels = availableModels;
  const sel = el("modelVariantSelect");
  if (!sel) return;
  const baseLabels = {
    sft: "MOSS-Image-SFT (Sharp Layout & Text)",
    base: "MOSS-Image-Base (Foundation Diversity)"
  };
  for (const info of availableModels) {
    const variant = String(info.variant || "").toLowerCase();
    const opt = sel.querySelector(`option[value="${variant}"]`);
    if (opt && baseLabels[variant]) {
      const statusSuffix = info.cached ? "" : " [Not Cached]";
      opt.textContent = `${baseLabels[variant]}${statusSuffix}`;
    }
  }
}

export function applyEngineDefaultsToDeck(defaults) {
  if (!defaults || typeof defaults !== "object") return;
  state.engineDefaults = defaults;

  if (defaults.model_variant) el("modelVariantSelect").value = defaults.model_variant;
  if (defaults.prompt_mode) el("promptModeSelect").value = defaults.prompt_mode;
  if (defaults.system_prompt !== undefined) el("systemPromptInput").value = defaults.system_prompt || "";
  if (defaults.thinking_trace !== undefined) {
    applyThinkingTraceToDeck(defaults.thinking_trace || "");
  }
  if (defaults.add_vision_id !== undefined && defaults.add_vision_id !== null) {
    el("addVisionIdCheck").checked = Boolean(defaults.add_vision_id);
  }
  if (defaults.max_sequence_length !== undefined && defaults.max_sequence_length !== null) {
    el("maxSeqLenInput").value = defaults.max_sequence_length;
  }
  if (defaults.max_vit_patches !== undefined && defaults.max_vit_patches !== null) {
    el("maxVitPatchesInput").value = defaults.max_vit_patches;
  }
  if (defaults.width !== undefined && defaults.width !== null) el("widthInput").value = defaults.width;
  if (defaults.height !== undefined && defaults.height !== null) el("heightInput").value = defaults.height;
  if (defaults.num_inference_steps !== undefined && defaults.num_inference_steps !== null) {
    el("stepsInput").value = defaults.num_inference_steps;
  }
  if (defaults.num_images_per_prompt !== undefined && defaults.num_images_per_prompt !== null) {
    el("numImagesInput").value = defaults.num_images_per_prompt;
  }
  if (defaults.early_solver) el("earlySolverSelect").value = defaults.early_solver;
  if (defaults.late_solver) el("lateSolverSelect").value = defaults.late_solver;
  if (defaults.handoff_threshold !== undefined && defaults.handoff_threshold !== null) {
    el("handoffInput").value = defaults.handoff_threshold;
  }
  if (defaults.early_cfg !== undefined && defaults.early_cfg !== null) {
    el("earlyCfgInput").value = defaults.early_cfg;
  }
  if (defaults.late_cfg !== undefined && defaults.late_cfg !== null) {
    el("lateCfgInput").value = defaults.late_cfg;
  }
  if (defaults.shift !== undefined && defaults.shift !== null) {
    el("shiftInput").value = defaults.shift;
  }
  if (Array.isArray(defaults.uncond_skip_layers)) {
    el("skipLayersInput").value = defaults.uncond_skip_layers.join(", ");
  }
  if (Array.isArray(defaults.slg_layers) && defaults.slg_layers.length > 0) {
    el("slgLayersInput").value = defaults.slg_layers.join(", ");
  }
  if (defaults.slg_scale !== undefined && defaults.slg_scale !== null) {
    el("slgScaleInput").value = defaults.slg_scale;
  }
  if (defaults.slg_start !== undefined && defaults.slg_start !== null) {
    el("slgStartInput").value = defaults.slg_start;
  }
  if (defaults.slg_stop !== undefined && defaults.slg_stop !== null) {
    el("slgStopInput").value = defaults.slg_stop;
  }
  if (defaults.eta !== undefined && defaults.eta !== null) {
    el("etaInput").value = defaults.eta;
  }
  if (defaults.s_noise !== undefined && defaults.s_noise !== null) {
    el("sNoiseInput").value = defaults.s_noise;
  }
  if (defaults.enable_thinking !== undefined && defaults.enable_thinking !== null) {
    el("enableThinkingCheck").checked = Boolean(defaults.enable_thinking);
  }
  if (defaults.use_v2_grid !== undefined && defaults.use_v2_grid !== null) {
    el("useV2GridCheck").checked = Boolean(defaults.use_v2_grid);
  }
  if (defaults.preserve_canvas_outside_boxes !== undefined && defaults.preserve_canvas_outside_boxes !== null) {
    el("preserveOutsideBoxesCheck").checked = Boolean(defaults.preserve_canvas_outside_boxes);
  }
  if (defaults.composite_box_images_to_init !== undefined && defaults.composite_box_images_to_init !== null) {
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
  if (el("refZeroTimestepSelect")) {
    if (defaults.reference_zero_timestep === true) {
      el("refZeroTimestepSelect").value = "true";
    } else if (defaults.reference_zero_timestep === false) {
      el("refZeroTimestepSelect").value = "false";
    } else {
      el("refZeroTimestepSelect").value = "";
    }
  }
  if (el("refIsolatedKvSelect")) {
    if (defaults.reference_isolated_kv === true) {
      el("refIsolatedKvSelect").value = "true";
    } else if (defaults.reference_isolated_kv === false) {
      el("refIsolatedKvSelect").value = "false";
    } else {
      el("refIsolatedKvSelect").value = "";
    }
  }
  if (defaults.mask_blend_mode) el("maskBlendModeSelect").value = defaults.mask_blend_mode;
  if (defaults.invert_inpaint_mask !== undefined && defaults.invert_inpaint_mask !== null) {
    el("invertMaskCheck").checked = Boolean(defaults.invert_inpaint_mask);
  }
  if (defaults.edit_format) el("editFormatSelect").value = defaults.edit_format;
  if (defaults.framing_mode) el("framingModeSelect").value = defaults.framing_mode;
  if (defaults.input_role) el("inputRoleSelect").value = defaults.input_role;
  if (defaults.denoise_strength !== undefined && defaults.denoise_strength !== null) {
    el("denoiseInput").value = defaults.denoise_strength;
  }
  if (defaults.mtf_boost !== undefined && defaults.mtf_boost !== null) {
    el("mtfBoostInput").value = defaults.mtf_boost;
  }
  if (defaults.soft_knee_tau !== undefined && defaults.soft_knee_tau !== null) {
    el("softKneeTauInput").value = defaults.soft_knee_tau;
  }
  if (defaults.mask_feather_radius !== undefined && defaults.mask_feather_radius !== null) {
    el("featherInput").value = defaults.mask_feather_radius;
  }
  if (defaults.latent_clamp_sigma !== undefined && defaults.latent_clamp_sigma !== null) {
    el("clampSigmaInput").value = defaults.latent_clamp_sigma;
  }
  if (defaults.deband_8bit !== undefined && defaults.deband_8bit !== null) {
    el("deband8bitCheck").checked = Boolean(defaults.deband_8bit);
  }
  if (defaults.apply_bayer_dither !== undefined && defaults.apply_bayer_dither !== null) {
    el("applyBayerDitherCheck").checked = Boolean(defaults.apply_bayer_dither);
  }
  if (defaults.use_reference_cache !== undefined && defaults.use_reference_cache !== null) {
    el("useRefCacheCheck").checked = Boolean(defaults.use_reference_cache);
  }
  if (defaults.prefer_vulkan !== undefined && defaults.prefer_vulkan !== null) {
    el("preferVulkanCheck").checked = Boolean(defaults.prefer_vulkan);
  }
  if (defaults.cpu_offload !== undefined && defaults.cpu_offload !== null) {
    el("cpuOffloadCheck").checked = Boolean(defaults.cpu_offload);
  }
  if (el("prepassWarmupCheck") && defaults.prepass_warmup !== undefined && defaults.prepass_warmup !== null) {
    el("prepassWarmupCheck").checked = Boolean(defaults.prepass_warmup);
  }
  if (defaults.edit_lora_path) {
    el("editLoraSelect").value = defaults.edit_lora_path;
  } else {
    el("editLoraSelect").value = "";
  }
  const effLoraScale = defaults.lora_scale ?? defaults.edit_lora_scale;
  if (effLoraScale !== undefined && effLoraScale !== null) {
    el("loraScaleInput").value = effLoraScale;
  }
  renderBindingsCard();
}

export async function pollHealthTelemetry(applyDefaults = false) {
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

    if (Array.isArray(data.available_models)) {
      updateAvailableModelsDropdown(data.available_models);
    }

    if (applyDefaults && bp) {
      applyEngineDefaultsToDeck(bp);
    }
  } catch (err) {
    el("hwStatusLine").textContent = `Daemon Offline: ${err.message}`;
  }
}

export async function authenticateUser(username) {
  const cleanUser = (username || "").trim();
  if (!cleanUser) return;
  el("loginErrorMsg").hidden = true;
  try {
    const data = await apiFetch("api/v1/auth/login", {
      method: "POST",
      body: { username: cleanUser }
    });
    const savedTheme = localStorage.getItem("vs_active_theme");
    const savedThemeUser = localStorage.getItem("vs_theme_user");

    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("vs_session_token", data.token);
    localStorage.setItem("vs_last_username", cleanUser);

    el("authBtn").textContent = `${data.user.display_name} (${data.user.slug})`;
    el("quotaBadge").textContent = data.user.is_admin
      ? "Quota: UNLIMITED"
      : `Quota: ${data.user.tokens_remaining}/${data.user.daily_quota}`;

    el("saveDefaultPresetBtn").hidden = !data.user.is_admin;

    if (Array.isArray(data.available_models)) {
      updateAvailableModelsDropdown(data.available_models);
    }

    if (data.theme && data.theme.theme_id) {
      if (data.theme.manifest) {
        state.themes[data.theme.theme_id] = data.theme.manifest;
      }
      const targetTheme =
        savedThemeUser === data.user.slug && savedTheme && state.themes[savedTheme]
          ? savedTheme
          : data.theme.theme_id;
      applyThemeVariables(targetTheme);
    }

    if (data.engine_defaults && Object.keys(state.engineDefaults).length === 0) {
      applyEngineDefaultsToDeck(data.engine_defaults);
    }
    if (Array.isArray(data.default_layout_items) && data.default_layout_items.length > 0 && state.layoutItems.length === 0) {
      state.layoutItems = JSON.parse(JSON.stringify(data.default_layout_items));
      ensureBoxZOrders();
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

export async function uploadDataUrlToVault(filename, role, dataUrl) {
  const res = await apiFetch("api/v1/references/upload", {
    method: "POST",
    body: { filename, role, data_url: dataUrl }
  });
  await refreshReferences();
  return res;
}

export function sha256Hex(ascii) {
  function rightRotate(value, amount) {
    return (value >>> amount) | (value << (32 - amount));
  }
  const maxWord = Math.pow(2, 32);
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
  ascii += "\x80";
  while ((ascii.length % 64) - 56) ascii += "\x00";
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

export function minePowTimeSliced(challenge, difficulty) {
  return new Promise((resolve) => {
    const target = "0".repeat(Math.max(0, difficulty | 0));
    let n = 0;
    function stepSlice() {
      const limit = n + 2048;
      while (n < limit) {
        const cand = n.toString(16);
        if (sha256Hex(`${challenge}:${cand}`).startsWith(target)) {
          resolve(cand);
          return;
        }
        n++;
      }
      setTimeout(stepSlice, 0);
    }
    stepSlice();
  });
}

export function solvePowInWorker(challenge, difficulty) {
  if (typeof Worker === "undefined" || typeof Blob === "undefined" || typeof URL === "undefined") {
    return minePowTimeSliced(challenge, difficulty);
  }
  return new Promise((resolve) => {
    let settled = false;
    let url = null;
    let worker = null;

    function cleanup() {
      if (url) {
        try {
          URL.revokeObjectURL(url);
        } catch (_) {}
        url = null;
      }
      if (worker) {
        try {
          worker.terminate();
        } catch (_) {}
        worker = null;
      }
    }

    function fallbackToMainThread() {
      if (settled) return;
      settled = true;
      cleanup();
      minePowTimeSliced(challenge, difficulty).then(resolve);
    }

    try {
      const workerCode = `${sha256Hex.toString()}
      self.onmessage = (e) => {
        const { challenge, difficulty } = e.data;
        const target = "0".repeat(Math.max(0, difficulty | 0));
        let n = 0;
        while (true) {
          const cand = n.toString(16);
          if (sha256Hex(challenge + ":" + cand).startsWith(target)) {
            self.postMessage(cand);
            break;
          }
          n++;
        }
      };`;
      const blob = new Blob([workerCode], { type: "application/javascript" });
      url = URL.createObjectURL(blob);
      worker = new Worker(url);
      worker.onmessage = (e) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(e.data);
      };
      worker.onerror = fallbackToMainThread;
      worker.onmessageerror = fallbackToMainThread;
      worker.postMessage({ challenge, difficulty });
    } catch (_) {
      fallbackToMainThread();
    }
  });
}