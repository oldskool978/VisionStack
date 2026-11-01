// webui/assets/app/synthesis.js
import {
  state,
  el,
  parseIntList,
  parseOptionalFloat,
  parseFiniteFloat,
  parseFiniteInt,
  parseTriStateBool,
  resolveThinkingTraceFromDeck,
  showStudioToast
} from "./state.js";
import {
  apiFetch,
  showLoginModal,
  solvePowInWorker,
  authenticateUser,
  applyEngineDefaultsToDeck,
  pollHealthTelemetry
} from "./api.js";
import { updateViewportAndTelemetry } from "./vault_projects.js";

export function monitorJobProgress(jobId) {
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
      const pct = parseFiniteInt(status.progress_pct, 0, 0, 100);
      fill.style.width = `${pct}%`;
      const aheadTag = status.users_ahead > 0 ? ` (Queue #${status.users_ahead + 1})` : "";
      stageTxt.textContent = `[${pct}%]${aheadTag} ${status.stage || status.status}`;

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
        const errMsg = status.error || "Unknown failure";
        stageTxt.textContent = `Synthesis Error: ${errMsg}`;
        showStudioToast(`Synthesis failed: ${errMsg}`, "error");
      }
    } catch (_) {}
  };

  es.onerror = () => {
    es.close();
    state.activeEventSource = null;
    el("synthesizeBtn").disabled = false;
  };
}

export function collectDeckPayload() {
  const promptMode = el("promptModeSelect").value;
  const rawSeed = el("seedInput").value.trim();
  const loraVal = el("editLoraSelect").value.trim();
  const sysPrompt = el("systemPromptInput").value.trim();
  const resolvedThinkTrace = resolveThinkingTraceFromDeck();
  const explicitLayoutSummary = el("layoutSummaryInput") ? el("layoutSummaryInput").value.trim() : "";

  const slgStart = parseFiniteFloat(el("slgStartInput").value, 0.20, 0.0, 1.0);
  const slgStop = Math.max(slgStart, parseFiniteFloat(el("slgStopInput").value, 0.70, 0.0, 1.0));
  const parsedSeed = rawSeed !== "" ? parseOptionalFloat(rawSeed, 0, Number.MAX_SAFE_INTEGER) : null;

  return {
    title: el("titleInput").value.trim() || "Untitled Vision Master",
    model_variant: el("modelVariantSelect").value,
    prompt: el("promptInput").value.trim(),
    negative_prompt: el("negativePromptInput").value.trim(),
    system_prompt: sysPrompt || null,
    thinking_trace: resolvedThinkTrace,
    add_vision_id: el("addVisionIdCheck").checked,
    prompt_mode: promptMode,
    layout_summary: explicitLayoutSummary || null,
    layout_items: state.layoutItems,
    layout_coords_are_pixels: true,
    preserve_canvas_outside_boxes: el("preserveOutsideBoxesCheck").checked,
    box_attn_isolation: parseFiniteFloat(el("boxAttnIsolationInput").value, 0.0, 0.0, 1.0),
    uncond_uses_references: parseTriStateBool(el("uncondUsesRefsSelect").value),
    reference_zero_timestep: parseTriStateBool(
      el("refZeroTimestepSelect") ? el("refZeroTimestepSelect").value : ""
    ),
    reference_isolated_kv: parseTriStateBool(
      el("refIsolatedKvSelect") ? el("refIsolatedKvSelect").value : ""
    ),
    composite_box_images_to_init: el("compositeBoxImagesCheck").checked,
    mask_blend_mode: el("maskBlendModeSelect").value || "auto",
    invert_inpaint_mask: el("invertMaskCheck").checked,
    width: parseFiniteInt(el("widthInput").value, 1024, 256, 2048),
    height: parseFiniteInt(el("heightInput").value, 1024, 256, 2048),
    num_inference_steps: parseFiniteInt(el("stepsInput").value, 50, 1, 200),
    num_images_per_prompt: parseFiniteInt(el("numImagesInput").value, 1, 1, 8),
    shift: parseFiniteFloat(el("shiftInput").value, 5.0, 0.1, 20.0),
    use_v2_grid: el("useV2GridCheck").checked,
    early_solver: el("earlySolverSelect").value,
    late_solver: el("lateSolverSelect").value,
    handoff_threshold: parseFiniteFloat(el("handoffInput").value, 0.35, 0.0, 1.0),
    early_cfg: parseFiniteFloat(el("earlyCfgInput").value, 3.2, 0.0, 30.0),
    late_cfg: parseFiniteFloat(el("lateCfgInput").value, 2.8, 0.0, 30.0),
    uncond_skip_layers: parseIntList(el("skipLayersInput").value),
    slg_layers: parseIntList(el("slgLayersInput").value),
    slg_scale: parseFiniteFloat(el("slgScaleInput").value, 0.0, 0.0, 20.0),
    slg_start: slgStart,
    slg_stop: slgStop,
    eta: parseFiniteFloat(el("etaInput").value, 0.0, 0.0, 1.0),
    s_noise: parseFiniteFloat(el("sNoiseInput").value, 1.0, 0.0, 5.0),
    max_sequence_length: parseFiniteInt(el("maxSeqLenInput").value, 16384, 64, 32768),
    max_vit_patches: parseFiniteInt(el("maxVitPatchesInput").value, 2304, 64, 9216),
    enable_thinking: el("enableThinkingCheck").checked,
    init_image_path: state.bindings.init_image_path,
    denoise_strength: parseFiniteFloat(el("denoiseInput").value, 1.0, 0.01, 1.0),
    inpaint_mask_path: state.bindings.inpaint_mask_path,
    control_image_path: state.bindings.control_image_path,
    reference_image_paths: state.bindings.reference_image_paths,
    edit_format: el("editFormatSelect").value,
    framing_mode: el("framingModeSelect").value,
    input_role: el("inputRoleSelect").value,
    mtf_boost: parseFiniteFloat(el("mtfBoostInput").value, 0.08, 0.0, 0.5),
    soft_knee_tau: parseFiniteFloat(el("softKneeTauInput").value, 0.015, 0.001, 0.25),
    mask_feather_radius: parseFiniteFloat(el("featherInput").value, 1.25, 0.0, 6.0),
    latent_clamp_sigma: parseFiniteFloat(el("clampSigmaInput").value, 4.5, 2.0, 12.0),
    deband_8bit: el("deband8bitCheck").checked,
    apply_bayer_dither: el("applyBayerDitherCheck").checked,
    use_reference_cache: el("useRefCacheCheck").checked,
    prefer_vulkan: el("preferVulkanCheck").checked,
    cpu_offload: el("cpuOffloadCheck").checked,
    prepass_warmup: el("prepassWarmupCheck") ? el("prepassWarmupCheck").checked : true,
    edit_lora_path: loraVal || null,
    edit_lora_scale: parseFiniteFloat(el("loraScaleInput").value, 1.0, 0.0, 4.0),
    export_vizj: el("exportVizjCheck").checked,
    export_vizj_tier: 2,
    seed: parsedSeed !== null ? Math.floor(parsedSeed) : null
  };
}

export async function triggerSynthesis() {
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
    btn.disabled = false;
    monitorJobProgress(jobStatus.job_id);
    await pollHealthTelemetry(false);
  } catch (err) {
    btn.disabled = false;
    stageTxt.textContent = `Dispatch Error: ${err.message}`;
    showStudioToast(`Dispatch error: ${err.message}`, "error");
  }
}

export async function saveActiveDeckAsEngineDefault() {
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
    showStudioToast("Locked active deck to ShadowBoxer/default.json", "success");
  } catch (err) {
    showStudioToast(`Failed updating default.json: ${err.message}`, "error");
  }
}