// webui/assets/app/state.js
export const DEFAULT_PROMPT_TEXT =
  'A wide view of a sleek brushed-steel refrigerator beside a warm walnut kitchen counter with a ceramic bowl of fresh citrus fruit. A small yellow sticky note on the refrigerator door is handwritten in crisp black marker: "Remember to test SFT & Base permutations @ 1024x1024"';

export const DEFAULT_BOX_REF_ISOLATION = 0.65;
export const DEFAULT_SPECULATIVE_COT_TOKENS = 64;

export const CHARACTER_SYNONYMS = new Set([
  "character",
  "person",
  "people",
  "human",
  "portrait",
  "face",
  "man",
  "woman",
  "boy",
  "girl",
  "child",
  "subject",
  "figure",
  "actor",
  "model"
]);

export const DEFAULT_LAYOUT_ITEMS = [
  {
    label: "object",
    category: "object",
    caption: "A sleek brushed-steel refrigerator door with a small yellow sticky note",
    bbox: [0, 0, 1024, 576],
    enabled: true,
    z_order: 1,
    inpaint_box: false,
    paste_to_canvas: false,
    image_path: "",
    input_role: "",
    framing_mode: "",
    isolate_box_attn: null,
    denoise_strength: null
  },
  {
    label: "text",
    category: "text",
    caption: "Handwritten crisp black marker text on a small yellow sticky note",
    text: "Remember to test SFT & Base permutations @ 1024x1024",
    bbox: [256, 112, 512, 464],
    enabled: true,
    z_order: 3,
    inpaint_box: false,
    paste_to_canvas: false,
    image_path: "",
    input_role: "",
    framing_mode: "",
    isolate_box_attn: null,
    denoise_strength: null
  },
  {
    label: "object",
    category: "object",
    caption: "A ceramic bowl of fresh citrus fruit on a warm walnut kitchen counter",
    bbox: [448, 544, 1024, 1024],
    enabled: true,
    z_order: 2,
    inpaint_box: false,
    paste_to_canvas: false,
    image_path: "",
    input_role: "",
    framing_mode: "",
    isolate_box_attn: null,
    denoise_strength: null
  }
];

export const MAX_MASK_HISTORY = 20;

export const state = {
  token: localStorage.getItem("vs_session_token") || "",
  user: null,
  themes: {},
  engineDefaults: {},
  availableModels: [],
  layoutItems: JSON.parse(JSON.stringify(DEFAULT_LAYOUT_ITEMS)),
  selectedBoxIndex: 0,
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
  showViewportHud: false,
  showCompareSplit: false,
  brushMode: "paint",
  isPainting: false,
  lastPaintPt: null,
  maskUndoStack: [],
  maskRedoStack: []
};

export const el = (id) => document.getElementById(id);

export function showStudioToast(message, kind = "info", durationMs = 4200) {
  if (!message || typeof document === "undefined") return;
  let stack = document.querySelector(".studio-toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.className = "studio-toast-stack";
    document.body.appendChild(stack);
  }
  const toast = document.createElement("div");
  const normKind = ["success", "warn", "error", "info"].includes(kind) ? kind : "info";
  toast.className = `studio-toast toast-${normKind}`;
  const span = document.createElement("span");
  span.textContent = String(message);
  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "btn-link";
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", () => toast.remove());
  toast.appendChild(span);
  toast.appendChild(closeBtn);
  stack.appendChild(toast);

  if (durationMs > 0) {
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
    }, durationMs);
  }
}

export function normalizeBoxLabel(rawLabel, hasText = false) {
  if (hasText) return "text";
  const s = String(rawLabel || "").trim().toLowerCase();
  if (["text", "typography", "caption", "title", "sign", "label", "lettering"].includes(s)) {
    return "text";
  }
  if (CHARACTER_SYNONYMS.has(s)) {
    return "character";
  }
  return "object";
}

export function isTypographyBox(item) {
  if (!item || typeof item !== "object") return false;
  const rawTxt = typeof item.text === "string" ? item.text.trim() : "";
  return normalizeBoxLabel(item.label || item.category, Boolean(rawTxt)) === "text";
}

export function parseIntList(raw) {
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

export function parseOptionalFloat(raw, minVal, maxVal) {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s) return null;
  const val = Number(s);
  if (!Number.isFinite(val)) return null;
  return Math.max(minVal, Math.min(maxVal, val));
}

export function parseFiniteFloat(raw, fallback, minVal = -Infinity, maxVal = Infinity) {
  if (raw === null || raw === undefined) return fallback;
  const s = String(raw).trim();
  if (!s) return fallback;
  const val = Number(s);
  if (!Number.isFinite(val)) return fallback;
  return Math.max(minVal, Math.min(maxVal, val));
}

export function parseFiniteInt(raw, fallback, minVal = -Infinity, maxVal = Infinity) {
  if (raw === null || raw === undefined) return fallback;
  const s = String(raw).trim();
  if (!s) return fallback;
  const val = parseInt(s, 10);
  if (!Number.isFinite(val)) return fallback;
  return Math.max(minVal, Math.min(maxVal, val));
}

export function parseTriStateBool(raw) {
  if (raw === "true" || raw === true) return true;
  if (raw === "false" || raw === false) return false;
  return null;
}

export function syncSpecCotInputDisabledState() {
  const modeSel = el("thinkingTraceModeSelect");
  const specInput = el("specCotTokensInput");
  if (specInput && modeSel) {
    specInput.disabled = modeSel.value !== "speculative";
  }
}

export function resolveThinkingTraceFromDeck() {
  const modeSel = el("thinkingTraceModeSelect");
  const specInput = el("specCotTokensInput");
  const rawInput = el("thinkingTraceInput");
  const rawStr = rawInput ? rawInput.value.trim() : "";

  if (!modeSel) {
    return rawStr || null;
  }

  const mode = modeSel.value;
  if (mode === "auto") {
    return "auto";
  }
  if (mode === "speculative") {
    const budget = parseFiniteInt(
      specInput ? specInput.value : DEFAULT_SPECULATIVE_COT_TOKENS,
      DEFAULT_SPECULATIVE_COT_TOKENS,
      4,
      512
    );
    return `speculative:${budget}`;
  }
  if (mode === "custom") {
    return rawStr || null;
  }
  if (rawStr) {
    return rawStr;
  }
  return null;
}

export function applyThinkingTraceToDeck(rawTrace) {
  const modeSel = el("thinkingTraceModeSelect");
  const specInput = el("specCotTokensInput");
  const rawInput = el("thinkingTraceInput");
  const s = typeof rawTrace === "string" ? rawTrace.trim() : "";
  const sLower = s.toLowerCase();

  if (rawInput) {
    rawInput.value = s;
  }

  if (!s) {
    if (modeSel) modeSel.value = "";
    syncSpecCotInputDisabledState();
    return;
  }
  if (["auto", "spatial", "compile"].includes(sLower)) {
    if (modeSel) modeSel.value = "auto";
    syncSpecCotInputDisabledState();
    return;
  }
  if (sLower === "speculative" || sLower.startsWith("speculative:")) {
    if (modeSel) modeSel.value = "speculative";
    let tokBudget = DEFAULT_SPECULATIVE_COT_TOKENS;
    if (sLower.includes(":")) {
      const tail = sLower.split(":", 2)[1];
      tokBudget = parseFiniteInt(tail, DEFAULT_SPECULATIVE_COT_TOKENS, 4, 512);
    }
    if (specInput) {
      specInput.value = String(tokBudget);
    }
    syncSpecCotInputDisabledState();
    return;
  }
  if (modeSel) {
    modeSel.value = "custom";
  }
  syncSpecCotInputDisabledState();
}

export function gcd(a, b) {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y) {
    const t = y;
    y = x % y;
    x = t;
  }
  return x || 1;
}

export function resolveEffectiveBoxIsolation(item, height, width) {
  if (!item || !item.image_path || !String(item.image_path).trim()) return 0.0;
  const explicitIso = parseOptionalFloat(item.isolate_box_attn, 0.0, 1.0);
  if (explicitIso !== null) {
    return explicitIso;
  }
  const isoInput = el("boxAttnIsolationInput");
  const globalIso = parseFiniteFloat(isoInput ? isoInput.value : 0.0, 0.0, 0.0, 1.0);
  if (globalIso > 1e-6) return globalIso;
  if (Array.isArray(item.bbox) && item.bbox.length === 4) {
    const hDenom = Math.max(1, height);
    const wDenom = Math.max(1, width);
    const uy0 = Math.max(0.0, Math.min(1.0, Math.min(item.bbox[0], item.bbox[2]) / hDenom));
    const ux0 = Math.max(0.0, Math.min(1.0, Math.min(item.bbox[1], item.bbox[3]) / wDenom));
    const uy1 = Math.max(uy0, Math.min(1.0, Math.max(item.bbox[0], item.bbox[2]) / hDenom));
    const ux1 = Math.max(ux0, Math.min(1.0, Math.max(item.bbox[1], item.bbox[3]) / wDenom));
    const area = (uy1 - uy0) * (ux1 - ux0);
    return area < 0.98 ? DEFAULT_BOX_REF_ISOLATION : 0.0;
  }
  return DEFAULT_BOX_REF_ISOLATION;
}

export function ensureBoxZOrders() {
  state.layoutItems.forEach((b, idx) => {
    if (typeof b.z_order !== "number" || Number.isNaN(b.z_order)) {
      b.z_order = idx + 1;
    }
    const hasTxt = typeof b.text === "string" && Boolean(b.text.trim());
    const normLbl = normalizeBoxLabel(b.label || b.category, hasTxt);
    if (!b.label) b.label = normLbl;
    if (!b.category) b.category = normLbl;
  });
}

export function findRefMeta(refId) {
  if (!refId) return null;
  return state.references.find((r) => r.ref_id === refId) || null;
}

export function updateProjectBadge() {
  const badge = el("activeProjectBadge");
  if (!badge) return;
  if (state.activeProjectId) {
    const label = state.activeProjectTitle || state.activeProjectId;
    badge.textContent = `Project: ${label}`;
  } else {
    badge.textContent = "Project: Unsaved Draft";
  }
}

export function computeActiveMaskSummary() {
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
    return hasSplatBoxes && !hasInit ? "Tweedie Velocity-Space Box Splat Prior" : "Img2Img Full-Canvas Denoise";
  }

  const opStr = activeSources.length > 1 ? ` [${blendMode.toUpperCase()}]` : "";
  const invStr = invert ? " (INVERTED)" : "";
  return `${activeSources.join(" + ")}${opStr}${invStr}`;
}