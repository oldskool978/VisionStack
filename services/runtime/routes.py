from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import mimetypes
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from .env import (
    ARTIFACTS_DIR,
    DEFAULT_LAYOUT_ITEMS,
    DEFAULT_PRESET_FILENAME,
    DEFAULT_SPECULATIVE_COT_TOKENS,
    GenerationRequest,
    PollingLogFilter,
    SHADOWBOXER_DIR,
    STORAGE_ROOT,
    SUPPORTED_BOX_LABELS,
    SUPPORTED_EDIT_FORMATS,
    SUPPORTED_FRAMING_MODES,
    SUPPORTED_INPUT_ROLES,
    SUPPORTED_MASK_BLEND_MODES,
    SUPPORTED_MODELS,
    SUPPORTED_PROMPT_MODES,
    SUPPORTED_SOLVERS,
    _FORBIDDEN_SPA_SEGMENTS,
    _NO_CACHE_HEADERS,
    _get_preview_tokenizer,
    compute_webui_asset_version,
    get_active_engine_defaults,
    get_runtime_override,
    get_themes_root,
    has_custom_default_preset,
    is_standalone_mode,
    load_user_registry,
    render_versioned_index_html,
    resolve_site_root,
    slugify,
)
from .models import (
    AuthPayload,
    PresetUpdatePayload,
    ProjectImportPayload,
    ProjectSavePayload,
    PromoteImagePayload,
    PromptPreviewPayload,
    ReferenceUploadPayload,
    SynthesisPayload,
)
from .queue import ComputeQueue
from .security import (
    MAX_GLOBAL_QUEUE_DEPTH,
    MAX_USER_INFLIGHT_JOBS,
    check_rate_limit,
    create_session_token,
    generate_pow_challenge,
    resolve_client_ip,
    verify_pow_solution,
    verify_session_token,
)
from .vault import (
    _build_vsproj_package,
    _decode_data_url_bytes,
    _extract_and_restore_vsproj,
    _inspect_reference_file,
    _probe_available_models,
)

compute_queue = ComputeQueue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    compute_queue.start()
    users, src = load_user_registry()
    defaults = get_active_engine_defaults()
    asset_ver = compute_webui_asset_version()
    print("=" * 84)
    print("  VISIONSTACK UNIFIED SPATIO-VISUAL ENGINE // SERVICE ROUTER ACTIVE")
    print(
        f"  Mode            : {'STANDALONE DESKTOP' if is_standalone_mode() else 'HEADLESS / PROXY DAEMON'}"
    )
    print(f"  Site Root       : {resolve_site_root()} (Bundle Hash: {asset_ver})")
    print(f"  User Registry   : {src}")
    print(
        f"  Engine Preset   : {'Active default.json Override' if has_custom_default_preset() else 'Optimal Baseline (V03 Heun->IPNDM)'}"
    )
    print(
        f"  Default Solvers : Early={defaults['early_solver'].upper()} (CFG {defaults['early_cfg']}) -> Late={defaults['late_solver'].upper()} (CFG {defaults['late_cfg']}) @ t*={defaults['handoff_threshold']}"
    )
    print(f"  Accounts ({len(users)})    : {', '.join(users.keys())}")
    print("=" * 84)
    try:
        yield
    finally:
        compute_queue.stop()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path_lower = request.url.path.lower()
        is_job_poll = "/jobs/" in path_lower
        is_webui_code = (
            path_lower == "/"
            or path_lower.endswith((".html", ".htm", ".js", ".css", ".json"))
        ) and not is_job_poll

        if is_webui_code:
            raw_hdrs = request.scope.get("headers")
            if isinstance(raw_hdrs, list):
                request.scope["headers"] = [
                    (k, v)
                    for (k, v) in raw_hdrs
                    if k.lower() not in (b"if-none-match", b"if-modified-since")
                ]

        response: Response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"

        if is_webui_code:
            response.headers["Cache-Control"] = (
                "no-cache, no-store, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            for stale_hdr in ("etag", "last-modified"):
                if stale_hdr in response.headers:
                    del response.headers[stale_hdr]
        return response


app = FastAPI(
    title="VisionStack Unified Spatio-Visual Daemon",
    version="2.0.0",
    root_path="",
    lifespan=lifespan,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter()


@api_router.get("/auth/challenge")
async def get_challenge(request: Request):
    client_ip = resolve_client_ip(request)
    recent_ch = check_rate_limit(
        client_ip, scope="challenge", max_requests=45, window_seconds=60.0
    )
    return generate_pow_challenge(
        client_ip=client_ip,
        queue_depth=compute_queue.queue.qsize(),
        recent_ip_challenges=recent_ch,
    )


@api_router.post("/auth/login")
async def login(payload: AuthPayload, request: Request):
    client_ip = resolve_client_ip(request)
    check_rate_limit(
        client_ip, scope="login", max_requests=30, window_seconds=60.0
    )

    raw_input = payload.username.strip()
    input_slug = slugify(raw_input)
    users_map, _ = load_user_registry()

    matched_slug = None
    if input_slug in users_map:
        matched_slug = input_slug
    else:
        for key, meta in users_map.items():
            if slugify(key) == input_slug:
                matched_slug = key
                break
            display_name = str(meta.get("display_name", "")).strip()
            if display_name and (
                display_name.lower() == raw_input.lower()
                or slugify(display_name) == input_slug
            ):
                matched_slug = key
                break

    if not matched_slug:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid creator credentials.",
        )

    raw_slug = slugify(matched_slug)
    user_meta = users_map.get(matched_slug, {})
    token = create_session_token(raw_slug)

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    user_dir = storage_root / raw_slug
    (user_dir / "images").mkdir(parents=True, exist_ok=True)
    (user_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (user_dir / "projects").mkdir(parents=True, exist_ok=True)

    history_file = user_dir / "history.json"
    images: List[Dict[str, Any]] = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8-sig") as f:
                images = json.load(f).get("images", [])
                for img in images:
                    if img.get("image_url") or img.get("recipe"):
                        img.setdefault("status", "COMPLETED")
        except Exception:
            pass

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tokens_used_today = sum(
        1
        for img in images
        if img.get("created_at", "").startswith(today_utc)
        and not img.get("is_default", False)
    )
    inflight_count = compute_queue.count_user_inflight_jobs(raw_slug)
    daily_quota = int(user_meta.get("daily_quota", 25))
    perms = user_meta.get("custom_permissions", [])
    is_admin = "admin" in perms or "unlimited_quota" in perms
    tokens_remaining = (
        999
        if is_admin
        else max(0, daily_quota - (tokens_used_today + inflight_count))
    )

    registry_path = get_themes_root() / "registry.json"
    assigned_theme = user_meta.get("assigned_theme") or "obsidian_gold"
    theme_manifest = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8-sig") as f:
                theme_manifest = (
                    json.load(f).get("themes", {}).get(assigned_theme, {})
                )
        except Exception:
            pass

    return {
        "status": "authenticated",
        "token": token,
        "user": {
            "slug": raw_slug,
            "display_name": user_meta.get("display_name", payload.username),
            "daily_quota": daily_quota,
            "tokens_remaining": tokens_remaining,
            "assigned_theme": assigned_theme,
            "is_admin": is_admin,
        },
        "theme": {
            "theme_id": assigned_theme,
            "manifest": theme_manifest,
        },
        "images": images,
        "engine_defaults": get_active_engine_defaults(),
        "default_layout_items": DEFAULT_LAYOUT_ITEMS,
        "available_models": _probe_available_models(
            compute_queue.pipeline._engine
        ),
    }


@api_router.get("/themes/registry")
async def get_themes():
    registry_path = get_themes_root() / "registry.json"
    if not registry_path.exists():
        return {"revolving_pool": [], "themes": {}}
    try:
        with open(registry_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {"revolving_pool": [], "themes": {}}


@api_router.get("/health")
async def health():
    defaults = get_active_engine_defaults()
    hw = compute_queue.pipeline.poll_hardware_telemetry()
    return {
        "status": "online",
        "engine_device": compute_queue.pipeline.device_str,
        "webui_asset_version": compute_webui_asset_version(),
        "active_job": (
            compute_queue.active_job.job_id if compute_queue.active_job else None
        ),
        "queue_depth": compute_queue.queue.qsize(),
        "engine_preset_active": has_custom_default_preset(),
        "baseline_parameters": dict(defaults),
        "supported_models": SUPPORTED_MODELS,
        "available_models": _probe_available_models(
            compute_queue.pipeline._engine
        ),
        "supported_solvers": SUPPORTED_SOLVERS,
        "supported_prompt_modes": SUPPORTED_PROMPT_MODES,
        "supported_box_labels": SUPPORTED_BOX_LABELS,
        "supported_edit_formats": SUPPORTED_EDIT_FORMATS,
        "supported_framing_modes": SUPPORTED_FRAMING_MODES,
        "supported_input_roles": SUPPORTED_INPUT_ROLES,
        "supported_mask_blend_modes": SUPPORTED_MASK_BLEND_MODES,
        "default_speculative_cot_tokens": DEFAULT_SPECULATIVE_COT_TOKENS,
        "hardware": hw,
    }


@api_router.post("/prompt/preview")
async def preview_compiled_prompt(payload: PromptPreviewPayload):
    req = GenerationRequest(
        prompt=payload.prompt,
        prompt_mode=payload.prompt_mode,
        layout_summary=payload.layout_summary,
        layout_items=payload.layout_items,
        layout_coords_are_pixels=payload.layout_coords_are_pixels,
        width=payload.width,
        height=payload.height,
        system_prompt=payload.system_prompt,
        thinking_trace=payload.thinking_trace,
        enable_thinking=payload.enable_thinking,
        add_vision_id=payload.add_vision_id,
        max_sequence_length=payload.max_sequence_length,
        max_vit_patches=payload.max_vit_patches,
        edit_format=payload.edit_format,
        framing_mode=payload.framing_mode,
        input_role=payload.input_role,
        box_attn_isolation=payload.box_attn_isolation,
        composite_box_images_to_init=payload.composite_box_images_to_init,
        uncond_uses_references=payload.uncond_uses_references,
        reference_zero_timestep=payload.reference_zero_timestep,
        reference_isolated_kv=payload.reference_isolated_kv,
        reference_image_paths=payload.reference_image_paths,
        control_image_path=payload.control_image_path,
    )
    effective_prompt = req.compile_effective_prompt()
    resolved_trace, spec_cot_tokens = req.resolve_thinking_trace()
    ref_specs, _ = req.resolve_visual_references()

    max_patches = max(64, int(req.max_vit_patches or 2304))
    default_llm_per_ref = max(16, max_patches // 4)
    patch_counts: List[int] = [
        default_llm_per_ref
        for spec in ref_specs
        if bool(spec.get("emits_vit"))
    ]
    vision_tokens = sum(patch_counts)
    max_seq = int(req.max_sequence_length or 16384)

    tok = _get_preview_tokenizer()
    if tok is not None:
        prompt_tokens = len(
            tok.encode(effective_prompt, allowed_special="all")
        )
        if spec_cot_tokens > 0:
            trace_tokens = int(spec_cot_tokens)
        elif resolved_trace:
            trace_tokens = len(
                tok.encode(resolved_trace, allowed_special="all")
            )
        else:
            trace_tokens = 0

        chat_ids = tok.encode_chat(
            prompt=effective_prompt,
            enable_thinking=bool(
                True if spec_cot_tokens > 0 else req.enable_thinking
            ),
            max_sequence_length=max_seq,
            image_patch_counts=patch_counts if patch_counts else None,
            system_prompt=req.system_prompt,
            thinking_trace=resolved_trace,
            add_vision_id=bool(
                req.add_vision_id if req.add_vision_id is not None else True
            ),
        )
        total_context_tokens = len(chat_ids) + int(spec_cot_tokens)
        viztoken_active = bool(tok.viztoken_active)
    else:
        prompt_tokens = max(1, len(effective_prompt.encode("utf-8")) // 4)
        trace_tokens = (
            int(spec_cot_tokens)
            if spec_cot_tokens > 0
            else (len(resolved_trace.encode("utf-8")) // 4 if resolved_trace else 0)
        )
        total_context_tokens = prompt_tokens + trace_tokens + vision_tokens + 16
        viztoken_active = False

    text_budget_used = prompt_tokens + trace_tokens
    horizon_utilization_pct = round(
        min(100.0, (float(text_budget_used) / float(max(1, max_seq))) * 100.0),
        2,
    )

    return {
        "effective_prompt": effective_prompt,
        "resolved_thinking_trace": resolved_trace,
        "speculative_cot_tokens": spec_cot_tokens,
        "prompt_tokens": prompt_tokens,
        "trace_tokens": trace_tokens,
        "vision_tokens": vision_tokens,
        "total_context_tokens": total_context_tokens,
        "max_sequence_length": max_seq,
        "horizon_utilization_pct": horizon_utilization_pct,
        "viztoken_active": viztoken_active,
        "effective_reference_zero_timestep": req.effective_reference_zero_timestep,
        "effective_reference_isolated_kv": req.effective_reference_isolated_kv,
        "effective_uncond_uses_references": req.effective_uncond_uses_references,
    }


@api_router.post("/presets/default")
async def update_default_preset(
    payload: PresetUpdatePayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    users_map, _ = load_user_registry()
    user_meta = users_map.get(slug, {})
    perms = user_meta.get("custom_permissions", [])
    if "admin" not in perms and "unlimited_quota" not in perms:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permissions required to update authoritative default.json.",
        )

    merged = dict(get_active_engine_defaults())
    merged.update(payload.engine_defaults)
    merged.setdefault("prompt", "Authoritative ShadowBoxer Studio Preset")
    req = GenerationRequest.from_dict(merged)
    sb_dir = Path(get_runtime_override("SHADOWBOXER_DIR", SHADOWBOXER_DIR))
    target_path = sb_dir / DEFAULT_PRESET_FILENAME
    req.save_preset(target_path)
    return {
        "status": "updated",
        "preset_path": str(target_path),
        "engine_preset_active": has_custom_default_preset(),
        "baseline_parameters": get_active_engine_defaults(),
    }


@api_router.post("/references/upload")
async def upload_reference_asset(
    payload: ReferenceUploadPayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    try:
        binary = _decode_data_url_bytes(payload.data_url)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image payload.",
        )
    if len(binary) > 40 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Reference asset exceeds 40 MB limit.",
        )

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    uploads_dir = storage_root / slug / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ref_id = f"ref_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"

    if len(binary) >= 16 and binary[:16] == b"VIZJ_CONTAINER01":
        dest_file = uploads_dir / f"{ref_id}.vizj"
        dest_file.write_bytes(binary)
        meta = _inspect_reference_file(dest_file, slug)
        meta["filename"] = payload.filename
        meta["role"] = payload.role
        return meta

    try:
        with Image.open(io.BytesIO(binary)) as img:
            rgba = img.convert("RGBA")
            dest_file = uploads_dir / f"{ref_id}.png"
            rgba.save(dest_file, format="PNG")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported or corrupted image file: {e}",
        )

    meta = _inspect_reference_file(dest_file, slug)
    meta["filename"] = payload.filename
    meta["role"] = payload.role
    return meta


@api_router.post("/references/from-image/{image_id}")
async def promote_gallery_image_to_reference(
    image_id: str,
    payload: Optional[PromoteImagePayload] = None,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    clean_id = Path(image_id).name
    var_idx = int(payload.variant_index) if payload is not None else 1
    role = str(payload.role) if payload is not None else "photo_ref"

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))
    user_images_dir = (storage_root / slug / "images").resolve()
    if var_idx <= 1:
        candidates = [
            user_images_dir / f"{clean_id}_master.png",
            user_images_dir / f"{clean_id}.png",
            art_dir / f"{clean_id}.png",
        ]
    else:
        candidates = [
            user_images_dir / f"{clean_id}_master_{var_idx}.png",
            art_dir / f"{clean_id}_{var_idx}.png",
        ]

    src_file: Optional[Path] = None
    for c in candidates:
        if c.is_file():
            src_file = c
            break

    if src_file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Gallery image '{clean_id}' not found for vault promotion.",
        )

    raw_bytes = src_file.read_bytes()
    target_sha = hashlib.sha256(raw_bytes).hexdigest()

    uploads_dir = (storage_root / slug / "uploads").resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    for existing_f in uploads_dir.iterdir():
        if existing_f.is_file() and existing_f.suffix.lower() == ".png":
            try:
                if (
                    hashlib.sha256(existing_f.read_bytes()).hexdigest()
                    == target_sha
                ):
                    meta = _inspect_reference_file(existing_f, slug)
                    meta["role"] = role
                    meta["deduplicated"] = True
                    return meta
            except Exception:
                pass

    dest_name = f"ref_{clean_id}_{target_sha[:6]}.png"
    dest_file = uploads_dir / dest_name
    dest_file.write_bytes(raw_bytes)
    meta = _inspect_reference_file(dest_file, slug)
    meta["role"] = role
    meta["deduplicated"] = False
    return meta


@api_router.post("/references/stage-defaults")
async def stage_default_test_assets(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    try:
        from ShadowBoxer.harness import stage_test_input_files
    except Exception:
        from harness import stage_test_input_files

    sb_dir = Path(get_runtime_override("SHADOWBOXER_DIR", SHADOWBOXER_DIR))
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    test_dir = sb_dir / "artifacts" / "test_inputs"
    staged_map = stage_test_input_files(test_dir)

    uploads_dir = storage_root / slug / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    staged_names: List[str] = []
    seen_files = set()
    for src_path in staged_map.values():
        p = Path(src_path)
        if p.is_file() and p.name not in seen_files:
            seen_files.add(p.name)
            dst_path = uploads_dir / p.name
            shutil.copy2(p, dst_path)
            staged_names.append(dst_path.name)

    return {
        "status": "staged",
        "staged_files": staged_names,
    }


@api_router.get("/references")
async def list_reference_assets(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    uploads_dir = storage_root / slug / "uploads"
    if not uploads_dir.exists():
        return {"references": []}
    items = []
    for f in sorted(
        uploads_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if f.is_file() and f.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".vizj",
        ):
            items.append(_inspect_reference_file(f, slug))
    return {"references": items}


@api_router.get("/references/stream/{user_slug}/{filename}")
async def stream_reference_asset(user_slug: str, filename: str):
    safe_slug = slugify(user_slug)
    safe_name = Path(filename).name
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    uploads_dir = (storage_root / safe_slug / "uploads").resolve()
    target = (uploads_dir / safe_name).resolve()
    if not target.is_file() or not target.is_relative_to(uploads_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reference asset not found.",
        )
    ext = target.suffix.lower()
    if ext == ".vizj":
        media = "application/octet-stream"
    else:
        media = mimetypes.guess_type(str(target))[0] or "image/png"
    return FileResponse(
        str(target),
        media_type=media,
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "no-cache",
        },
    )


@api_router.delete("/references/{ref_id}")
async def delete_reference_asset(
    ref_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    safe_name = Path(ref_id).name
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    uploads_dir = (storage_root / slug / "uploads").resolve()
    target = (uploads_dir / safe_name).resolve()
    if target.is_file() and target.is_relative_to(uploads_dir):
        target.unlink(missing_ok=True)
    return {"status": "deleted", "ref_id": safe_name}


@api_router.post("/projects/save")
async def save_studio_project(
    payload: ProjectSavePayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    raw_id = (payload.project_id or "").strip()
    proj_id = (
        slugify(raw_id)
        if raw_id
        else f"proj_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    )

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / slug / "projects").resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes, manifest_data, preview_bytes = _build_vsproj_package(
        slug, payload, proj_id
    )
    vsproj_path = projects_dir / f"{proj_id}.vsproj"
    vsproj_path.write_bytes(zip_bytes)

    preview_url: Optional[str] = None
    if preview_bytes is not None:
        prev_file = projects_dir / f"{proj_id}_preview.png"
        prev_file.write_bytes(preview_bytes)
        preview_url = f"api/v1/projects/stream/{slug}/{proj_id}_preview.png"

    summary = {
        "project_id": proj_id,
        "title": manifest_data["title"],
        "created_at": manifest_data["created_at"],
        "updated_at": manifest_data["updated_at"],
        "width": manifest_data["width"],
        "height": manifest_data["height"],
        "model_variant": manifest_data["model_variant"],
        "prompt_mode": manifest_data["prompt_mode"],
        "asset_count": manifest_data["asset_count"],
        "size_bytes": len(zip_bytes),
        "preview_url": preview_url,
        "download_url": f"api/v1/projects/{proj_id}/download",
    }
    meta_file = projects_dir / f"{proj_id}.json"
    meta_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"status": "saved", "project": summary}


@api_router.get("/projects")
async def list_studio_projects(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / slug / "projects").resolve()
    if not projects_dir.is_dir():
        return {"projects": []}

    items: List[Dict[str, Any]] = []
    for vs_file in sorted(
        projects_dir.glob("*.vsproj"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        proj_id = vs_file.stem
        meta_file = projects_dir / f"{proj_id}.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                meta["size_bytes"] = vs_file.stat().st_size
                meta.setdefault(
                    "download_url", f"api/v1/projects/{proj_id}/download"
                )
                items.append(meta)
                continue
            except Exception:
                pass
        st = vs_file.stat()
        items.append(
            {
                "project_id": proj_id,
                "title": proj_id,
                "created_at": datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc
                ).isoformat(),
                "updated_at": datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc
                ).isoformat(),
                "width": 1024,
                "height": 1024,
                "model_variant": "sft",
                "prompt_mode": "natural",
                "asset_count": 0,
                "size_bytes": st.st_size,
                "preview_url": None,
                "download_url": f"api/v1/projects/{proj_id}/download",
            }
        )
    return {"projects": items}


@api_router.post("/projects/{project_id}/load")
async def load_studio_project(
    project_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    safe_id = slugify(Path(project_id).stem)
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / slug / "projects").resolve()
    vsproj_file = (projects_dir / f"{safe_id}.vsproj").resolve()
    if not vsproj_file.is_file() or not vsproj_file.is_relative_to(projects_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{safe_id}' not found in user vault.",
        )
    return _extract_and_restore_vsproj(
        slug,
        vsproj_file.read_bytes(),
        persist_to_projects_vault=False,
    )


@api_router.post("/projects/import")
async def import_studio_project(
    payload: ProjectImportPayload,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    try:
        raw_zip = _decode_data_url_bytes(payload.data_url)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 .vsproj payload.",
        )
    return _extract_and_restore_vsproj(
        slug, raw_zip, persist_to_projects_vault=True
    )


@api_router.get("/projects/{project_id}/download")
async def download_studio_project(
    project_id: str,
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None,
):
    auth_token = token or (
        authorization.replace("Bearer ", "").strip() if authorization else None
    )
    slug = verify_session_token(auth_token)
    safe_id = slugify(Path(project_id).stem)
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / slug / "projects").resolve()
    vsproj_file = (projects_dir / f"{safe_id}.vsproj").resolve()
    if not vsproj_file.is_file() or not vsproj_file.is_relative_to(projects_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{safe_id}' not found.",
        )
    return FileResponse(
        str(vsproj_file),
        media_type="application/octet-stream",
        filename=f"{safe_id}.vsproj",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "no-cache",
        },
    )


@api_router.get("/projects/stream/{user_slug}/{filename}")
async def stream_project_preview(user_slug: str, filename: str):
    safe_slug = slugify(user_slug)
    safe_name = Path(filename).name
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / safe_slug / "projects").resolve()
    target = (projects_dir / safe_name).resolve()
    if not target.is_file() or not target.is_relative_to(projects_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project preview not found.",
        )
    return FileResponse(
        str(target),
        media_type="image/png",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cache-Control": "no-cache",
        },
    )


@api_router.delete("/projects/{project_id}")
async def delete_studio_project(
    project_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    safe_id = slugify(Path(project_id).stem)
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    projects_dir = (storage_root / slug / "projects").resolve()
    for suffix in (
        f"{safe_id}.vsproj",
        f"{safe_id}.json",
        f"{safe_id}_preview.png",
    ):
        cand = (projects_dir / suffix).resolve()
        if cand.is_file() and cand.is_relative_to(projects_dir):
            cand.unlink(missing_ok=True)
    return {"status": "deleted", "project_id": safe_id}


@api_router.post("/synthesize")
async def synthesize(
    payload: SynthesisPayload,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)
    client_ip = resolve_client_ip(request)

    if not payload.prompt.strip() and not payload.layout_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt or spatial layout boxes required.",
        )

    max_q_depth = int(
        get_runtime_override("MAX_GLOBAL_QUEUE_DEPTH", MAX_GLOBAL_QUEUE_DEPTH)
    )
    if compute_queue.queue.qsize() >= max_q_depth:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Compute queue is currently at maximum capacity. Please retry shortly.",
        )

    users_map, _ = load_user_registry()
    user_meta = users_map.get(slug, {})
    perms = user_meta.get("custom_permissions", [])
    is_unlimited = "admin" in perms or "unlimited_quota" in perms

    inflight_count = compute_queue.count_user_inflight_jobs(slug)
    if not is_unlimited:
        max_user_inflight = int(
            get_runtime_override(
                "MAX_USER_INFLIGHT_JOBS", MAX_USER_INFLIGHT_JOBS
            )
        )
        if inflight_count >= max_user_inflight:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Maximum concurrent in-flight jobs ({max_user_inflight}) reached.",
            )

        storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
        history_file = storage_root / slug / "history.json"
        tokens_used_today = 0
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8-sig") as f:
                    raw_images = json.load(f).get("images", [])
                    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    tokens_used_today = sum(
                        1
                        for img in raw_images
                        if img.get("created_at", "").startswith(today_utc)
                        and not img.get("is_default", False)
                    )
            except Exception:
                pass

        daily_quota = int(user_meta.get("daily_quota", 25))
        if (tokens_used_today + inflight_count) >= daily_quota:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily synthesis token quota reached. Quotas reset at 00:00 UTC.",
            )

    if not verify_pow_solution(
        payload.pow.challenge,
        payload.pow.signature,
        payload.pow.solution_nonce,
        client_ip=client_ip,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bot defense proof-of-work verification failed.",
        )

    job = compute_queue.enqueue(slug, payload.model_dump())
    return compute_queue.get_status(job.job_id)


@api_router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    verify_session_token(token)

    info = compute_queue.get_status(job_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master job record not found.",
        )

    etag = f'W/"{job_id}-{info.get("status")}-{info.get("progress_pct")}-{info.get("users_ahead")}"'
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(
            status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag}
        )

    return JSONResponse(content=info, headers={"ETag": etag})


@api_router.get("/jobs/{job_id}/stream")
async def stream_job_status(
    job_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
    token: Optional[str] = None,
):
    auth_token = token or (
        authorization.replace("Bearer ", "").strip() if authorization else None
    )
    verify_session_token(auth_token)

    job = compute_queue.jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master job record not found.",
        )

    async def event_generator():
        last_state_str = None
        event = compute_queue.get_job_event(job_id)
        while True:
            if await request.is_disconnected():
                break
            current_status = compute_queue.get_status(job_id)
            if not current_status:
                break
            state_str = json.dumps(current_status)
            if state_str != last_state_str:
                last_state_str = state_str
                yield f"data: {state_str}\n\n"
            if current_status.get("status") in ("COMPLETED", "FAILED"):
                break
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                event.clear()
            except asyncio.TimeoutError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@api_router.get("/images/{job_id}")
async def get_image_direct(job_id: str):
    job = compute_queue.jobs.get(job_id)
    if (
        not job
        or job.status != "COMPLETED"
        or not job.output_file
        or not job.output_file.exists()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Master image artifact not ready.",
        )

    return FileResponse(
        str(job.output_file),
        media_type="image/png",
        headers={
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@api_router.get("/images/stream/{user_slug}/{filename}")
async def get_image_stream_user(user_slug: str, filename: str):
    safe_slug = slugify(user_slug)
    safe_name = Path(filename).name
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))
    target_file = storage_root / safe_slug / "images" / safe_name

    media_type = (
        "application/octet-stream"
        if safe_name.lower().endswith(".vizj")
        else "image/png"
    )
    headers = {
        "Cross-Origin-Resource-Policy": "cross-origin",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
    }

    if target_file.exists() and target_file.is_file():
        return FileResponse(
            str(target_file), media_type=media_type, headers=headers
        )

    clean_stem = (
        safe_name.replace("_master.png", "")
        .replace(".png", "")
        .replace(".vizj", "")
    )
    ext = ".vizj" if safe_name.lower().endswith(".vizj") else ".png"
    artifact_file = art_dir / f"{clean_stem}{ext}"
    if artifact_file.exists() and artifact_file.is_file():
        return FileResponse(
            str(artifact_file), media_type=media_type, headers=headers
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Master image artifact unavailable.",
    )


@api_router.delete("/images/{image_id}")
async def delete_user_image(
    image_id: str,
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "").strip() if authorization else None
    slug = verify_session_token(token)

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    user_dir = storage_root / slug
    history_file = user_dir / "history.json"
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8-sig") as f:
                history_data = json.load(f)
            original_images = history_data.get("images", [])
            history_data["images"] = [
                img
                for img in original_images
                if img.get("image_id") != image_id
            ]
            temp_hist = history_file.with_suffix(".tmp")
            with open(temp_hist, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            temp_hist.replace(history_file)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed updating vault storage: {e}"
            )

    images_dir = user_dir / "images"
    for suffix in (f"{image_id}_master.png", f"{image_id}.vizj"):
        f_path = images_dir / suffix
        if f_path.exists():
            f_path.unlink(missing_ok=True)
    if images_dir.is_dir():
        for extra_p in images_dir.glob(f"{image_id}_master_*.png"):
            extra_p.unlink(missing_ok=True)

    return {"status": "deleted", "image_id": image_id}


for _prefix in (
    "/api/v1",
    "/v1",
    "/VisionStack/api/v1",
    "/visionstack/api/v1",
    "/ShadowBoxer/api/v1",
    "/shadowboxer/api/v1",
):
    app.include_router(api_router, prefix=_prefix)


def _serve_versioned_index_response() -> Response:
    html_bytes, asset_ver = render_versioned_index_html()
    if html_bytes:
        hdrs = dict(_NO_CACHE_HEADERS)
        hdrs["X-VisionStack-Asset-Version"] = asset_ver
        return Response(
            content=html_bytes,
            media_type="text/html; charset=utf-8",
            headers=hdrs,
        )
    return JSONResponse(
        content={
            "service": "VisionStack Unified Spatio-Visual Daemon",
            "status": "online",
        }
    )


def mount_static_and_spa() -> None:
    site_dir = resolve_site_root()
    if site_dir.exists() and (site_dir / "index.html").exists():
        for sub in ("assets", "themes", "public", "wasm"):
            sub_dir = site_dir / sub
            if sub_dir.exists():
                app.mount(
                    f"/{sub}", StaticFiles(directory=str(sub_dir)), name=sub
                )

    @app.get("/")
    async def serve_index():
        return _serve_versioned_index_response()

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        norm_path = full_path.replace("\\", "/").strip("/")
        path_lower = norm_path.lower()
        parts_lower = [p for p in path_lower.split("/") if p]
        if any(
            p in _FORBIDDEN_SPA_SEGMENTS or p.startswith(".")
            for p in parts_lower
        ):
            raise HTTPException(status_code=403, detail="Forbidden")

        if (
            path_lower.startswith("auth/")
            or path_lower.startswith("synthesize")
            or path_lower.startswith("prompt/")
            or path_lower.startswith("jobs/")
            or path_lower.startswith("images/")
            or path_lower.startswith("references")
            or path_lower.startswith("projects")
            or path_lower.startswith("presets/")
            or path_lower.startswith("health")
            or path_lower.startswith("themes/")
            or path_lower.startswith("api/")
            or path_lower.startswith("v1/")
            or path_lower.startswith("visionstack/")
            or path_lower.startswith("shadowboxer/")
        ):
            return JSONResponse(
                status_code=404,
                content={"detail": f"Route not found: /{full_path}"},
            )

        if path_lower in ("", "index.html", "index.htm"):
            return _serve_versioned_index_response()

        current_site = resolve_site_root().resolve()
        try:
            target = (current_site / norm_path).resolve()
            if not target.is_relative_to(current_site):
                raise HTTPException(status_code=403, detail="Forbidden")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Not Found")

        if target.exists() and target.is_file():
            if any(
                part.lower() in _FORBIDDEN_SPA_SEGMENTS
                for part in target.parts
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            return FileResponse(str(target), headers=dict(_NO_CACHE_HEADERS))

        suffix = Path(norm_path).suffix.lower()
        if suffix and suffix not in (".html", ".htm"):
            raise HTTPException(status_code=404, detail="Not Found")

        idx_fallback = current_site / "index.html"
        if idx_fallback.is_file():
            return _serve_versioned_index_response()
        raise HTTPException(status_code=404, detail="Not Found")


mount_static_and_spa()


def run_server(host: str, port: int) -> None:
    uv_access = logging.getLogger("uvicorn.access")
    uv_access.addFilter(PollingLogFilter())
    uvicorn.run(app, host=host, port=port, log_level="info")


def launch_standalone(host: str, port: int) -> None:
    asset_ver = compute_webui_asset_version()
    target_url = f"http://{host}:{port}/?_v={asset_ver}"
    try:
        import webview
    except ImportError:
        import webbrowser

        threading.Thread(
            target=run_server, args=(host, port), daemon=True
        ).start()
        time.sleep(1.0)
        webbrowser.open(target_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)

    server_thread = threading.Thread(
        target=run_server, args=(host, port), daemon=True
    )
    server_thread.start()
    time.sleep(1.0)
    webview.create_window(
        title="VisionStack — Sovereign Spatio-Visual Studio",
        url=target_url,
        width=1440,
        height=920,
        min_size=(1080, 720),
        background_color="#07090e",
    )
    webview.start(debug=False)
    sys.exit(0)


__all__ = [
    "compute_queue",
    "lifespan",
    "SecurityHeadersMiddleware",
    "app",
    "api_router",
    "mount_static_and_spa",
    "run_server",
    "launch_standalone",
]