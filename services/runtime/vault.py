# services/runtime/vault.py
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import zipfile

from fastapi import HTTPException, status
from PIL import Image

from .env import (
    ARTIFACTS_DIR,
    SHADOWBOXER_DIR,
    STORAGE_ROOT,
    SUPPORTED_MODELS,
    get_runtime_override,
    slugify,
)
from .models import ProjectSavePayload


def _probe_vizj_dimensions(binary: bytes) -> Tuple[int, int]:
    if len(binary) >= 48 and binary[:16] == b"VIZJ_CONTAINER01":
        try:
            orig_w, orig_h, vae_w, vae_h = struct.unpack("<IIII", binary[32:48])
            if vae_w > 0 and vae_h > 0:
                return int(vae_w), int(vae_h)
            if orig_w > 0 and orig_h > 0:
                return int(orig_w), int(orig_h)
        except Exception:
            pass
    return 1024, 1024


def _inspect_reference_file(file_path: Path, user_slug: str) -> Dict[str, Any]:
    st = file_path.stat()
    is_vizj = file_path.suffix.lower() == ".vizj"
    width, height = 1024, 1024
    has_alpha = False
    try:
        if is_vizj:
            with open(file_path, "rb") as f:
                hdr = f.read(64)
            width, height = _probe_vizj_dimensions(hdr)
        else:
            with Image.open(file_path) as img:
                width, height = int(img.width), int(img.height)
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    alpha_band = img.convert("RGBA").getchannel("A")
                    extrema = alpha_band.getextrema()
                    if extrema and extrema[0] < 250:
                        has_alpha = True
    except Exception:
        pass
    aspect = round(float(width) / float(max(1, height)), 4)
    return {
        "ref_id": file_path.name,
        "filename": file_path.name,
        "width": width,
        "height": height,
        "aspect_ratio": aspect,
        "has_alpha": has_alpha,
        "is_vizj": is_vizj,
        "size_bytes": st.st_size,
        "created_at": datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc
        ).isoformat(),
        "url": f"api/v1/references/stream/{user_slug}/{file_path.name}",
    }


def _decode_data_url_bytes(raw_data: str) -> bytes:
    clean = raw_data.strip()
    if "," in clean and clean.startswith("data:"):
        clean = clean.split(",", 1)[1]
    return base64.b64decode(clean)


def _resolve_user_upload_path(
    user_slug: str, raw_ref: Optional[str]
) -> Optional[str]:
    if not raw_ref or not str(raw_ref).strip():
        return None
    ref_str = str(raw_ref).strip()
    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    sb_dir = Path(get_runtime_override("SHADOWBOXER_DIR", SHADOWBOXER_DIR))
    art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))

    user_uploads_dir = (storage_root / user_slug / "uploads").resolve()
    user_images_dir = (storage_root / user_slug / "images").resolve()
    test_inputs_dir = (sb_dir / "artifacts" / "test_inputs").resolve()

    cand_name = Path(ref_str).name
    for base_dir in (user_uploads_dir, user_images_dir, test_inputs_dir):
        cand_path = (base_dir / cand_name).resolve()
        if cand_path.is_file() and cand_path.is_relative_to(base_dir):
            return str(cand_path)

    direct = Path(ref_str)
    if direct.is_file():
        resolved = direct.resolve()
        for allowed_root in (
            user_uploads_dir,
            user_images_dir,
            test_inputs_dir,
            art_dir.resolve(),
        ):
            if resolved.is_relative_to(allowed_root):
                return str(resolved)
    return None


def _collect_draft_asset_names(working_draft: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    seen = set()

    def _add(raw_val: Any) -> None:
        if isinstance(raw_val, str) and raw_val.strip():
            name = Path(raw_val.strip()).name
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    for key in ("init_image_path", "inpaint_mask_path", "control_image_path"):
        _add(working_draft.get(key))
    for r_item in working_draft.get("reference_image_paths") or []:
        _add(r_item)
    for box in working_draft.get("layout_items") or []:
        if isinstance(box, dict):
            _add(box.get("image_path") or box.get("image") or box.get("vizj_path"))
    return names


def _build_vsproj_package(
    user_slug: str,
    payload: ProjectSavePayload,
    project_id: str,
) -> Tuple[bytes, Dict[str, Any], Optional[bytes]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    draft = dict(payload.working_draft or {})
    title = (
        payload.title.strip()
        or str(draft.get("title") or "").strip()
        or "Untitled Studio Project"
    )
    draft["title"] = title

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))
    art_dir = Path(get_runtime_override("ARTIFACTS_DIR", ARTIFACTS_DIR))

    uploads_dir = (storage_root / user_slug / "uploads").resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ref_names = _collect_draft_asset_names(draft)
    if payload.include_all_vault_refs and uploads_dir.is_dir():
        for f in sorted(uploads_dir.iterdir()):
            if (
                f.is_file()
                and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".vizj")
                and f.name not in ref_names
            ):
                ref_names.append(f.name)

    asset_manifest: List[Dict[str, Any]] = []
    bundled_files: List[Tuple[str, bytes]] = []
    for name in ref_names:
        resolved_str = _resolve_user_upload_path(user_slug, name)
        if not resolved_str:
            continue
        src_p = Path(resolved_str)
        if not src_p.is_file():
            continue
        raw_bytes = src_p.read_bytes()
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        meta = _inspect_reference_file(src_p, user_slug)
        asset_manifest.append(
            {
                "ref_id": src_p.name,
                "sha256": sha256_hex,
                "size_bytes": len(raw_bytes),
                "width": meta["width"],
                "height": meta["height"],
                "has_alpha": meta["has_alpha"],
                "is_vizj": meta["is_vizj"],
            }
        )
        bundled_files.append((src_p.name, raw_bytes))

    inpaint_canvas_bytes: Optional[bytes] = None
    if payload.inpaint_canvas_data_url and payload.inpaint_canvas_data_url.strip():
        try:
            inpaint_canvas_bytes = _decode_data_url_bytes(
                payload.inpaint_canvas_data_url
            )
        except Exception:
            inpaint_canvas_bytes = None

    preview_bytes: Optional[bytes] = None
    active_img_id = payload.active_image_id or draft.get("active_image_id")
    if isinstance(active_img_id, str) and active_img_id.strip():
        clean_id = Path(active_img_id.strip()).name
        cand_user_img = storage_root / user_slug / "images" / f"{clean_id}_master.png"
        cand_art_img = art_dir / f"{clean_id}.png"
        if cand_user_img.is_file():
            preview_bytes = cand_user_img.read_bytes()
        elif cand_art_img.is_file():
            preview_bytes = cand_art_img.read_bytes()

    manifest_data: Dict[str, Any] = {
        "package_format": "VISIONSTACK_PROJECT_V2",
        "project_id": project_id,
        "title": title,
        "user_slug": user_slug,
        "created_at": now_iso,
        "updated_at": now_iso,
        "width": int(draft.get("width") or 1024),
        "height": int(draft.get("height") or 1024),
        "model_variant": str(draft.get("model_variant") or "sft"),
        "prompt_mode": str(draft.get("prompt_mode") or "natural"),
        "active_image_id": active_img_id,
        "asset_count": len(asset_manifest),
        "assets": asset_manifest,
        "has_inpaint_canvas": inpaint_canvas_bytes is not None,
        "has_master_preview": preview_bytes is not None,
        "working_draft": draft,
        "telemetry": payload.telemetry,
    }

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(
        zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        zf.writestr(
            "project_manifest.json",
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
        )
        for arc_name, data_b in bundled_files:
            zf.writestr(f"assets/{arc_name}", data_b)
        if inpaint_canvas_bytes is not None:
            zf.writestr("workspace/inpaint_canvas.png", inpaint_canvas_bytes)
        if preview_bytes is not None:
            zf.writestr("workspace/master_preview.png", preview_bytes)

    return zip_buf.getvalue(), manifest_data, preview_bytes


def _extract_and_restore_vsproj(
    user_slug: str,
    raw_zip_bytes: bytes,
    persist_to_projects_vault: bool = True,
) -> Dict[str, Any]:
    if len(raw_zip_bytes) > 250 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Project archive exceeds 250 MB limit.",
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_zip_bytes), mode="r")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted .vsproj archive: {e}",
        )

    storage_root = Path(get_runtime_override("STORAGE_ROOT", STORAGE_ROOT))

    with zf:
        names = set(zf.namelist())
        if "project_manifest.json" not in names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing project_manifest.json in .vsproj archive.",
            )
        try:
            manifest = json.loads(
                zf.read("project_manifest.json").decode("utf-8")
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed project_manifest.json: {e}",
            )

        uploads_dir = (storage_root / user_slug / "uploads").resolve()
        uploads_dir.mkdir(parents=True, exist_ok=True)

        existing_sha_map: Dict[str, str] = {}
        for f in uploads_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".vizj",
            ):
                try:
                    existing_sha_map[
                        hashlib.sha256(f.read_bytes()).hexdigest()
                    ] = f.name
                except Exception:
                    pass

        id_remap: Dict[str, str] = {}
        restored_assets: List[str] = []

        for member in zf.namelist():
            if not member.startswith("assets/") or member.endswith("/"):
                continue
            safe_name = Path(member).name
            if not safe_name or safe_name.startswith("."):
                continue
            ext = Path(safe_name).suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp", ".vizj"):
                continue
            asset_bytes = zf.read(member)
            if len(asset_bytes) > 40 * 1024 * 1024:
                continue
            asset_sha = hashlib.sha256(asset_bytes).hexdigest()

            cand_dest = (uploads_dir / safe_name).resolve()
            if not cand_dest.is_relative_to(uploads_dir):
                continue

            if cand_dest.is_file():
                cur_sha = hashlib.sha256(cand_dest.read_bytes()).hexdigest()
                if cur_sha == asset_sha:
                    id_remap[safe_name] = safe_name
                    restored_assets.append(safe_name)
                    continue
                if asset_sha in existing_sha_map:
                    mapped_name = existing_sha_map[asset_sha]
                    id_remap[safe_name] = mapped_name
                    restored_assets.append(mapped_name)
                    continue
                new_name = f"ref_{int(time.time() * 1000)}_{asset_sha[:6]}{ext}"
                new_dest = uploads_dir / new_name
                new_dest.write_bytes(asset_bytes)
                existing_sha_map[asset_sha] = new_name
                id_remap[safe_name] = new_name
                restored_assets.append(new_name)
            else:
                cand_dest.write_bytes(asset_bytes)
                existing_sha_map[asset_sha] = safe_name
                id_remap[safe_name] = safe_name
                restored_assets.append(safe_name)

        draft = dict(manifest.get("working_draft") or {})
        for single_key in (
            "init_image_path",
            "inpaint_mask_path",
            "control_image_path",
        ):
            val = draft.get(single_key)
            if isinstance(val, str) and val.strip():
                b_name = Path(val.strip()).name
                draft[single_key] = id_remap.get(b_name, b_name)

        raw_refs = draft.get("reference_image_paths") or []
        if isinstance(raw_refs, list):
            draft["reference_image_paths"] = [
                id_remap.get(Path(str(r)).name, Path(str(r)).name)
                for r in raw_refs
                if isinstance(r, str) and str(r).strip()
            ]

        raw_boxes = draft.get("layout_items") or []
        if isinstance(raw_boxes, list):
            updated_boxes: List[Dict[str, Any]] = []
            for box in raw_boxes:
                if not isinstance(box, dict):
                    continue
                b_copy = dict(box)
                b_img = (
                    b_copy.get("image_path")
                    or b_copy.get("image")
                    or b_copy.get("vizj_path")
                )
                if isinstance(b_img, str) and b_img.strip():
                    b_name = Path(b_img.strip()).name
                    b_copy["image_path"] = id_remap.get(b_name, b_name)
                updated_boxes.append(b_copy)
            draft["layout_items"] = updated_boxes

        manifest["working_draft"] = draft

        inpaint_data_url: Optional[str] = None
        if "workspace/inpaint_canvas.png" in names:
            ic_bytes = zf.read("workspace/inpaint_canvas.png")
            inpaint_data_url = "data:image/png;base64," + base64.b64encode(
                ic_bytes
            ).decode("ascii")

        preview_bytes: Optional[bytes] = None
        if "workspace/master_preview.png" in names:
            preview_bytes = zf.read("workspace/master_preview.png")

    raw_proj_id = str(manifest.get("project_id") or "").strip()
    proj_id = (
        slugify(raw_proj_id)
        if raw_proj_id
        else f"proj_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    )
    manifest["project_id"] = proj_id
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

    projects_dir = (storage_root / user_slug / "projects").resolve()
    projects_dir.mkdir(parents=True, exist_ok=True)

    preview_url: Optional[str] = None
    if preview_bytes is not None:
        prev_file = projects_dir / f"{proj_id}_preview.png"
        prev_file.write_bytes(preview_bytes)
        preview_url = (
            f"api/v1/projects/stream/{user_slug}/{proj_id}_preview.png"
        )

    if persist_to_projects_vault:
        vsproj_file = projects_dir / f"{proj_id}.vsproj"
        vsproj_file.write_bytes(raw_zip_bytes)
        summary_meta = {
            "project_id": proj_id,
            "title": manifest.get("title", "Untitled Studio Project"),
            "created_at": manifest.get("created_at", manifest["updated_at"]),
            "updated_at": manifest["updated_at"],
            "width": manifest.get("width", 1024),
            "height": manifest.get("height", 1024),
            "model_variant": manifest.get("model_variant", "sft"),
            "prompt_mode": manifest.get("prompt_mode", "natural"),
            "asset_count": len(restored_assets),
            "size_bytes": len(raw_zip_bytes),
            "preview_url": preview_url,
            "download_url": f"api/v1/projects/{proj_id}/download",
        }
        (projects_dir / f"{proj_id}.json").write_text(
            json.dumps(summary_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "status": "loaded",
        "project_id": proj_id,
        "title": manifest.get("title", "Untitled Studio Project"),
        "working_draft": draft,
        "telemetry": manifest.get("telemetry"),
        "restored_assets": restored_assets,
        "id_remap": id_remap,
        "inpaint_canvas_data_url": inpaint_data_url,
        "preview_url": preview_url,
        "download_url": f"api/v1/projects/{proj_id}/download",
    }


def _probe_available_models(engine_instance: Optional[Any] = None) -> List[Dict[str, Any]]:
    if engine_instance is not None and hasattr(engine_instance, "list_available_models"):
        try:
            return list(engine_instance.list_available_models())
        except Exception:
            pass
    try:
        from ShadowBoxer.engine import resolve_model_path
    except Exception:
        try:
            from engine import resolve_model_path
        except Exception:
            resolve_model_path = None

    models_info: List[Dict[str, Any]] = []
    for variant, repo in SUPPORTED_MODELS.items():
        if resolve_model_path is None:
            models_info.append(
                {
                    "variant": variant,
                    "repo_id": repo,
                    "cached": False,
                    "snapshot_path": None,
                    "active": False,
                    "conditioning": None,
                }
            )
            continue
        try:
            snap = resolve_model_path(repo)
            manifest_file = snap / "conditioning_manifest.json"
            cond_meta = (
                json.loads(manifest_file.read_text(encoding="utf-8"))
                if manifest_file.is_file()
                else {"precision": "native"}
            )
            models_info.append(
                {
                    "variant": variant,
                    "repo_id": repo,
                    "cached": True,
                    "snapshot_path": str(snap),
                    "active": False,
                    "conditioning": cond_meta,
                }
            )
        except Exception:
            models_info.append(
                {
                    "variant": variant,
                    "repo_id": repo,
                    "cached": False,
                    "snapshot_path": None,
                    "active": False,
                    "conditioning": None,
                }
            )
    return models_info


__all__ = [
    "_probe_vizj_dimensions",
    "_inspect_reference_file",
    "_decode_data_url_bytes",
    "_resolve_user_upload_path",
    "_collect_draft_asset_names",
    "_build_vsproj_package",
    "_extract_and_restore_vsproj",
    "_probe_available_models",
]