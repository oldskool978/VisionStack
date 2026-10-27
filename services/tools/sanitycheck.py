#!/usr/bin/env python3
# services/tools/sanitycheck.py
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

TOOLS_DIR = Path(__file__).resolve().parent
SERVICES_DIR = TOOLS_DIR.parent
BACKEND_ROOT = SERVICES_DIR.parent
SHADOWBOXER_DIR = BACKEND_ROOT / "ShadowBoxer"

for _p in (TOOLS_DIR, SERVICES_DIR, SHADOWBOXER_DIR, BACKEND_ROOT):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import numpy as np
from PIL import Image

from readmegen import generate_services_readme
import services.router as router_mod
from services.router import (
    BASELINE_ENGINE_DEFAULTS,
    PowSubmission,
    SynthesisJob,
    SynthesisPayload,
    _resolve_user_upload_path,
    app,
    compute_queue,
    get_active_engine_defaults,
    verify_pow_solution,
)
import ShadowBoxer.schema as sb_schema


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _AsgiClient:
    def __init__(self, asgi_app: Any):
        self.app = asgi_app

    def request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        return asyncio.run(
            self._async_request(
                method=method,
                path=path,
                headers=headers,
                json_body=json_body,
                query=query,
            )
        )

    async def _async_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        body_bytes = (
            json.dumps(json_body).encode("utf-8") if json_body is not None else b""
        )
        raw_headers: List[Tuple[bytes, bytes]] = [
            (b"host", b"127.0.0.1:8766"),
        ]
        if json_body is not None:
            raw_headers.append((b"content-type", b"application/json"))
            raw_headers.append(
                (b"content-length", str(len(body_bytes)).encode("ascii"))
            )
        if headers:
            for k, v in headers.items():
                raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

        qs = urlencode(query).encode("ascii") if query else b""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": qs,
            "root_path": "",
            "headers": raw_headers,
            "client": ("127.0.0.1", 54321),
            "server": ("127.0.0.1", 8766),
        }

        req_sent = False
        status_code = 500
        resp_headers: Dict[str, str] = {}
        resp_chunks: List[bytes] = []

        async def receive() -> Dict[str, Any]:
            nonlocal req_sent
            if not req_sent:
                req_sent = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

        async def send(message: Dict[str, Any]) -> None:
            nonlocal status_code, resp_headers
            msg_type = message.get("type")
            if msg_type == "http.response.start":
                status_code = int(message.get("status", 500))
                hdrs = {}
                for k_b, v_b in message.get("headers", []):
                    hdrs[k_b.decode("latin-1").lower()] = v_b.decode("latin-1")
                resp_headers = hdrs
            elif msg_type == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    resp_chunks.append(chunk)

        await self.app(scope, receive, send)
        return status_code, resp_headers, b"".join(resp_chunks)


def _solve_pow(challenge: str, difficulty: int) -> str:
    prefix = "0" * difficulty
    nonce = 0
    while True:
        cand = f"{nonce:x}"
        digest = hashlib.sha256(f"{challenge}:{cand}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            return cand
        nonce += 1


def gate_1_multi_prefix_and_security_headers(client: _AsgiClient) -> None:
    print(
        "\n--- Gate 1: Multi-Prefix Ingress Parity, Security Headers & V03 Health Telemetry ---"
    )
    _assert(
        BASELINE_ENGINE_DEFAULTS["early_solver"] == "heun"
        and BASELINE_ENGINE_DEFAULTS["late_solver"] == "ipndm"
        and abs(float(BASELINE_ENGINE_DEFAULTS["early_cfg"]) - 3.2) < 1e-6
        and abs(float(BASELINE_ENGINE_DEFAULTS["late_cfg"]) - 2.8) < 1e-6
        and BASELINE_ENGINE_DEFAULTS["uncond_skip_layers"] == [24]
        and "box_attn_isolation" in BASELINE_ENGINE_DEFAULTS
        and "composite_box_images_to_init" in BASELINE_ENGINE_DEFAULTS
        and "mask_blend_mode" in BASELINE_ENGINE_DEFAULTS
        and "invert_inpaint_mask" in BASELINE_ENGINE_DEFAULTS
        and "add_vision_id" in BASELINE_ENGINE_DEFAULTS,
        f"Unexpected BASELINE_ENGINE_DEFAULTS: {BASELINE_ENGINE_DEFAULTS}",
    )
    expected_active = get_active_engine_defaults()

    prefixes = (
        "/api/v1",
        "/v1",
        "/VisionStack/api/v1",
        "/visionstack/api/v1",
        "/ShadowBoxer/api/v1",
        "/shadowboxer/api/v1",
    )
    last_hw: Dict[str, Any] = {}
    for pfx in prefixes:
        code, hdrs, raw = client.request("GET", f"{pfx}/health")
        _assert(code == 200, f"Expected 200 on {pfx}/health, got {code}")
        _assert(
            hdrs.get("cross-origin-opener-policy") == "same-origin"
            and hdrs.get("cross-origin-embedder-policy") == "require-corp"
            and hdrs.get("cross-origin-resource-policy") == "cross-origin",
            f"Missing COOP/COEP/CORP security headers on {pfx}/health: {hdrs}",
        )
        payload = json.loads(raw.decode("utf-8"))
        _assert(
            payload.get("status") == "online",
            f"Unexpected health payload: {payload}",
        )
        bp = payload.get("baseline_parameters", {})
        hw = payload.get("hardware", {})
        last_hw = hw
        _assert(
            bp.get("early_solver") == expected_active["early_solver"]
            and bp.get("late_solver") == expected_active["late_solver"]
            and bp.get("uncond_skip_layers") == expected_active["uncond_skip_layers"],
            f"Health baseline_parameters diverged from active defaults: {bp}",
        )
        _assert(
            "supported_mask_blend_modes" in payload
            and len(payload["supported_mask_blend_modes"]) >= 6,
            f"Missing supported_mask_blend_modes in /health payload: {payload.keys()}",
        )
        _assert(
            "contiguous_avail_gb" in hw
            and "pt_inactive_split_gb" in hw
            and "host_avail_gb" in hw
            and float(hw.get("dedicated_total_gb", 0.0)) >= 0.0,
            f"Incomplete hardware telemetry in /health: {hw}",
        )
    print(
        f"  [+] Multi-Prefix Health & Headers : Verified {len(prefixes)} gateway prefixes + COOP/COEP/CORP + Standby WDDM ({last_hw.get('adapter_name', 'GPU')}: {last_hw.get('dedicated_avail_gb', 0.0):.2f}/{last_hw.get('dedicated_total_gb', 0.0):.2f} GB)"
    )


def gate_2_pow_bot_defense(client: _AsgiClient) -> None:
    print(
        "\n--- Gate 2: Cryptographic Proof-of-Work (SHA-256 HMAC) & Replay Protection ---"
    )
    prev_diff = router_mod.POW_DIFFICULTY
    try:
        router_mod.POW_DIFFICULTY = 2
        code, _, raw = client.request("GET", "/api/v1/auth/challenge")
        _assert(code == 200, f"Challenge request failed: {code}")
        ch_data = json.loads(raw.decode("utf-8"))
        challenge = ch_data["challenge"]
        signature = ch_data["signature"]
        difficulty = int(ch_data["difficulty"])
        _assert(difficulty == 2, f"Expected difficulty 2, got {difficulty}")

        solution = _solve_pow(challenge, difficulty)
        _assert(
            not verify_pow_solution(challenge, "00" * 32, solution),
            "Tampered signature must be rejected",
        )
        _assert(
            verify_pow_solution(challenge, signature, solution),
            "Valid PoW solution was rejected",
        )
        _assert(
            not verify_pow_solution(challenge, signature, solution),
            "Replayed PoW solution must be rejected by _REPLAY_CACHE",
        )
    finally:
        router_mod.POW_DIFFICULTY = prev_diff
    print(
        "  [+] PoW Challenge & Replay Guard  : HMAC-SHA256 Signature + Nonce Mining + Single-Use Replay Eviction Verified"
    )


def gate_3_auth_session_and_preset_sync(client: _AsgiClient) -> str:
    print(
        "\n--- Gate 3: Creator Authentication, Session HMAC, Payload Coercion & Preset Sync ---"
    )
    code_bad, _, _ = client.request(
        "POST", "/api/v1/auth/login", json_body={"username": "nonexistent_user_xyz"}
    )
    _assert(code_bad == 401, f"Expected 401 for unknown user, got {code_bad}")

    code, _, raw = client.request(
        "POST", "/api/v1/auth/login", json_body={"username": "administrator"}
    )
    _assert(code == 200, f"Administrator login failed: {code}")
    auth_resp = json.loads(raw.decode("utf-8"))
    token = auth_resp["token"]
    _assert(
        auth_resp["status"] == "authenticated"
        and auth_resp["user"]["slug"] == "administrator"
        and auth_resp["user"]["is_admin"] is True,
        f"Unexpected login response: {auth_resp}",
    )

    test_payload = SynthesisPayload(
        title="Payload Coercion Audit",
        prompt="Spatial composition test",
        system_prompt="  You are a spatial layout architect.  ",
        thinking_trace="  ",
        add_vision_id=False,
        box_attn_isolation="0.85",
        uncond_uses_references=False,
        composite_box_images_to_init=True,
        mask_blend_mode="union",
        invert_inpaint_mask=True,
        num_images_per_prompt=2,
        uncond_skip_layers="22, 24",
        width=1030,
        height=1020,
        pow=PowSubmission(challenge="c", signature="s", solution_nonce="n"),
    )
    _assert(
        test_payload.system_prompt == "You are a spatial layout architect."
        and test_payload.thinking_trace is None
        and test_payload.add_vision_id is False
        and abs(float(test_payload.box_attn_isolation or 0.0) - 0.85) < 1e-6
        and test_payload.uncond_uses_references is False
        and test_payload.composite_box_images_to_init is True
        and test_payload.mask_blend_mode == "union"
        and test_payload.invert_inpaint_mask is True
        and test_payload.num_images_per_prompt == 2
        and test_payload.uncond_skip_layers == [22, 24]
        and test_payload.width == 1024
        and test_payload.height == 1008,
        f"SynthesisPayload field coercion failed: {test_payload}",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_preset = Path(tmpdir) / "default.json"
        prev_env = os.environ.get("SHADOWBOXER_DEFAULT_JSON")
        orig_dir = router_mod.SHADOWBOXER_DIR
        try:
            os.environ["SHADOWBOXER_DEFAULT_JSON"] = str(tmp_preset)
            router_mod.SHADOWBOXER_DIR = Path(tmpdir)
            p_code, _, p_raw = client.request(
                "POST",
                "/api/v1/presets/default",
                headers={"Authorization": f"Bearer {token}"},
                json_body={
                    "engine_defaults": {
                        "early_solver": "sde_gpu_pp",
                        "late_solver": "ipndm",
                        "early_cfg": 3.4,
                        "late_cfg": 2.6,
                        "box_attn_isolation": 0.75,
                        "composite_box_images_to_init": True,
                        "uncond_uses_references": False,
                        "mask_blend_mode": "intersect",
                        "invert_inpaint_mask": True,
                        "system_prompt": "Studio Default System Prompt",
                        "thinking_trace": "Studio Default Thinking Trace",
                        "add_vision_id": False,
                    }
                },
            )
            _assert(p_code == 200, f"Preset update failed: {p_code} {p_raw}")
            p_data = json.loads(p_raw.decode("utf-8"))
            bp = p_data["baseline_parameters"]
            _assert(
                p_data["engine_preset_active"] is True
                and bp["early_solver"] == "sde_gpu_pp"
                and abs(float(bp["early_cfg"]) - 3.4) < 1e-6
                and abs(float(bp["box_attn_isolation"]) - 0.75) < 1e-6
                and bp["composite_box_images_to_init"] is True
                and bp["uncond_uses_references"] is False
                and bp["mask_blend_mode"] == "intersect"
                and bp["invert_inpaint_mask"] is True
                and bp["system_prompt"] == "Studio Default System Prompt"
                and bp["thinking_trace"] == "Studio Default Thinking Trace"
                and bp["add_vision_id"] is False,
                f"Preset hot-reload mismatch: {p_data}",
            )
        finally:
            router_mod.SHADOWBOXER_DIR = orig_dir
            if prev_env is None:
                os.environ.pop("SHADOWBOXER_DEFAULT_JSON", None)
            else:
                os.environ["SHADOWBOXER_DEFAULT_JSON"] = prev_env
            sb_schema._PRESET_CACHE["mtime"] = -1.0
            router_mod.get_active_engine_defaults()

    print(
        "  [+] Creator Auth & Preset Sync    : Session Token + SynthesisPayload Coercion + default.json Hot-Reload Verified"
    )
    return token


def gate_4_bifurcated_vault_projects_and_references(
    client: _AsgiClient, token: str
) -> None:
    print(
        "\n--- Gate 4: Bifurcated Vault, Alpha/Header Probe, Gallery Promotion & .VSPROJ Packages ---"
    )
    auth_hdr = {"Authorization": f"Bearer {token}"}

    rgba = np.full((64, 96, 4), 160, dtype=np.uint8)
    rgba[:16, :16, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    b64_png = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(
        "ascii"
    )

    u_code, _, u_raw = client.request(
        "POST",
        "/api/v1/references/upload",
        headers=auth_hdr,
        json_body={
            "filename": "sanity_alpha_ref.png",
            "role": "photo_ref",
            "data_url": b64_png,
        },
    )
    _assert(u_code == 200, f"PNG reference upload failed: {u_code} {u_raw}")
    u_info = json.loads(u_raw.decode("utf-8"))
    png_ref_id = u_info["ref_id"]
    _assert(
        u_info["width"] == 96
        and u_info["height"] == 64
        and u_info["has_alpha"] is True
        and u_info["is_vizj"] is False
        and abs(float(u_info["aspect_ratio"]) - 1.5) < 1e-4,
        f"Unexpected uploaded RGBA PNG metadata: {u_info}",
    )

    vizj_hdr = (
        b"VIZJ_CONTAINER01"
        + b"\x00" * 16
        + struct.pack("<IIII", 1024, 768, 768, 512)
        + b"\x00" * 80
    )
    b64_vizj = base64.b64encode(vizj_hdr).decode("ascii")
    v_code, _, v_raw = client.request(
        "POST",
        "/api/v1/references/upload",
        headers=auth_hdr,
        json_body={
            "filename": "tier2_cache.vizj",
            "role": "photo_ref",
            "data_url": b64_vizj,
        },
    )
    _assert(v_code == 200, f".VIZJ reference upload failed: {v_code}")
    v_info = json.loads(v_raw.decode("utf-8"))
    vizj_ref_id = v_info["ref_id"]
    _assert(
        vizj_ref_id.endswith(".vizj")
        and v_info["is_vizj"] is True
        and v_info["width"] == 768
        and v_info["height"] == 512,
        f"Expected .vizj extension and probed 768x512 VAE dims: {v_info}",
    )

    user_images_dir = router_mod.STORAGE_ROOT / "administrator" / "images"
    user_images_dir.mkdir(parents=True, exist_ok=True)
    fake_job_id = "vs_sanity_promo_test"
    fake_master_png = user_images_dir / f"{fake_job_id}_master.png"
    fake_master_png.write_bytes(buf.getvalue())

    try:
        pr_code, _, pr_raw = client.request(
            "POST",
            f"/api/v1/references/from-image/{fake_job_id}",
            headers=auth_hdr,
            json_body={"role": "photo_ref", "variant_index": 1},
        )
        _assert(pr_code == 200, f"Gallery image promotion failed: {pr_code} {pr_raw}")
        pr_info = json.loads(pr_raw.decode("utf-8"))
        _assert(
            pr_info["ref_id"] == png_ref_id and pr_info.get("deduplicated") is True,
            f"Expected SHA-256 deduplicated promotion to match {png_ref_id}, got {pr_info}",
        )

        proj_draft = {
            "title": "Album Cover Master Project",
            "model_variant": "sft",
            "prompt": 'Dark vinyl album cover with title "NO SLEEP FOR THE WICKED"',
            "prompt_mode": "layout_json",
            "width": 1024,
            "height": 1024,
            "init_image_path": png_ref_id,
            "reference_image_paths": [vizj_ref_id],
            "layout_items": [
                {
                    "category": "subject",
                    "caption": "Album hero portrait",
                    "bbox": [128, 128, 896, 896],
                    "image_path": png_ref_id,
                    "isolate_box_attn": 0.85,
                    "paste_to_canvas": True,
                }
            ],
        }
        ps_code, _, ps_raw = client.request(
            "POST",
            "/api/v1/projects/save",
            headers=auth_hdr,
            json_body={
                "project_id": "album_cover_v1",
                "title": "Album Cover Master Project",
                "working_draft": proj_draft,
                "active_image_id": fake_job_id,
                "inpaint_canvas_data_url": b64_png,
            },
        )
        _assert(ps_code == 200, f"Project save failed: {ps_code} {ps_raw}")
        saved_proj = json.loads(ps_raw.decode("utf-8"))["project"]
        proj_id = saved_proj["project_id"]
        _assert(
            proj_id == "album_cover_v1" and saved_proj["asset_count"] == 2,
            f"Unexpected saved project metadata: {saved_proj}",
        )

        dl_code, _, dl_bytes = client.request(
            "GET", f"/api/v1/projects/{proj_id}/download", headers=auth_hdr
        )
        _assert(
            dl_code == 200 and len(dl_bytes) > 256,
            f"Project .vsproj download failed: {dl_code}",
        )

        client.request(
            "DELETE", f"/api/v1/references/{png_ref_id}", headers=auth_hdr
        )
        client.request(
            "DELETE", f"/api/v1/references/{vizj_ref_id}", headers=auth_hdr
        )
        _assert(
            not (router_mod.STORAGE_ROOT / "administrator" / "uploads" / png_ref_id).exists(),
            "Deleted reference asset still exists on disk",
        )

        ld_code, _, ld_raw = client.request(
            "POST", f"/api/v1/projects/{proj_id}/load", headers=auth_hdr
        )
        _assert(ld_code == 200, f"Project load failed: {ld_code} {ld_raw}")
        ld_data = json.loads(ld_raw.decode("utf-8"))
        _assert(
            png_ref_id in ld_data["restored_assets"]
            and vizj_ref_id in ld_data["restored_assets"]
            and ld_data["working_draft"]["init_image_path"] == png_ref_id
            and ld_data["inpaint_canvas_data_url"] is not None,
            f"Project load failed to self-heal and restore bundled assets: {ld_data}",
        )

        b64_vsproj = base64.b64encode(dl_bytes).decode("ascii")
        im_code, _, im_raw = client.request(
            "POST",
            "/api/v1/projects/import",
            headers=auth_hdr,
            json_body={"filename": "album_cover_v1.vsproj", "data_url": b64_vsproj},
        )
        _assert(im_code == 200, f"Project import failed: {im_code} {im_raw}")

        del_p_code, _, _ = client.request(
            "DELETE", f"/api/v1/projects/{proj_id}", headers=auth_hdr
        )
        _assert(del_p_code == 200, f"Project delete failed: {del_p_code}")
    finally:
        fake_master_png.unlink(missing_ok=True)

    st_code, _, st_raw = client.request(
        "POST", "/api/v1/references/stage-defaults", headers=auth_hdr
    )
    _assert(st_code == 200, f"Stage default test assets failed: {st_code} {st_raw}")
    staged_files = json.loads(st_raw.decode("utf-8")).get("staged_files", [])
    _assert(
        len(staged_files) >= 5 and "ref_subject_a.png" in staged_files,
        f"Expected 5 staged test input files, got {staged_files}",
    )

    l_code, _, l_raw = client.request("GET", "/api/v1/references", headers=auth_hdr)
    _assert(l_code == 200, f"Reference list failed: {l_code}")
    ref_list = json.loads(l_raw.decode("utf-8"))["references"]
    ref_map = {r["ref_id"]: r for r in ref_list}
    _assert(
        png_ref_id in ref_map
        and vizj_ref_id in ref_map
        and "ref_subject_a.png" in ref_map
        and ref_map[png_ref_id]["has_alpha"] is True
        and ref_map[vizj_ref_id]["is_vizj"] is True,
        f"Uploaded or staged references missing rich metadata in vault list: {ref_map.keys()}",
    )

    s_code, _, s_bytes = client.request(
        "GET", f"/api/v1/references/stream/administrator/{vizj_ref_id}"
    )
    _assert(
        s_code == 200 and s_bytes == vizj_hdr,
        "Streamed .vizj payload mismatch",
    )

    resolved_ok = _resolve_user_upload_path("administrator", png_ref_id)
    resolved_other = _resolve_user_upload_path("studio", png_ref_id)
    resolved_traversal = _resolve_user_upload_path(
        "studio", f"../administrator/uploads/{png_ref_id}"
    )
    _assert(
        resolved_ok is not None
        and resolved_other is None
        and resolved_traversal is None,
        "Cross-user vault isolation or traversal guard failed",
    )

    for rid in [png_ref_id, vizj_ref_id] + list(staged_files):
        d_code, _, _ = client.request(
            "DELETE", f"/api/v1/references/{rid}", headers=auth_hdr
        )
        _assert(d_code == 200, f"Failed deleting reference {rid}: {d_code}")

    print(
        "  [+] Per-User Vault & .VSPROJ      : RGBA Alpha + .VIZJ Probe + Gallery Promo + Hermetic .VSPROJ Save/Load/Import Verified"
    )


def gate_5_queue_etag_and_spa_guard(client: _AsgiClient, token: str) -> None:
    print(
        "\n--- Gate 5: ComputeQueue ETag Polling, Path-Traversal Guard & README Generation ---"
    )
    auth_hdr = {"Authorization": f"Bearer {token}"}

    dummy_job = SynthesisJob("administrator", {"title": "Sanity Test"})
    dummy_job.status = "PROCESSING"
    dummy_job.progress_pct = 42
    dummy_job.stage_description = "Integrating 4D-RoPE Velocity Field (10/21 passes)..."
    compute_queue.jobs[dummy_job.job_id] = dummy_job

    try:
        j_code, j_hdrs, j_raw = client.request(
            "GET", f"/api/v1/jobs/{dummy_job.job_id}", headers=auth_hdr
        )
        _assert(j_code == 200, f"Job status poll failed: {j_code}")
        j_info = json.loads(j_raw.decode("utf-8"))
        _assert(
            "additional_image_urls" in j_info,
            f"Missing additional_image_urls in job status: {j_info}",
        )
        etag = j_hdrs.get("etag")
        _assert(etag is not None, "Missing ETag header on /jobs/{job_id}")

        cached_code, _, _ = client.request(
            "GET",
            f"/api/v1/jobs/{dummy_job.job_id}",
            headers={**auth_hdr, "If-None-Match": etag},
        )
        _assert(
            cached_code == 304,
            f"Expected 304 Not Modified on matching ETag, got {cached_code}",
        )
    finally:
        compute_queue.jobs.pop(dummy_job.job_id, None)

    for forbidden_path in (
        "/ShadowBoxer/engine.py",
        "/services/router.py",
        "/storage/users/administrator/history.json",
    ):
        f_code, _, _ = client.request("GET", forbidden_path)
        _assert(
            f_code in (403, 404),
            f"Forbidden backend path '{forbidden_path}' was not blocked (status={f_code})",
        )

    readme_file = generate_services_readme()
    _assert(
        readme_file.is_file() and readme_file.stat().st_size > 1024,
        "services/README.md generation failed",
    )
    print(
        "  [+] Queue ETag & SPA Guard        : 304 ETag Cache + Batch URLs + Backend Traversal Lock + README.md Verified"
    )


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 92)
    print(
        "          VISIONSTACK SERVICES ROUTER & DAEMON 5-GATE AUDIT SUITE"
    )
    print("=" * 92)
    client = _AsgiClient(app)
    gate_1_multi_prefix_and_security_headers(client)
    gate_2_pow_bot_defense(client)
    token = gate_3_auth_session_and_preset_sync(client)
    gate_4_bifurcated_vault_projects_and_references(client, token)
    gate_5_queue_etag_and_spa_guard(client, token)
    if getattr(compute_queue.pipeline, "_memory_guard", None) is not None:
        try:
            compute_queue.pipeline._memory_guard.close()
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    print("\n" + "-" * 92)
    print(
        f"[+] STATUS: SERVICES ROUTER SECURE, ROBUST & CONVERGED IN {elapsed:.2f}s. ZERO DRIFT ACROSS ALL 5 GATES."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()