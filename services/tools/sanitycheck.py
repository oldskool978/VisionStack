#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path
import posixpath
import re
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
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
import services.runtime as runtime_pkg
from services.router import (
    BASELINE_ENGINE_DEFAULTS,
    GenerationRequest,
    PowSubmission,
    SUPPORTED_BOX_LABELS,
    SynthesisJob,
    SynthesisPayload,
    _RATE_LIMIT_BUCKETS,
    _REPLAY_CACHE,
    _close_preview_tokenizer,
    _resolve_user_upload_path,
    app,
    check_rate_limit,
    compute_queue,
    compute_webui_asset_version,
    get_active_engine_defaults,
    resolve_site_root,
    scale_layout_items_to_canvas,
    verify_pow_solution,
)
import ShadowBoxer.schema as sb_schema

_FIPS_180_4_H0: Tuple[int, ...] = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)

_FIPS_180_4_K: Tuple[int, ...] = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)

_HTML_TO_MODEL_NUMERIC_SPEC: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("widthInput", "width", "width"),
    ("heightInput", "height", "height"),
    ("stepsInput", "num_inference_steps", "num_inference_steps"),
    ("numImagesInput", "num_images_per_prompt", "num_images_per_prompt"),
    ("handoffInput", "handoff_threshold", "handoff_threshold"),
    ("earlyCfgInput", "early_cfg", "early_cfg"),
    ("lateCfgInput", "late_cfg", "late_cfg"),
    ("shiftInput", "shift", "shift"),
    ("slgScaleInput", "slg_scale", "slg_scale"),
    ("slgStartInput", "slg_start", "slg_start"),
    ("slgStopInput", "slg_stop", "slg_stop"),
    ("etaInput", "eta", "eta"),
    ("sNoiseInput", "s_noise", "s_noise"),
    ("maxSeqLenInput", "max_sequence_length", "max_sequence_length"),
    ("maxVitPatchesInput", "max_vit_patches", "max_vit_patches"),
    ("boxAttnIsolationInput", "box_attn_isolation", "box_attn_isolation"),
    ("denoiseInput", "denoise_strength", "denoise_strength"),
    ("mtfBoostInput", "mtf_boost", "mtf_boost"),
    ("featherInput", "mask_feather_radius", "mask_feather_radius"),
    ("clampSigmaInput", "latent_clamp_sigma", "latent_clamp_sigma"),
    ("softKneeTauInput", "soft_knee_tau", "soft_knee_tau"),
    ("loraScaleInput", "edit_lora_scale", "lora_scale"),
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _rotr32(v: int, n: int) -> int:
    v &= 0xFFFFFFFF
    return ((v >> n) | (v << (32 - n))) & 0xFFFFFFFF


def _simulate_extracted_js_sha256(
    ascii_msg: str, h_init: Sequence[int], k_table: Sequence[int]
) -> str:
    msg_bytes = bytearray(ascii_msg.encode("latin-1"))
    bit_len = len(msg_bytes) * 8
    msg_bytes.append(0x80)
    while (len(msg_bytes) % 64) != 56:
        msg_bytes.append(0x00)
    msg_bytes.extend(struct.pack(">Q", bit_len))

    h = [x & 0xFFFFFFFF for x in h_init]
    for chunk_start in range(0, len(msg_bytes), 64):
        w = list(struct.unpack(">16I", msg_bytes[chunk_start : chunk_start + 64])) + [
            0
        ] * 48
        for i in range(16, 64):
            s0 = _rotr32(w[i - 15], 7) ^ _rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _rotr32(w[i - 2], 17) ^ _rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF

        a, b, c, d, e, f, g, hh = h
        for i in range(64):
            S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            ch = (e & f) ^ ((~e & 0xFFFFFFFF) & g)
            temp1 = (hh + S1 + ch + k_table[i] + w[i]) & 0xFFFFFFFF
            S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & 0xFFFFFFFF
            hh = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF

        h = [
            (h[0] + a) & 0xFFFFFFFF,
            (h[1] + b) & 0xFFFFFFFF,
            (h[2] + c) & 0xFFFFFFFF,
            (h[3] + d) & 0xFFFFFFFF,
            (h[4] + e) & 0xFFFFFFFF,
            (h[5] + f) & 0xFFFFFFFF,
            (h[6] + g) & 0xFFFFFFFF,
            (h[7] + hh) & 0xFFFFFFFF,
        ]
    return "".join(f"{word:08x}" for word in h)


def _js_round(val: float) -> int:
    return int(math.floor(val + 0.5))


def _js_snap_lattice_floor_half(val: float, mult: int = 16) -> int:
    return int(math.ceil((float(val) / float(mult)) - 0.5 - 1e-9)) * int(mult)


def _js_pixel_to_abs_centralized(
    ymin: float, xmin: float, ymax: float, xmax: float, height: int, width: int
) -> List[int]:
    h_int = int(height)
    w_int = int(width)
    cy = h_int // 2
    cx = w_int // 2
    max_y = h_int - cy
    max_x = w_int - cx
    eff_ymax = float(h_int) if _js_round(ymax) == h_int - 1 else float(ymax)
    eff_xmax = float(w_int) if _js_round(xmax) == w_int - 1 else float(xmax)
    return [
        max(-cy, min(max_y, _js_round(ymin - cy))),
        max(-cx, min(max_x, _js_round(xmin - cx))),
        max(-cy, min(max_y, _js_round(eff_ymax - cy))),
        max(-cx, min(max_x, _js_round(eff_xmax - cx))),
    ]


def _js_scale_bbox_for_resolution(
    bbox: Sequence[float], old_w: int, old_h: int, new_w: int, new_h: int
) -> List[int]:
    y0, x0, y1, x1 = [float(v) for v in bbox]
    sh = float(max(16, int(old_h)))
    sw = float(max(16, int(old_w)))
    h = max(16, int(new_h))
    w = max(16, int(new_w))
    py0 = max(0, min(int(sh) - 1, _js_round(min(y0, y1))))
    px0 = max(0, min(int(sw) - 1, _js_round(min(x0, x1))))
    py1 = max(py0 + 1, min(int(sh), _js_round(max(y0, y1))))
    px1 = max(px0 + 1, min(int(sw), _js_round(max(x0, x1))))
    uy0 = max(0.0, min(1.0, py0 / sh))
    ux0 = max(0.0, min(1.0, px0 / sw))
    uy1 = max(0.0, min(1.0, py1 / sh))
    ux1 = max(0.0, min(1.0, px1 / sw))
    ny0 = (
        0
        if uy0 <= 1e-5
        else max(0, min(h - 16, _js_snap_lattice_floor_half(uy0 * h, 16)))
    )
    nx0 = (
        0
        if ux0 <= 1e-5
        else max(0, min(w - 16, _js_snap_lattice_floor_half(ux0 * w, 16)))
    )
    ny1 = (
        h
        if uy1 >= 1.0 - 1e-5
        else max(ny0 + 16, min(h, _js_snap_lattice_floor_half(uy1 * h, 16)))
    )
    nx1 = (
        w
        if ux1 >= 1.0 - 1e-5
        else max(nx0 + 16, min(w, _js_snap_lattice_floor_half(ux1 * w, 16)))
    )
    return [ny0, nx0, ny1, nx1]


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


def _solve_pow_nonces(
    challenge: str, difficulty: int, count: int = 1
) -> List[str]:
    prefix = "0" * difficulty
    found: List[str] = []
    nonce = 0
    while len(found) < count:
        cand = f"{nonce:x}"
        digest = hashlib.sha256(f"{challenge}:{cand}".encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            found.append(cand)
        nonce += 1
    return found


def gate_1_multi_prefix_and_security_headers(client: _AsgiClient) -> None:
    print(
        "\n--- Gate 1: Runtime Bifurcation, Multi-Prefix Ingress & V03 Health Telemetry ---"
    )
    runtime_dir = SERVICES_DIR / "runtime"
    for mod_file in (
        "__init__.py",
        "env.py",
        "security.py",
        "models.py",
        "vault.py",
        "queue.py",
        "routes.py",
    ):
        _assert(
            (runtime_dir / mod_file).is_file(),
            f"Missing bifurcated runtime module: {runtime_dir / mod_file}",
        )
    _assert(
        runtime_pkg.app is app and runtime_pkg.compute_queue is compute_queue,
        "services.runtime package failed to bind canonical app or compute_queue",
    )

    _assert(
        BASELINE_ENGINE_DEFAULTS["early_solver"] == "heun"
        and BASELINE_ENGINE_DEFAULTS["late_solver"] == "ipndm"
        and abs(float(BASELINE_ENGINE_DEFAULTS["early_cfg"]) - 3.2) < 1e-6
        and abs(float(BASELINE_ENGINE_DEFAULTS["late_cfg"]) - 2.8) < 1e-6
        and BASELINE_ENGINE_DEFAULTS["uncond_skip_layers"] == [24]
        and BASELINE_ENGINE_DEFAULTS["max_sequence_length"] == 16384
        and "box_attn_isolation" in BASELINE_ENGINE_DEFAULTS
        and "composite_box_images_to_init" in BASELINE_ENGINE_DEFAULTS
        and "mask_blend_mode" in BASELINE_ENGINE_DEFAULTS
        and "invert_inpaint_mask" in BASELINE_ENGINE_DEFAULTS
        and "add_vision_id" in BASELINE_ENGINE_DEFAULTS,
        f"Unexpected BASELINE_ENGINE_DEFAULTS: {BASELINE_ENGINE_DEFAULTS}",
    )

    expected_active = get_active_engine_defaults()
    expected_asset_ver = compute_webui_asset_version()
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
            payload.get("status") == "online"
            and payload.get("webui_asset_version") == expected_asset_ver,
            f"Unexpected health payload: {payload}",
        )
        bp = payload.get("baseline_parameters", {})
        hw = payload.get("hardware", {})
        last_hw = hw
        _assert(
            bp.get("early_solver") == expected_active["early_solver"]
            and bp.get("late_solver") == expected_active["late_solver"]
            and bp.get("uncond_skip_layers") == expected_active["uncond_skip_layers"]
            and bp.get("max_sequence_length") == expected_active["max_sequence_length"],
            f"Health baseline_parameters diverged from active defaults: {bp}",
        )
        _assert(
            "supported_mask_blend_modes" in payload
            and len(payload["supported_mask_blend_modes"]) >= 6
            and "available_models" in payload
            and payload.get("supported_box_labels") == SUPPORTED_BOX_LABELS
            and int(payload.get("default_speculative_cot_tokens", 0)) >= 16,
            f"Missing capability metadata in /health payload: {payload.keys()}",
        )
        _assert(
            "contiguous_avail_gb" in hw
            and "pt_inactive_split_gb" in hw
            and "host_avail_gb" in hw
            and float(hw.get("dedicated_total_gb", 0.0)) >= 0.0,
            f"Incomplete hardware telemetry in /health: {hw}",
        )

    print(
        f"  [+] Runtime & Multi-Prefix Health : 7 Modules in services/runtime/ + {len(prefixes)} Prefixes + Bundle #{expected_asset_ver} + Standby WDDM ({last_hw.get('adapter_name', 'GPU')}: {last_hw.get('dedicated_avail_gb', 0.0):.2f}/{last_hw.get('dedicated_total_gb', 0.0):.2f} GB)"
    )


def gate_2_pow_bot_defense(client: _AsgiClient) -> None:
    print(
        "\n--- Gate 2: Proxy-Hardened PoW, Single-Burn Nonce, Rate Guard & TOCTOU Quota Lock ---"
    )
    prev_diff = router_mod.POW_DIFFICULTY
    prev_max_inflight = router_mod.MAX_USER_INFLIGHT_JOBS
    try:
        router_mod.POW_DIFFICULTY = 2
        router_mod.MAX_USER_INFLIGHT_JOBS = 2
        _REPLAY_CACHE.clear()
        _RATE_LIMIT_BUCKETS.clear()

        proxy_hdr_a = {"X-Forwarded-For": "203.0.113.10, 127.0.0.1"}
        proxy_hdr_b = {"X-Forwarded-For": "198.51.100.77, 127.0.0.1"}

        code, _, raw = client.request(
            "GET", "/api/v1/auth/challenge", headers=proxy_hdr_a
        )
        _assert(code == 200, f"Challenge request failed: {code}")
        ch_data = json.loads(raw.decode("utf-8"))
        challenge = ch_data["challenge"]
        signature = ch_data["signature"]
        difficulty = int(ch_data["difficulty"])
        _assert(difficulty == 2, f"Expected difficulty 2, got {difficulty}")

        sol_1, sol_2 = _solve_pow_nonces(challenge, difficulty, count=2)
        _assert(sol_1 != sol_2, "Expected two distinct valid nonces for single challenge")

        _assert(
            not verify_pow_solution(
                challenge, "00" * 32, sol_1, client_ip="203.0.113.10"
            ),
            "Tampered signature must be rejected",
        )
        _assert(
            not verify_pow_solution(
                challenge, signature, sol_1, client_ip="198.51.100.77"
            ),
            "Challenge bound to Proxy IP 203.0.113.10 must be rejected when redeemed from 198.51.100.77",
        )

        downgraded_ch = f"{int(time.time())}:downgrade_nonce_123456:0"
        downgraded_sig = hmac.new(
            router_mod.POW_SECRET.encode("utf-8"),
            downgraded_ch.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        _assert(
            not verify_pow_solution(
                downgraded_ch, downgraded_sig, "0", client_ip="127.0.0.1"
            ),
            "Difficulty downgrade below POW_DIFFICULTY floor must be rejected",
        )

        _assert(
            verify_pow_solution(
                challenge, signature, sol_1, client_ip="203.0.113.10"
            ),
            "Valid IP-bound PoW solution was rejected",
        )
        _assert(
            not verify_pow_solution(
                challenge, signature, sol_1, client_ip="203.0.113.10"
            ),
            "Exact replayed PoW solution must be rejected",
        )
        _assert(
            not verify_pow_solution(
                challenge, signature, sol_2, client_ip="203.0.113.10"
            ),
            "Second mined nonce on already-burned challenge ID must be rejected",
        )

        l_code, _, l_raw = client.request(
            "POST",
            "/api/v1/auth/login",
            headers=proxy_hdr_a,
            json_body={"username": "studio"},
        )
        _assert(l_code == 200, f"Studio login failed: {l_code}")
        studio_token = json.loads(l_raw.decode("utf-8"))["token"]
        studio_headers = {
            **proxy_hdr_a,
            "Authorization": f"Bearer {studio_token}",
        }

        _, _, ch_s1_raw = client.request(
            "GET", "/api/v1/auth/challenge", headers=proxy_hdr_a
        )
        ch_s1 = json.loads(ch_s1_raw.decode("utf-8"))
        s1_n1, s1_n2 = _solve_pow_nonces(
            ch_s1["challenge"], int(ch_s1["difficulty"]), count=2
        )

        syn_1_code, _, syn_1_raw = client.request(
            "POST",
            "/api/v1/synthesize",
            headers=studio_headers,
            json_body={
                "title": "PoW Gate Job 1",
                "prompt": "A brushed-steel refrigerator",
                "pow": {
                    "challenge": ch_s1["challenge"],
                    "signature": ch_s1["signature"],
                    "solution_nonce": s1_n1,
                },
            },
        )
        _assert(
            syn_1_code == 200,
            f"Expected 200 on valid PoW /synthesize submission, got {syn_1_code}: {syn_1_raw}",
        )

        syn_burn_code, _, _ = client.request(
            "POST",
            "/api/v1/synthesize",
            headers=studio_headers,
            json_body={
                "title": "PoW Multi-Nonce Replay Attempt",
                "prompt": "A brushed-steel refrigerator",
                "pow": {
                    "challenge": ch_s1["challenge"],
                    "signature": ch_s1["signature"],
                    "solution_nonce": s1_n2,
                },
            },
        )
        _assert(
            syn_burn_code == 403,
            f"Expected 403 Forbidden when reusing burned challenge with 2nd valid nonce on /synthesize, got {syn_burn_code}",
        )

        _, _, ch_spoof_raw = client.request(
            "GET", "/api/v1/auth/challenge", headers=proxy_hdr_a
        )
        ch_spoof = json.loads(ch_spoof_raw.decode("utf-8"))
        spoof_nonce = _solve_pow_nonces(
            ch_spoof["challenge"], int(ch_spoof["difficulty"]), count=1
        )[0]
        syn_spoof_code, _, _ = client.request(
            "POST",
            "/api/v1/synthesize",
            headers={**proxy_hdr_b, "Authorization": f"Bearer {studio_token}"},
            json_body={
                "title": "Cross-IP Spoof Attempt",
                "prompt": "A brushed-steel refrigerator",
                "pow": {
                    "challenge": ch_spoof["challenge"],
                    "signature": ch_spoof["signature"],
                    "solution_nonce": spoof_nonce,
                },
            },
        )
        _assert(
            syn_spoof_code == 403,
            f"Expected 403 Forbidden when redeeming challenge from spoofed proxy IP, got {syn_spoof_code}",
        )

        _, _, ch_s2_raw = client.request(
            "GET", "/api/v1/auth/challenge", headers=proxy_hdr_a
        )
        ch_s2 = json.loads(ch_s2_raw.decode("utf-8"))
        s2_n1 = _solve_pow_nonces(
            ch_s2["challenge"], int(ch_s2["difficulty"]), count=1
        )[0]
        syn_2_code, _, _ = client.request(
            "POST",
            "/api/v1/synthesize",
            headers=studio_headers,
            json_body={
                "title": "PoW Gate Job 2",
                "prompt": "A brushed-steel refrigerator",
                "pow": {
                    "challenge": ch_s2["challenge"],
                    "signature": ch_s2["signature"],
                    "solution_nonce": s2_n1,
                },
            },
        )
        _assert(syn_2_code == 200, f"Expected 200 on 2nd in-flight job, got {syn_2_code}")

        _, _, ch_s3_raw = client.request(
            "GET", "/api/v1/auth/challenge", headers=proxy_hdr_a
        )
        ch_s3 = json.loads(ch_s3_raw.decode("utf-8"))
        s3_n1 = _solve_pow_nonces(
            ch_s3["challenge"], int(ch_s3["difficulty"]), count=1
        )[0]
        syn_3_code, _, _ = client.request(
            "POST",
            "/api/v1/synthesize",
            headers=studio_headers,
            json_body={
                "title": "In-Flight Cap Overflow Job 3",
                "prompt": "A brushed-steel refrigerator",
                "pow": {
                    "challenge": ch_s3["challenge"],
                    "signature": ch_s3["signature"],
                    "solution_nonce": s3_n1,
                },
            },
        )
        _assert(
            syn_3_code == 429,
            f"Expected 429 Too Many Requests when exceeding MAX_USER_INFLIGHT_JOBS, got {syn_3_code}",
        )

        rate_blocked = False
        for _ in range(5):
            try:
                check_rate_limit(
                    "192.0.2.99",
                    scope="test_flood",
                    max_requests=3,
                    window_seconds=60.0,
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    rate_blocked = True
                    break
        _assert(rate_blocked, "Sliding-window IP rate limiter failed to trigger 429")
    finally:
        router_mod.POW_DIFFICULTY = prev_diff
        router_mod.MAX_USER_INFLIGHT_JOBS = prev_max_inflight
        compute_queue.jobs.clear()
        compute_queue.events.clear()
        compute_queue.queue._queue.clear()
        _REPLAY_CACHE.clear()
        _RATE_LIMIT_BUCKETS.clear()

    print(
        "  [+] Proxy PoW & TOCTOU Quota Guard: IP-Bound HMAC + Single-Burn Nonce + Difficulty Floor + In-Flight 429 Lock Verified"
    )


def gate_3_auth_session_and_preset_sync(client: _AsgiClient) -> str:
    print(
        "\n--- Gate 3: Creator Authentication, Session HMAC, 32K Payload, CoT Preview & Preset Sync ---"
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
        and auth_resp["user"]["is_admin"] is True
        and "available_models" in auth_resp,
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
        reference_zero_timestep=True,
        reference_isolated_kv=False,
        prepass_warmup=False,
        max_sequence_length=16384,
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
        and test_payload.reference_zero_timestep is True
        and test_payload.reference_isolated_kv is False
        and test_payload.prepass_warmup is False
        and test_payload.max_sequence_length == 16384
        and test_payload.mask_blend_mode == "union"
        and test_payload.invert_inpaint_mask is True
        and test_payload.num_images_per_prompt == 2
        and test_payload.uncond_skip_layers == [22, 24]
        and test_payload.width == 1024
        and test_payload.height == 1008,
        f"SynthesisPayload field coercion failed: {test_payload}",
    )

    prv_auto_code, _, prv_auto_raw = client.request(
        "POST",
        "/api/v1/prompt/preview",
        json_body={
            "prompt": 'A brushed-steel door with sticky note "SOVEREIGN STUDIO"',
            "prompt_mode": "layout_json",
            "thinking_trace": "auto",
            "max_sequence_length": 16384,
            "max_vit_patches": 2304,
            "width": 1024,
            "height": 1024,
            "box_attn_isolation": 0.85,
            "composite_box_images_to_init": True,
            "reference_image_paths": [],
            "layout_items": [
                {
                    "label": "object",
                    "caption": "Brushed-steel refrigerator door",
                    "bbox": [0, 0, 1024, 576],
                    "image_path": "ref_subject_a.png",
                    "input_role": "photo_ref",
                    "paste_to_canvas": True,
                    "isolate_box_attn": 0.85,
                    "z_order": 1,
                    "enabled": True,
                },
                {
                    "label": "object",
                    "caption": "Structural depth guide",
                    "bbox": [0, 0, 1024, 576],
                    "image_path": "control_map.png",
                    "input_role": "depth_map",
                    "isolate_box_attn": 0.85,
                    "z_order": 2,
                    "enabled": True,
                },
                {
                    "label": "text",
                    "text": "SOVEREIGN STUDIO",
                    "bbox": [256, 112, 512, 464],
                    "z_order": 5,
                    "enabled": True,
                },
            ],
        },
    )
    _assert(
        prv_auto_code == 200,
        f"POST /api/v1/prompt/preview (auto) failed: {prv_auto_code}",
    )
    prv_auto = json.loads(prv_auto_raw.decode("utf-8"))
    sv_graph = prv_auto.get("spatio_visual_graph", {})
    _assert(
        '"SOVEREIGN STUDIO"' in prv_auto["effective_prompt"]
        and '"Crisp legible typography with clean kerning"'
        in prv_auto["effective_prompt"]
        and prv_auto["resolved_thinking_trace"] is not None
        and "Typography Lock" in prv_auto["resolved_thinking_trace"]
        and prv_auto["speculative_cot_tokens"] == 0
        and int(prv_auto.get("prompt_tokens", 0)) > 20
        and int(prv_auto.get("trace_tokens", 0)) > 20
        and int(prv_auto.get("vision_tokens", 0)) > 100
        and int(prv_auto.get("total_context_tokens", 0))
        > int(prv_auto["vision_tokens"]) + int(prv_auto["prompt_tokens"])
        and int(prv_auto.get("max_sequence_length", 0)) == 16384
        and 0.0 < float(prv_auto.get("horizon_utilization_pct", 0.0)) < 10.0
        and sv_graph.get("dit_reference_count") == 2
        and sv_graph.get("collaborating_ref_groups") == [[0, 1], [0, 1]]
        and len(sv_graph.get("aperture_overrides", [])) == 1
        and len(sv_graph.get("foreground_carveouts", [])) == 1,
        f"Unexpected auto spatial CoT preview response or Spatio-Visual Graph: {prv_auto}",
    )

    prv_spec_code, _, prv_spec_raw = client.request(
        "POST",
        "/api/v1/prompt/preview",
        json_body={
            "prompt": "A brushed-steel door",
            "prompt_mode": "natural",
            "thinking_trace": "speculative:48",
        },
    )
    _assert(
        prv_spec_code == 200,
        f"POST /api/v1/prompt/preview (speculative:48) failed: {prv_spec_code}",
    )
    prv_spec = json.loads(prv_spec_raw.decode("utf-8"))
    _assert(
        prv_spec["resolved_thinking_trace"] is None
        and prv_spec["speculative_cot_tokens"] == 48
        and int(prv_spec.get("trace_tokens", 0)) == 48
        and int(prv_spec.get("total_context_tokens", 0)) > 50,
        f"Unexpected speculative:48 CoT preview response: {prv_spec}",
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
                        "max_sequence_length": 16384,
                        "box_attn_isolation": 0.75,
                        "composite_box_images_to_init": True,
                        "uncond_uses_references": False,
                        "mask_blend_mode": "intersect",
                        "invert_inpaint_mask": True,
                        "system_prompt": "Studio Default System Prompt",
                        "thinking_trace": "speculative:64",
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
                and bp["max_sequence_length"] == 16384
                and abs(float(bp["box_attn_isolation"]) - 0.75) < 1e-6
                and bp["composite_box_images_to_init"] is True
                and bp["uncond_uses_references"] is False
                and bp["mask_blend_mode"] == "intersect"
                and bp["invert_inpaint_mask"] is True
                and bp["system_prompt"] == "Studio Default System Prompt"
                and bp["thinking_trace"] == "speculative:64"
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
        f"  [+] Creator Auth & CoT Preview    : Session Token + VizToken Horizon ({prv_auto['total_context_tokens']}/{prv_auto['max_sequence_length']} tok) + SpatioVisualGraph + default.json Verified"
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

    prv_cut_code, _, prv_cut_raw = client.request(
        "POST",
        "/api/v1/prompt/preview",
        headers=auth_hdr,
        json_body={
            "prompt": "Cutout telemetry audit",
            "prompt_mode": "layout_json",
            "thinking_trace": "auto",
            "width": 1024,
            "height": 1024,
            "layout_items": [
                {
                    "label": "object",
                    "caption": "Translucent cutout badge",
                    "bbox": [256, 256, 768, 768],
                    "image_path": png_ref_id,
                    "input_role": "photo_ref",
                    "paste_to_canvas": True,
                    "enabled": True,
                }
            ],
        },
    )
    _assert(prv_cut_code == 200, f"Vault cutout preview failed: {prv_cut_code}")
    prv_cut_data = json.loads(prv_cut_raw.decode("utf-8"))
    _assert(
        "cutout=True" in (prv_cut_data.get("resolved_thinking_trace") or "")
        and "vae=512x512" in (prv_cut_data.get("resolved_thinking_trace") or ""),
        f"Vault-aware /prompt/preview failed to inject live cutout=True or vae=512x512 telemetry: {prv_cut_data}",
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
                    "label": "character",
                    "category": "character",
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
        len(staged_files) >= 6 and "ref_cutout_c.png" in staged_files,
        f"Expected 6 staged test input files including ref_cutout_c.png, got {staged_files}",
    )

    l_code, _, l_raw = client.request("GET", "/api/v1/references", headers=auth_hdr)
    _assert(l_code == 200, f"Reference list failed: {l_code}")
    ref_list = json.loads(l_raw.decode("utf-8"))["references"]
    ref_map = {r["ref_id"]: r for r in ref_list}
    _assert(
        png_ref_id in ref_map
        and vizj_ref_id in ref_map
        and "ref_subject_a.png" in ref_map
        and "ref_cutout_c.png" in ref_map
        and ref_map[png_ref_id]["has_alpha"] is True
        and ref_map["ref_cutout_c.png"]["has_alpha"] is True
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


def _extract_pydantic_bounds(field_info: Any) -> Tuple[Optional[float], Optional[float]]:
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    for meta in getattr(field_info, "metadata", ()):
        if hasattr(meta, "ge") and meta.ge is not None:
            min_val = float(meta.ge)
        if hasattr(meta, "gt") and meta.gt is not None:
            min_val = float(meta.gt)
        if hasattr(meta, "le") and meta.le is not None:
            max_val = float(meta.le)
        if hasattr(meta, "lt") and meta.lt is not None:
            max_val = float(meta.lt)
    return min_val, max_val


def _audit_cross_stack_numeric_and_box_schema_supremacy(
    site_root: Path, html_text: str
) -> int:
    layout_js_path = site_root / "assets" / "app" / "layout_editor.js"
    geom_js_path = site_root / "assets" / "app" / "geometry.js"
    styles_css_path = site_root / "assets" / "styles.css"
    _assert(
        layout_js_path.is_file()
        and geom_js_path.is_file()
        and styles_css_path.is_file(),
        "Missing layout_editor.js, geometry.js, or styles.css in webui/assets",
    )
    layout_src = layout_js_path.read_text(encoding="utf-8")
    geom_src = geom_js_path.read_text(encoding="utf-8")
    styles_src = styles_css_path.read_text(encoding="utf-8")

    for css_token in (
        "--char-accent",
        ".spatial-box.char-box",
        ".spatial-box.text-box",
        ".viewport-hud-box.char-hud",
        ".viewport-hud-box.text-hud",
        ".inpaint-box-guide.char-guide",
        ".inpaint-box-guide.text-guide",
    ):
        _assert(
            css_token in styles_src,
            f"Missing 3-class box trichotomy CSS rule '{css_token}' in styles.css",
        )
    _assert(
        "char-hud" in geom_src and "char-guide" in geom_src,
        "geometry.js is missing char-hud or char-guide class bindings",
    )

    _assert(
        re.search(r"<input[^>]+data-box-cat=", layout_src) is None,
        "Found legacy <input data-box-cat=...> in layout_editor.js; box categories must use constrained <select>",
    )
    cat_select_match = re.search(
        r"<select[^>]+data-box-cat=[^>]*>(.*?)</select>",
        layout_src,
        re.DOTALL,
    )
    _assert(
        cat_select_match is not None,
        "Missing <select ... data-box-cat=...> dropdown in layout_editor.js",
    )
    cat_options = re.findall(
        r'<option\s+value="([^"]+)"', cat_select_match.group(1)
    )
    _assert(
        cat_options == list(SUPPORTED_BOX_LABELS),
        f"Box category dropdown options {cat_options} diverged from SUPPORTED_BOX_LABELS {SUPPORTED_BOX_LABELS}",
    )

    for html_id, model_field, baseline_key in _HTML_TO_MODEL_NUMERIC_SPEC:
        tag_match = re.search(
            rf'<input\b[^>]*\bid="{re.escape(html_id)}"[^>]*>', html_text
        )
        _assert(
            tag_match is not None,
            f"Missing numeric <input id='{html_id}'> in index.html",
        )
        tag_str = tag_match.group(0)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag_str))
        _assert(
            attrs.get("type") == "number",
            f"Expected type='number' on #{html_id}, got {attrs.get('type')}",
        )

        f_info = SynthesisPayload.model_fields.get(model_field)
        _assert(
            f_info is not None,
            f"Field '{model_field}' missing from SynthesisPayload",
        )
        p_min, p_max = _extract_pydantic_bounds(f_info)
        if p_min is not None:
            _assert(
                "min" in attrs and abs(float(attrs["min"]) - p_min) < 1e-6,
                f"HTML min='{attrs.get('min')}' on #{html_id} diverged from SynthesisPayload.{model_field} ge={p_min}",
            )
        if p_max is not None:
            _assert(
                "max" in attrs and abs(float(attrs["max"]) - p_max) < 1e-6,
                f"HTML max='{attrs.get('max')}' on #{html_id} diverged from SynthesisPayload.{model_field} le={p_max}",
            )
        if baseline_key and baseline_key in BASELINE_ENGINE_DEFAULTS:
            expected_val = float(BASELINE_ENGINE_DEFAULTS[baseline_key])
            _assert(
                "value" in attrs
                and abs(float(attrs["value"]) - expected_val) < 1e-6,
                f"HTML default value='{attrs.get('value')}' on #{html_id} diverged from BASELINE_ENGINE_DEFAULTS['{baseline_key}']={expected_val}",
            )

    aspect_presets: Tuple[Tuple[int, int], ...] = (
        (1024, 1024),
        (512, 512),
        (1280, 832),
        (832, 1280),
        (1344, 768),
        (768, 1344),
    )
    seed_boxes = [
        {"label": "object", "caption": "Left door", "bbox": [0, 0, 1024, 576]},
        {
            "label": "character",
            "caption": "Subject",
            "bbox": [160, 288, 896, 736],
        },
        {
            "label": "text",
            "text": "SOVEREIGN 2026",
            "bbox": [256, 112, 512, 464],
        },
    ]
    for w_px, h_px in aspect_presets:
        _assert(
            w_px % 16 == 0 and h_px % 16 == 0,
            f"Aspect preset {w_px}x{h_px} is not aligned to 16px Flux2-VAE stride",
        )
        py_scaled = scale_layout_items_to_canvas(
            seed_boxes,
            height=h_px,
            width=w_px,
            src_height=1024,
            src_width=1024,
            coords_are_pixels=True,
        )
        scaled_items: List[Dict[str, Any]] = []
        js_centralized_map: Dict[str, List[int]] = {}
        for idx_b, item in enumerate(seed_boxes):
            sb = _js_scale_bbox_for_resolution(
                item["bbox"], 1024, 1024, w_px, h_px
            )
            _assert(
                sb == py_scaled[idx_b]["bbox"]
                and all(c % 16 == 0 for c in sb)
                and 0 <= sb[0] < sb[2] <= h_px
                and 0 <= sb[1] < sb[3] <= w_px,
                f"Scaled bbox {sb} diverged from ShadowBoxer scale_layout_items_to_canvas {py_scaled[idx_b]['bbox']} at {w_px}x{h_px}",
            )
            entry = dict(item)
            entry["bbox"] = sb
            scaled_items.append(entry)
            js_centralized_map[item["label"]] = _js_pixel_to_abs_centralized(
                sb[0], sb[1], sb[2], sb[3], h_px, w_px
            )

        req = GenerationRequest(
            prompt='Studio scene with "SOVEREIGN 2026"',
            prompt_mode="layout_json",
            width=w_px,
            height=h_px,
            layout_items=scaled_items,
            layout_coords_are_pixels=True,
        )
        compiled_json = json.loads(req.compile_effective_prompt())
        for py_item in compiled_json["items"]:
            lbl = py_item["label"]
            _assert(
                lbl in SUPPORTED_BOX_LABELS,
                f"Compiled box label '{lbl}' not in SUPPORTED_BOX_LABELS",
            )
            _assert(
                py_item["bbox"] == js_centralized_map[lbl],
                f"0-ULP JS vs Python abs_centralized bbox mismatch at {w_px}x{h_px} for '{lbl}': JS={js_centralized_map[lbl]} vs Py={py_item['bbox']}",
            )

    return len(_HTML_TO_MODEL_NUMERIC_SPEC)


def _audit_es_module_specifier_graph(
    client: _AsgiClient,
    site_root: Path,
    html_text: str,
    imports_map: Dict[str, Any],
    js_files: List[Path],
) -> int:
    _assert(
        re.search(
            r'<script\s+type="module"\s+src="assets/app\.js\?[^"]+"></script>',
            html_text,
        )
        is not None,
        'index.html must load assets/app.js via <script type="module" src="assets/app.js?v=...">',
    )

    normalized_importmap: Dict[str, str] = {}
    for raw_key, raw_target in imports_map.items():
        norm_key = posixpath.normpath(posixpath.join("/", str(raw_key)))
        target_path_only = str(raw_target).split("?", 1)[0]
        norm_target = posixpath.normpath(posixpath.join("/", target_path_only))
        normalized_importmap[norm_key] = norm_target

    static_import_re = re.compile(
        r'\bimport\s+(?:[\s\S]*?\s+from\s+)?["\'](\.[^"\']+)["\']'
    )
    dynamic_import_re = re.compile(
        r'\bimport\s*\(\s*["\'](\.[^"\']+)["\']\s*\)'
    )

    all_module_files = list(js_files)
    entry_js = (site_root / "assets" / "app.js").resolve()
    _assert(entry_js.is_file(), f"Missing entrypoint {entry_js}")
    if entry_js not in all_module_files:
        all_module_files.append(entry_js)

    verified_edges = 0
    for mod_file in all_module_files:
        rel_posix = "/" + mod_file.relative_to(site_root).as_posix()
        mod_dir = posixpath.dirname(rel_posix)
        src = mod_file.read_text(encoding="utf-8")

        specifiers: List[str] = []
        specifiers.extend(static_import_re.findall(src))
        specifiers.extend(dynamic_import_re.findall(src))
        _assert(
            len(specifiers) > 0 or mod_file.name == "state.js",
            f"Expected ES module imports in {mod_file.name}",
        )

        for spec in specifiers:
            resolved_url = posixpath.normpath(posixpath.join(mod_dir, spec))
            _assert(
                not resolved_url.startswith("/assets/assets/"),
                f"Path-doubled module specifier '{spec}' in {rel_posix} resolved to '{resolved_url}'",
            )
            _assert(
                resolved_url in normalized_importmap,
                f"Module specifier '{spec}' in {rel_posix} resolved to '{resolved_url}', which is not registered in <script type='importmap'> ({list(normalized_importmap.keys())})",
            )
            mapped_target = normalized_importmap[resolved_url]
            code, hdrs, body = client.request("GET", mapped_target)
            _assert(
                code == 200
                and len(body) > 64
                and "no-store" in hdrs.get("cache-control", ""),
                f"ASGI GET on resolved importmap target '{mapped_target}' (from '{spec}' in {rel_posix}) failed: status={code}",
            )
            verified_edges += 1

    return verified_edges


def _audit_webui_js_and_fips_pow(
    site_root: Path, html_text: str, js_files: List[Path], versions_seen: Set[str]
) -> None:
    for asset_pattern in (
        r'href="assets/styles\.css\?([^"]+)"',
        r'src="assets/app\.js\?([^"]+)"',
    ):
        m_ver = re.search(asset_pattern, html_text)
        _assert(
            m_ver is not None,
            f"Missing versioned asset tag matching {asset_pattern} in index.html",
        )
        versions_seen.add(m_ver.group(1))

    _assert(
        len(versions_seen) == 1,
        f"Desynchronized cache-busting versions across index.html & importmap: {versions_seen}",
    )

    all_js_targets = list(js_files)
    entry_js = site_root / "assets" / "app.js"
    if entry_js.is_file():
        all_js_targets.append(entry_js)

    malformed_hex_re = re.compile(
        r"(?<![0-9a-zA-Z_.\"'])0[0-9]+[a-fA-F][0-9a-fA-F]*\b"
    )
    blocking_alert_re = re.compile(r"\balert\s*\(")
    for js_file in all_js_targets:
        src = js_file.read_text(encoding="utf-8")
        bad_lit = malformed_hex_re.search(src)
        _assert(
            bad_lit is None,
            f"Malformed numeric/hex literal '{bad_lit.group(0) if bad_lit else ''}' in {js_file.name}",
        )
        _assert(
            blocking_alert_re.search(src) is None,
            f"Blocking browser alert() found in {js_file.name}; use showStudioToast()",
        )

    api_js_path = site_root / "assets" / "app" / "api.js"
    _assert(api_js_path.is_file(), f"Missing {api_js_path}")
    api_src = api_js_path.read_text(encoding="utf-8")
    _assert(
        "export function sha256Hex(" in api_src
        and "export function minePowTimeSliced(" in api_src
        and "worker.onerror = fallbackToMainThread" in api_src
        and "worker.onmessageerror = fallbackToMainThread" in api_src,
        "api.js is missing first-class sha256Hex, minePowTimeSliced, or worker.onerror fallback",
    )

    h_match = re.search(
        r"let\s+hash\s*=\s*\[([^\]]+)\]", api_src, re.DOTALL
    )
    k_match = re.search(
        r"const\s+k\s*=\s*\[([^\]]+)\]", api_src, re.DOTALL
    )
    _assert(
        h_match is not None and k_match is not None,
        "Failed to extract SHA-256 'hash' or 'k' tables from api.js",
    )

    h_tokens = [
        tok.strip() for tok in h_match.group(1).split(",") if tok.strip()
    ]
    k_tokens = [
        tok.strip() for tok in k_match.group(1).split(",") if tok.strip()
    ]
    _assert(
        len(h_tokens) == 8 and len(k_tokens) == 64,
        f"Expected 8 H0 words and 64 K words in api.js, got {len(h_tokens)} and {len(k_tokens)}",
    )

    for idx, tok in enumerate(h_tokens):
        _assert(
            tok.lower().startswith("0x"),
            f"Non-hex H0 literal at index {idx} in api.js: {tok}",
        )
        val = int(tok, 16)
        _assert(
            val == _FIPS_180_4_H0[idx],
            f"FIPS 180-4 H0[{idx}] mismatch in api.js: got 0x{val:08x}, expected 0x{_FIPS_180_4_H0[idx]:08x}",
        )

    for idx, tok in enumerate(k_tokens):
        _assert(
            tok.lower().startswith("0x"),
            f"Non-hex K[{idx}] literal in api.js: {tok}",
        )
        val = int(tok, 16)
        _assert(
            val == _FIPS_180_4_K[idx],
            f"FIPS 180-4 K[{idx}] mismatch in api.js: got 0x{val:08x}, expected 0x{_FIPS_180_4_K[idx]:08x}",
        )

    extracted_h = [int(t, 16) for t in h_tokens]
    extracted_k = [int(t, 16) for t in k_tokens]
    for sample in ("", "abc", "1730419200:deadbeefcafe1234:4:0"):
        sim_digest = _simulate_extracted_js_sha256(
            sample, extracted_h, extracted_k
        )
        ref_digest = hashlib.sha256(sample.encode("ascii")).hexdigest()
        _assert(
            sim_digest == ref_digest,
            f"Simulated JS SHA-256 mismatch for '{sample}': {sim_digest} != {ref_digest}",
        )


def _audit_webui_dom_and_importmap_contract(
    client: _AsgiClient,
) -> Tuple[int, int, int, int]:
    site_root = resolve_site_root()
    idx_file = site_root / "index.html"
    if not idx_file.is_file():
        return 0, 0, 0, 0

    html_text = idx_file.read_text(encoding="utf-8")
    html_ids: Set[str] = set(re.findall(r'\bid="([^"]+)"', html_text))
    _assert(
        len(html_ids) >= 96 and "addCharacterBoxBtn" in html_ids,
        f"Expected >= 96 DOM element IDs (including addCharacterBoxBtn) in {idx_file}, found {len(html_ids)}",
    )

    im_match = re.search(
        r'<script\s+type="importmap">\s*(\{.*?\})\s*</script>',
        html_text,
        re.DOTALL,
    )
    _assert(im_match is not None, f'Missing <script type="importmap"> in {idx_file}')
    im_data = json.loads(im_match.group(1))
    imports_map = im_data.get("imports", {})
    _assert(
        len(imports_map) >= 8,
        f"Expected >= 8 ES module entries in importmap, found {len(imports_map)}",
    )

    versions_seen: Set[str] = set()
    js_files: List[Path] = []
    for mod_key, mod_target in imports_map.items():
        clean_rel = str(mod_target).split("?", 1)[0].lstrip("./")
        if "?" in str(mod_target):
            versions_seen.add(str(mod_target).split("?", 1)[1])
        mod_path = (site_root / clean_rel).resolve()
        _assert(
            mod_path.is_file(),
            f"Importmap module '{mod_key}' -> '{mod_target}' missing on disk at {mod_path}",
        )
        js_files.append(mod_path)

    referenced_ids: Set[str] = set()
    for js_file in js_files:
        js_src = js_file.read_text(encoding="utf-8")
        for m in re.finditer(r'\bel\(\s*["\']([a-zA-Z0-9_-]+)["\']\s*\)', js_src):
            ref_id = m.group(1)
            referenced_ids.add(ref_id)
            _assert(
                ref_id in html_ids,
                f"Orphan DOM selector el('{ref_id}') in {js_file.name} not found in {idx_file.name}",
            )

    _audit_webui_js_and_fips_pow(
        site_root, html_text, js_files, versions_seen
    )
    num_edges = _audit_es_module_specifier_graph(
        client, site_root, html_text, imports_map, js_files
    )
    num_numeric_specs = _audit_cross_stack_numeric_and_box_schema_supremacy(
        site_root, html_text
    )

    return len(js_files), len(referenced_ids), num_numeric_specs, num_edges


def gate_5_queue_etag_and_spa_guard(client: _AsgiClient, token: str) -> None:
    print(
        "\n--- Gate 5: Cache-Busting, RFC 3986 Importmap, FIPS PoW & Cross-Stack Numeric Audit ---"
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

    expected_ver = compute_webui_asset_version()
    idx_code, idx_hdrs, idx_bytes = client.request(
        "GET",
        "/",
        headers={"If-None-Match": '"stale-webview-etag"'},
    )
    idx_html_served = idx_bytes.decode("utf-8")
    _assert(
        idx_code == 200
        and idx_hdrs.get("x-visionstack-asset-version") == expected_ver
        and "no-store" in idx_hdrs.get("cache-control", "")
        and "etag" not in idx_hdrs
        and f"?v={expected_ver}" in idx_html_served,
        f"Root index.html failed dynamic SHA-256 version stamping or no-store enforcement: status={idx_code}, hdrs={idx_hdrs}",
    )

    for static_asset in ("/assets/app.js", "/assets/app/api.js", "/assets/styles.css"):
        st_code, st_hdrs, _ = client.request(
            "GET",
            static_asset,
            headers={
                "If-None-Match": '"stale-etag"',
                "If-Modified-Since": "Wed, 21 Oct 2099 07:28:00 GMT",
            },
        )
        _assert(
            st_code == 200
            and "no-store" in st_hdrs.get("cache-control", "")
            and "etag" not in st_hdrs,
            f"Static asset {static_asset} must return 200 OK with no-store (never 304): status={st_code}, hdrs={st_hdrs}",
        )

    for forbidden_path in (
        "/ShadowBoxer/engine.py",
        "/services/router.py",
        "/services/runtime/security.py",
        "/storage/users/administrator/history.json",
    ):
        f_code, _, _ = client.request("GET", forbidden_path)
        _assert(
            f_code in (403, 404),
            f"Forbidden backend path '{forbidden_path}' was not blocked (status={f_code})",
        )

    num_modules, num_dom_ids, num_numeric, num_edges = (
        _audit_webui_dom_and_importmap_contract(client)
    )

    readme_file = generate_services_readme()
    _assert(
        readme_file.is_file() and readme_file.stat().st_size > 1024,
        "services/README.md generation failed",
    )
    print(
        f"  [+] Full-Stack Supremacy Audit    : Bundle #{expected_ver} + {num_edges} RFC3986 Module Edges + 64/64 FIPS SHA-256 + {num_numeric} Numeric Specs + {num_modules} ES Modules ({num_dom_ids}/{num_dom_ids} DOM IDs) Verified"
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

    _close_preview_tokenizer()
    if getattr(compute_queue.pipeline, "_memory_guard", None) is not None:
        try:
            compute_queue.pipeline._memory_guard.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 92)
    print(
        f"[+] STATUS: SERVICES ROUTER SECURE, ROBUST & CONVERGED IN {elapsed:.2f}s. ZERO DRIFT ACROSS ALL 5 GATES."
    )
    print("=" * 92)


if __name__ == "__main__":
    main()