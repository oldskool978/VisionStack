#!/usr/bin/env python3
# services/tools/readmegen.py
from __future__ import annotations

from pathlib import Path


def generate_services_readme() -> Path:
    bt = chr(96) * 3
    sbt = chr(96)

    tools_dir = Path(__file__).resolve().parent
    services_dir = tools_dir.parent
    readme_path = services_dir / "README.md"

    content = f"""# VisionStack Unified Services & Spatio-Visual Compute Daemon Routing Layer

The **VisionStack Service Router** ({sbt}services/router.py{sbt}) orchestrates the single-consumer spatio-visual synthesis pipeline, cryptographic proof-of-work bot defense, HMAC session management, bifurcated per-user reference vaults, and dual-mode deployment architectures (Caddy/Nginx reverse proxy and hermetic standalone desktop environments).

## Architecture & Topology

{bt}text
+-----------------------------------------------------------------------------+
|                            Client Ingress Layer                             |
|  [Caddy Gateway: Reverse Proxy]     OR     [Native Standalone: PyWebView]   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                       VisionStack FastAPI Daemon                            |
|                                                                             |
|  1. Security Middleware   : COOP (same-origin), COEP (require-corp)         |
|  2. PoW Guard Engine      : SHA-256 Nonce Verification & Replay Protection  |
|  3. Multi-Prefix Router   : /v1, /api/v1, /VisionStack/api/v1, /ShadowBoxer |
|  4. Async Compute Queue   : Strict Serial Single-Job Worker Synchronization |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                 Sovereign ShadowBoxer Accelerator Pipeline                  |
|                                                                             |
|  [Stage 0: Vision Conditioning] --> VizInjector (Vulkan GLSL 450 + .VIZJ)   |
|  [Stage 1: Hybrid Context]      --> VizToken SWISS5 + Qwen3.5 (32L + ViT)   |
|  [Stage 2: Velocity Integration]--> 42-Layer MOSS-DiT (Heun -> IPNDM 4D)    |
|  [Stage 3: Radiometric Decode]  --> Flux2-VAE FP32 + Ordered Bayer Dither   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                            Persistence Engine                               |
|  - User Image Vault  : storage/users/{{slug}}/images/{{job_id}}_master.png      |
|  - User Upload Vault : storage/users/{{slug}}/uploads/{{ref_id}}.png|.vizj     |
|  - Gallery State     : storage/users/{{slug}}/history.json                    |
|  - Ephemeral Outputs : artifacts/{{job_id}}.png|.vizj (24h TTL Auto-Prune)   |
+-----------------------------------------------------------------------------+
{bt}

## Core System Invariants

* **Single-Consumer Queue & Warm Pool Retention**: GPU operations across {sbt}MOSSImageEngine{sbt} are strictly serialized through {sbt}ComputeQueue{sbt}, retaining {sbt}MOSS-DiT{sbt}, {sbt}Qwen3.5{sbt}, and {sbt}Flux2-VAE{sbt} weights plus the pre-warmed HIP activation pool in VRAM while enforcing capacity-adaptive {sbt}WDDMMemoryGuard{sbt} headroom budgets.
* **Authoritative Point-of-Truth Preset Sync**: Automatically inherits optimal baseline parameters ({sbt}V03{sbt} {sbt}Heun -> IPNDM{sbt}, {sbt}CFG 3.2 -> 2.8{sbt}, layer {sbt}[24]{sbt} skip) from {sbt}ShadowBoxer/schema.py{sbt} and hot-reloads {sbt}ShadowBoxer/default.json{sbt} overrides via {sbt}mtime{sbt} inspection.
* **Spatio-Visual & ChatML Conditioning Graph**: Full routing support for regional box-aperture attention isolation ({sbt}box_attn_isolation{sbt}), regional latent canvas splatting ({sbt}composite_box_images_to_init{sbt}), unconditional reference gating ({sbt}uncond_uses_references{sbt}), and Qwen3.5 ChatML controls ({sbt}system_prompt{sbt}, {sbt}thinking_trace{sbt}, {sbt}add_vision_id{sbt}).
* **Bifurcated Per-User Reference Vault**: User-uploaded reference photos, inpaint masks, control maps, and {sbt}.vizj{sbt} containers are isolated under {sbt}storage/users/{{slug}}/uploads/{sbt} with binary header dimension probing and strict path-traversal guards.
* **Dual Gateway Route Parity**: Uniform API binding across {sbt}/api/v1{sbt}, {sbt}/v1{sbt}, {sbt}/VisionStack/api/v1{sbt}, and {sbt}/ShadowBoxer/api/v1{sbt} guarantees identical behavior between reverse-proxy subpaths and local standalone WebViews.
* **Proof-of-Work Bot Mitigation**: Cryptographically signed HMAC-SHA256 challenges with sliding-window replay caching validate client CPU proof prior to queue admission.

## Endpoints Specification

| Method | Route | Description |
| :--- | :--- | :--- |
| **GET** | {sbt}/auth/challenge{sbt} | Issues dynamic PoW challenge with difficulty target and HMAC signature |
| **POST** | {sbt}/auth/login{sbt} | Authenticates creator handle, computes daily quota, returns token and gallery |
| **GET** | {sbt}/themes/registry{sbt} | Serves registered studio theme palettes and UI manifests |
| **GET** | {sbt}/health{sbt} | Reports active compute device, WDDM VRAM telemetry, and baseline parameters |
| **POST** | {sbt}/presets/default{sbt} | Updates authoritative {sbt}ShadowBoxer/default.json{sbt} preset (Admin only) |
| **POST** | {sbt}/references/upload{sbt} | Stages base64 PNG/WebP or {sbt}.VIZJ{sbt} reference asset into user vault |
| **POST** | {sbt}/references/stage-defaults{sbt} | Stages deterministic test reference, init, mask, and control assets into user vault |
| **GET** | {sbt}/references{sbt} | Lists staged reference assets in the authenticated user's vault |
| **GET** | {sbt}/references/stream/{{user_slug}}/{{filename}}{sbt} | Streams a staged reference image or {sbt}.VIZJ{sbt} bundle from user vault |
| **DELETE** | {sbt}/references/{{ref_id}}{sbt} | Deletes a staged reference asset from the authenticated user's vault |
| **POST** | {sbt}/synthesize{sbt} | Validates PoW solution, admits synthesis job to {sbt}ComputeQueue{sbt} |
| **GET** | {sbt}/jobs/{{job_id}}{sbt} | Polls real-time progress, stage description, queue position, and ETag |
| **GET** | {sbt}/jobs/{{job_id}}/stream{sbt} | Streams live Server-Sent Events (SSE) progress and completion telemetry |
| **GET** | {sbt}/images/{{job_id}}{sbt} | Direct download stream for ephemeral master PNG artifacts |
| **GET** | {sbt}/images/stream/{{user_slug}}/{{filename}}{sbt} | Serves persisted master PNG or {sbt}.VIZJ{sbt} bundles from user gallery vault |
| **DELETE** | {sbt}/images/{{image_id}}{sbt} | Deletes persisted master image and gallery record from user vault |

## Execution Modes

### 1. Standalone Desktop Mode
Launches the native desktop interface with an embedded hardware-accelerated WebView:
{bt}bash
python services/router.py --standalone
{bt}

### 2. Headless Daemon Mode (Proxy Backend)
Runs the unified ASGI daemon bound to localhost for Caddy or Nginx reverse-proxy ingress:
{bt}bash
python services/router.py --host 127.0.0.1 --port 8766 --headless
{bt}
"""

    readme_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"[+] Successfully generated services documentation: {readme_path}")
    return readme_path


if __name__ == "__main__":
    generate_services_readme()