#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def generate_services_readme() -> Path:
    bt = chr(96) * 3
    sbt = chr(96)
    tools_dir = Path(__file__).resolve().parent
    services_dir = tools_dir.parent
    readme_path = services_dir / "README.md"

    content = f"""# VisionStack Unified Services & Spatio-Visual Compute Daemon Routing Layer

The **VisionStack Service Layer** ({sbt}services/router.py{sbt} & {sbt}services/runtime/*{sbt}) orchestrates the single-consumer spatio-visual synthesis pipeline, proxy-hardened proof-of-work bot defense, HMAC session management, bifurcated per-user reference and project vaults (`.vsproj`), and dual-mode deployment architectures (Caddy/Nginx reverse proxy and hermetic standalone desktop environments).

## Modular Runtime Architecture ({sbt}services/runtime/{sbt})

* **{sbt}services/router.py{sbt}**: Lightweight CLI and ASGI entrypoint delegating to {sbt}services.runtime{sbt}.
* **{sbt}services/runtime/env.py{sbt}**: Path anchors, deterministic CUDA/ROCm flags, site-root discovery, user registry loader, and thread-safe {sbt}MOSSImageTokenizer{sbt} ({sbt}VizToken{sbt} SWISS5) preview singleton.
* **{sbt}services/runtime/security.py{sbt}**: Persistent 256-bit CSPRNG secret bootstrap ({sbt}config/.runtime_secrets.json{sbt}), trusted reverse-proxy client IP extraction ({sbt}X-Forwarded-For{sbt}, {sbt}X-Real-IP{sbt}, {sbt}CF-Connecting-IP{sbt}), sliding-window IP rate limiter, load-adaptive PoW difficulty scaling, minimum difficulty floor enforcement, and bounded LRU single-burn challenge nonce cache.
* **{sbt}services/runtime/models.py{sbt}**: Validated Pydantic wire schemas ({sbt}SynthesisPayload{sbt} up to 32K context length, {sbt}PromptPreviewPayload{sbt}, {sbt}ProjectSavePayload{sbt}, {sbt}ReferenceUploadPayload{sbt}).
* **{sbt}services/runtime/vault.py{sbt}**: Binary {sbt}.vizj{sbt} header & RGBA alpha-matte probing, path-traversal-guarded vault resolution, and SHA-256 deduplicated {sbt}.vsproj{sbt} archive packaging and self-healing extraction.
* **{sbt}services/runtime/queue.py{sbt}**: {sbt}SynthesisJob{sbt}, {sbt}EnginePipeline{sbt}, and {sbt}ComputeQueue{sbt} with atomic in-flight job accounting ({sbt}QUEUED{sbt} + {sbt}PROCESSING{sbt}) to prevent TOCTOU quota bypasses.
* **{sbt}services/runtime/routes.py{sbt}**: FastAPI application factory, COOP/COEP/CORP security middleware, multi-prefix API router, and SPA static mount guard.

## Architecture & Topology

{bt}text
+-----------------------------------------------------------------------------+
|                            Client Ingress Layer                             |
|  [Caddy Gateway: Reverse Proxy]     OR     [Native Standalone: PyWebView]   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                VisionStack FastAPI Daemon (services/runtime/)               |
|                                                                             |
|  1. Security Middleware   : COOP (same-origin), COEP (require-corp)         |
|  2. Proxy PoW & Rate Guard: IP-Bound HMAC, Single-Burn Nonce & Adaptive Diff|
|  3. Multi-Prefix Router   : /v1, /api/v1, /VisionStack/api/v1, /ShadowBoxer |
|  4. Async Compute Queue   : Serial Worker + Atomic In-Flight Quota Lock     |
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
|  - User Image Vault   : storage/users/{{slug}}/images/{{job_id}}_master.png     |
|  - User Upload Vault  : storage/users/{{slug}}/uploads/{{ref_id}}.png|.vizj    |
|  - User Project Vault : storage/users/{{slug}}/projects/{{proj_id}}.vsproj     |
|  - Gallery State      : storage/users/{{slug}}/history.json                   |
|  - Ephemeral Outputs  : artifacts/{{job_id}}.png|.vizj (24h TTL Auto-Prune)  |
+-----------------------------------------------------------------------------+
{bt}

## Core System Invariants

* **Single-Consumer Queue & Warm Pool Retention**: GPU operations across {sbt}MOSSImageEngine{sbt} are strictly serialized through {sbt}ComputeQueue{sbt}, retaining {sbt}MOSS-DiT{sbt}, {sbt}Qwen3.5{sbt}, and {sbt}Flux2-VAE{sbt} weights plus the pre-warmed HIP activation pool in VRAM while enforcing capacity-adaptive {sbt}WDDMMemoryGuard{sbt} headroom budgets.
* **Authoritative Point-of-Truth Preset Sync**: Automatically inherits optimal baseline parameters ({sbt}V03{sbt} {sbt}Heun -> IPNDM{sbt}, {sbt}CFG 3.2 -> 2.8{sbt}, layer {sbt}[24]{sbt} skip, pre-staged SLG layer {sbt}[24]{sbt} with {sbt}slg_scale = 0.0{sbt}, {sbt}max_sequence_length = 16384{sbt}) from {sbt}ShadowBoxer/schema.py{sbt} and hot-reloads {sbt}ShadowBoxer/default.json{sbt} overrides via {sbt}mtime{sbt} inspection.
* **Collaborative Spatio-Visual, Speculative CoT & Mask Algebra Graph**: Full routing support for permutation-invariant coarse-to-fine canonical box sorting, collaborative IsoKV attention across overlapping bounding boxes, regional box-aperture attention isolation ({sbt}box_attn_isolation{sbt}), commutative normalized feather-weighted latent canvas splatting ({sbt}composite_box_images_to_init{sbt}), 6-mode composable latent inpaint mask algebra ({sbt}auto{sbt}, {sbt}union{sbt}, {sbt}intersect{sbt}, {sbt}paint_only{sbt}, {sbt}boxes_only{sbt}, {sbt}alpha_only{sbt} + {sbt}invert_inpaint_mask{sbt}), unconditional reference gating ({sbt}uncond_uses_references{sbt}), explicit {sbt}reference_zero_timestep{sbt} / {sbt}reference_isolated_kv{sbt} overrides, and Qwen3.5 ChatML / CoT reasoning modes ({sbt}system_prompt{sbt}, {sbt}thinking_trace="auto"{sbt}, {sbt}thinking_trace="speculative:N"{sbt}, custom trace, {sbt}add_vision_id{sbt}).
* **Proxy-Hardened Proof-of-Work & Atomic In-Flight Quota Enforcement**:
  * **Persistent CSPRNG Secrets**: Automatically provisions 256-bit random HMAC secrets in {sbt}config/.runtime_secrets.json{sbt} when environment variables are unset.
  * **Client-IP Bound & Single-Burn Challenges**: Binds PoW HMAC signatures to the resolved reverse-proxy client IP, enforces the minimum difficulty floor, and burns each challenge {sbt}nonce{sbt} upon its first valid redemption so no challenge can be reused with a second mined solution nonce.
  * **TOCTOU Quota & Concurrency Lock**: {sbt}POST /synthesize{sbt} accounts for both completed daily syntheses and active in-flight ({sbt}QUEUED{sbt} + {sbt}PROCESSING{sbt}) jobs, enforcing per-user concurrency caps ({sbt}MAX_USER_INFLIGHT_JOBS = 4{sbt}) and global queue depth bounds ({sbt}MAX_GLOBAL_QUEUE_DEPTH = 32{sbt}).

## Endpoints Specification

| Method | Route | Description |
| :--- | :--- | :--- |
| **GET** | {sbt}/auth/challenge{sbt} | Issues IP-bound, load-adaptive PoW challenge with difficulty target and HMAC signature |
| **POST** | {sbt}/auth/login{sbt} | Authenticates creator handle, computes daily + in-flight quota, returns token and gallery |
| **GET** | {sbt}/themes/registry{sbt} | Serves registered studio theme palettes and UI manifests |
| **GET** | {sbt}/health{sbt} | Reports active compute device, WDDM VRAM telemetry, available models, and baseline parameters |
| **POST** | {sbt}/prompt/preview{sbt} | Compiles effective Qwen3.5 layout JSON prompt, CoT trace, and exact {sbt}VizToken{sbt} BPE horizon counts |
| **POST** | {sbt}/presets/default{sbt} | Updates authoritative {sbt}ShadowBoxer/default.json{sbt} preset (Admin only) |
| **POST** | {sbt}/references/upload{sbt} | Stages base64 PNG/WebP or {sbt}.VIZJ{sbt} reference asset into user vault |
| **POST** | {sbt}/references/from-image/{{image_id}}{sbt} | Promotes a synthesized gallery image into the user's reference vault with SHA-256 deduplication |
| **POST** | {sbt}/references/stage-defaults{sbt} | Stages deterministic test reference, init, mask, and control assets into user vault |
| **GET** | {sbt}/references{sbt} | Lists staged reference assets in the authenticated user's vault |
| **GET** | {sbt}/references/stream/{{user_slug}}/{{filename}}{sbt} | Streams a staged reference image or {sbt}.VIZJ{sbt} bundle from user vault |
| **DELETE** | {sbt}/references/{{ref_id}}{sbt} | Deletes a staged reference asset from the authenticated user's vault |
| **POST** | {sbt}/projects/save{sbt} | Packages working draft, layout boxes, painted mask, and bound assets into a `.vsproj` archive |
| **GET** | {sbt}/projects{sbt} | Lists saved `.vsproj` studio projects in the authenticated user's project vault |
| **POST** | {sbt}/projects/{{project_id}}/load{sbt} | Extracts a saved `.vsproj` archive, self-heals missing vault assets, and returns the studio state |
| **POST** | {sbt}/projects/import{sbt} | Imports an external `.vsproj` archive into the user's project and asset vaults |
| **GET** | {sbt}/projects/{{project_id}}/download{sbt} | Streams a portable `.vsproj` archive for local backup or sharing |
| **GET** | {sbt}/projects/stream/{{user_slug}}/{{filename}}{sbt} | Streams a saved project preview thumbnail from the user's project vault |
| **DELETE** | {sbt}/projects/{{project_id}}{sbt} | Deletes a saved `.vsproj` project package and metadata from the user's vault |
| **POST** | {sbt}/synthesize{sbt} | Validates IP-bound PoW solution & in-flight quota, admits synthesis job to {sbt}ComputeQueue{sbt} |
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