#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SERVICES_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SERVICES_DIR.parent
for _p in (SERVICES_DIR, BACKEND_ROOT):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from services.runtime import (
    ARTIFACTS_DIR,
    ARTIFACT_TTL_SECONDS,
    AuthPayload,
    BASELINE_ENGINE_DEFAULTS,
    CONFIG_DIR,
    ComputeQueue,
    DEFAULT_LAYOUT_ITEMS,
    DEFAULT_PRESET_FILENAME,
    DEFAULT_SPECULATIVE_COT_TOKENS,
    EnginePipeline,
    GenerationRequest,
    MAX_GLOBAL_QUEUE_DEPTH,
    MAX_REPLAY_CACHE_SIZE,
    MAX_USER_INFLIGHT_JOBS,
    POW_DIFFICULTY,
    POW_EXPIRATION_SECONDS,
    POW_SECRET,
    PollingLogFilter,
    PowSubmission,
    PresetUpdatePayload,
    ProjectImportPayload,
    ProjectSavePayload,
    PromoteImagePayload,
    PromptPreviewPayload,
    RUNTIME_DIR,
    ReferenceUploadPayload,
    SESSION_SECRET,
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
    SecurityHeadersMiddleware,
    SynthesisJob,
    SynthesisPayload,
    _FORBIDDEN_SPA_SEGMENTS,
    _NO_CACHE_HEADERS,
    _RATE_LIMIT_BUCKETS,
    _REPLAY_CACHE,
    _build_vsproj_package,
    _close_preview_tokenizer,
    _collect_draft_asset_names,
    _decode_data_url_bytes,
    _extract_and_restore_vsproj,
    _get_preview_tokenizer,
    _inspect_reference_file,
    _probe_available_models,
    _probe_vizj_dimensions,
    _resolve_user_upload_path,
    api_router,
    app,
    check_rate_limit,
    compute_queue,
    compute_webui_asset_version,
    create_session_token,
    generate_pow_challenge,
    get_active_engine_defaults,
    get_runtime_override,
    get_themes_root,
    get_user_registry_candidates,
    has_custom_default_preset,
    is_standalone_mode,
    launch_standalone,
    lifespan,
    load_user_registry,
    mount_static_and_spa,
    parse_layer_list,
    render_versioned_index_html,
    resolve_client_ip,
    resolve_site_root,
    run_server,
    set_cli_overrides,
    slugify,
    verify_pow_solution,
    verify_session_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionStack Unified Services & Compute Daemon Router"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host interface to bind"
    )
    parser.add_argument(
        "--port", type=int, default=8766, help="Port to bind"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Launch native desktop window",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run strictly as headless daemon",
    )
    parser.add_argument(
        "--site-root",
        type=str,
        default=None,
        help="Explicit path to site/webui assets directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit path to users.json file",
    )
    args = parser.parse_args()

    set_cli_overrides(
        standalone=(
            args.standalone
            if args.standalone
            else (False if args.headless else None)
        ),
        site_root=args.site_root,
        config_file=args.config,
    )

    if is_standalone_mode():
        launch_standalone(args.host, args.port)
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()