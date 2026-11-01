from __future__ import annotations

import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import hashlib
import json
import logging
import mimetypes
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("application/octet-stream", ".vizj")
mimetypes.add_type("application/octet-stream", ".vsproj")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

RUNTIME_DIR = Path(__file__).resolve().parent
SERVICES_DIR = RUNTIME_DIR.parent
BACKEND_ROOT = SERVICES_DIR.parent
SHADOWBOXER_DIR = BACKEND_ROOT / "ShadowBoxer"

PATH_ANCHORS = [
    BACKEND_ROOT,
    SHADOWBOXER_DIR,
    SERVICES_DIR,
]
for _p in reversed(PATH_ANCHORS):
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

for _pkg_dir in [
    SHADOWBOXER_DIR,
    SHADOWBOXER_DIR / "models",
    SHADOWBOXER_DIR / "viztoken",
    SHADOWBOXER_DIR / "vizinjector",
    SERVICES_DIR,
    RUNTIME_DIR,
]:
    if _pkg_dir.exists():
        _init_file = _pkg_dir / "__init__.py"
        if not _init_file.exists():
            try:
                _init_file.touch()
            except OSError:
                pass

try:
    import ShadowBoxer.schema as shadowboxer_schema_mod

    sys.modules["schema"] = shadowboxer_schema_mod
except Exception:
    import schema as shadowboxer_schema_mod

    sys.modules["ShadowBoxer.schema"] = shadowboxer_schema_mod

from ShadowBoxer.schema import (
    BASELINE_ENGINE_DEFAULTS,
    DEFAULT_LAYOUT_ITEMS,
    DEFAULT_PRESET_FILENAME,
    DEFAULT_SPECULATIVE_COT_TOKENS,
    GenerationRequest,
    SUPPORTED_BOX_LABELS,
    SUPPORTED_EDIT_FORMATS,
    SUPPORTED_FRAMING_MODES,
    SUPPORTED_INPUT_ROLES,
    SUPPORTED_MODELS,
    SUPPORTED_PROMPT_MODES,
    SUPPORTED_SOLVERS,
    get_active_engine_defaults,
    has_custom_default_preset,
    parse_layer_list,
)

SUPPORTED_MASK_BLEND_MODES: List[str] = list(
    getattr(
        shadowboxer_schema_mod,
        "SUPPORTED_MASK_BLEND_MODES",
        ["auto", "union", "intersect", "paint_only", "boxes_only", "alpha_only"],
    )
)

ARTIFACTS_DIR = BACKEND_ROOT / "artifacts"
STORAGE_ROOT = BACKEND_ROOT / "storage" / "users"
CONFIG_DIR = BACKEND_ROOT / "config"

for _d in (ARTIFACTS_DIR, STORAGE_ROOT, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ARTIFACT_TTL_SECONDS = 86400

_IS_STANDALONE: bool = os.environ.get("VISIONSTACK_STANDALONE", "0").lower() in (
    "1",
    "true",
    "yes",
)
_CLI_SITE_ROOT: Optional[Path] = None
_CLI_CONFIG_FILE: Optional[Path] = None

_FORBIDDEN_SPA_SEGMENTS = {
    "..",
    "services",
    "shadowboxer",
    "vizinjector",
    "viztoken",
    "storage",
    "config",
    "artifacts",
    "tools",
}

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

_PREVIEW_TOKENIZER_LOCK = threading.Lock()
_PREVIEW_TOKENIZER: Optional[Any] = None

_ASSET_VERSION_LOCK = threading.Lock()
_ASSET_VERSION_CACHE: Dict[str, Any] = {
    "stat_sig": None,
    "version": "20261031.2",
    "rendered_html": b"",
}
_VERSION_QUERY_RE = re.compile(r"(\.(?:js|css)\?v=)[0-9a-zA-Z._-]+")


def get_runtime_override(attr_name: str, default_val: Any) -> Any:
    for mod_name in ("services.router", "router", "services.runtime"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr_name):
            return getattr(mod, attr_name)
    return default_val


def _get_preview_tokenizer() -> Optional[Any]:
    global _PREVIEW_TOKENIZER
    if _PREVIEW_TOKENIZER is not None:
        return _PREVIEW_TOKENIZER
    with _PREVIEW_TOKENIZER_LOCK:
        if _PREVIEW_TOKENIZER is not None:
            return _PREVIEW_TOKENIZER
        try:
            from ShadowBoxer.models.tokenizer import MOSSImageTokenizer
        except Exception:
            try:
                from models.tokenizer import MOSSImageTokenizer
            except Exception:
                return None
        try:
            sb_dir = Path(get_runtime_override("SHADOWBOXER_DIR", SHADOWBOXER_DIR))
            _PREVIEW_TOKENIZER = MOSSImageTokenizer(sb_dir / "viztoken")
        except Exception:
            _PREVIEW_TOKENIZER = None
        return _PREVIEW_TOKENIZER


def _close_preview_tokenizer() -> None:
    global _PREVIEW_TOKENIZER
    with _PREVIEW_TOKENIZER_LOCK:
        if _PREVIEW_TOKENIZER is not None:
            try:
                _PREVIEW_TOKENIZER.close()
            except Exception:
                pass
            _PREVIEW_TOKENIZER = None


def _bootstrap_cli_environment() -> None:
    global _IS_STANDALONE, _CLI_SITE_ROOT, _CLI_CONFIG_FILE
    for i, arg in enumerate(sys.argv):
        if arg == "--site-root" and i + 1 < len(sys.argv):
            _CLI_SITE_ROOT = Path(sys.argv[i + 1]).resolve()
        elif arg.startswith("--site-root="):
            _CLI_SITE_ROOT = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--config" and i + 1 < len(sys.argv):
            _CLI_CONFIG_FILE = Path(sys.argv[i + 1]).resolve()
        elif arg.startswith("--config="):
            _CLI_CONFIG_FILE = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--standalone":
            _IS_STANDALONE = True
            os.environ["VISIONSTACK_STANDALONE"] = "1"
        elif arg == "--headless":
            _IS_STANDALONE = False
            os.environ["VISIONSTACK_STANDALONE"] = "0"


_bootstrap_cli_environment()


def set_cli_overrides(
    standalone: Optional[bool] = None,
    site_root: Optional[str] = None,
    config_file: Optional[str] = None,
) -> None:
    global _IS_STANDALONE, _CLI_SITE_ROOT, _CLI_CONFIG_FILE
    if standalone is not None:
        _IS_STANDALONE = standalone
        os.environ["VISIONSTACK_STANDALONE"] = "1" if standalone else "0"
    if site_root:
        _CLI_SITE_ROOT = Path(site_root).resolve()
    if config_file:
        _CLI_CONFIG_FILE = Path(config_file).resolve()


def is_standalone_mode() -> bool:
    return _IS_STANDALONE or os.environ.get(
        "VISIONSTACK_STANDALONE", "0"
    ).lower() in ("1", "true", "yes")


def resolve_site_root() -> Path:
    if is_standalone_mode():
        candidates = [
            BACKEND_ROOT / "webui",
            Path.cwd() / "webui",
        ]
        for c in candidates:
            if (c / "index.html").exists():
                return c.resolve()
        return (BACKEND_ROOT / "webui").resolve()

    if _CLI_SITE_ROOT and _CLI_SITE_ROOT.exists():
        return _CLI_SITE_ROOT.resolve()
    env_site = os.environ.get("VISIONSTACK_SITE_DIR")
    if env_site and Path(env_site).exists():
        return Path(env_site).resolve()
    candidates = [
        BACKEND_ROOT / "webui",
        Path.cwd() / "webui",
        BACKEND_ROOT / "site",
        Path.cwd() / "site",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c.resolve()
    return (BACKEND_ROOT / "webui").resolve()


def _refresh_webui_asset_bundle() -> Tuple[str, bytes]:
    site_root = resolve_site_root()
    idx_file = site_root / "index.html"
    if not idx_file.is_file():
        return "20261031.2", b""

    tracked_files: List[Path] = [idx_file]
    assets_dir = site_root / "assets"
    if assets_dir.is_dir():
        for pat in ("*.css", "*.js", "app/*.js"):
            tracked_files.extend(sorted(assets_dir.glob(pat)))

    stat_entries: List[Tuple[str, int, int]] = []
    for f in tracked_files:
        if f.is_file():
            st = f.stat()
            stat_entries.append((str(f), int(st.st_mtime_ns), int(st.st_size)))

    stat_sig = tuple(stat_entries)
    with _ASSET_VERSION_LOCK:
        if (
            _ASSET_VERSION_CACHE["stat_sig"] == stat_sig
            and _ASSET_VERSION_CACHE["rendered_html"]
        ):
            return (
                str(_ASSET_VERSION_CACHE["version"]),
                bytes(_ASSET_VERSION_CACHE["rendered_html"]),
            )

        hasher = hashlib.sha256()
        for f, _, _ in stat_entries:
            try:
                hasher.update(Path(f).read_bytes())
            except OSError:
                pass
        digest_ver = hasher.hexdigest()[:12]

        raw_html = idx_file.read_text(encoding="utf-8")
        stamped_html = _VERSION_QUERY_RE.sub(rf"\g<1>{digest_ver}", raw_html)
        rendered_bytes = stamped_html.encode("utf-8")

        _ASSET_VERSION_CACHE["stat_sig"] = stat_sig
        _ASSET_VERSION_CACHE["version"] = digest_ver
        _ASSET_VERSION_CACHE["rendered_html"] = rendered_bytes
        return digest_ver, rendered_bytes


def compute_webui_asset_version() -> str:
    ver, _ = _refresh_webui_asset_bundle()
    return ver


def render_versioned_index_html() -> Tuple[bytes, str]:
    ver, html_bytes = _refresh_webui_asset_bundle()
    return html_bytes, ver


def get_themes_root() -> Path:
    return resolve_site_root() / "themes"


class PollingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if '"GET ' in msg:
            if "/jobs/" in msg or "/health" in msg:
                if (
                    " 200 " in msg
                    or " 304 " in msg
                    or " 200 OK" in msg
                    or " 304 Not Modified" in msg
                ):
                    return False
        return True


def slugify(username: str) -> str:
    slug = username.strip().lower()
    slug = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in slug])
    return "_".join(filter(None, slug.split("_"))) or "user"


def get_user_registry_candidates() -> List[Path]:
    if _CLI_CONFIG_FILE and _CLI_CONFIG_FILE.exists():
        return [_CLI_CONFIG_FILE.resolve()]
    custom_file = os.environ.get("VISIONSTACK_USERS_FILE")
    if custom_file and Path(custom_file).exists():
        return [Path(custom_file).resolve()]
    custom_dir = os.environ.get("VISIONSTACK_CONFIG_DIR")
    if custom_dir and (Path(custom_dir) / "users.json").exists():
        return [(Path(custom_dir) / "users.json").resolve()]
    resolved_site = resolve_site_root()
    return [
        resolved_site / "config" / "users.json",
        BACKEND_ROOT / "config" / "users.json",
        BACKEND_ROOT / "webui" / "config" / "users.json",
        SERVICES_DIR / "config" / "users.json",
    ]


def load_user_registry() -> Tuple[Dict[str, Any], Path]:
    candidates = get_user_registry_candidates()
    seen_paths = set()
    for p in candidates:
        resolved = p.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if resolved.exists() and resolved.is_file():
            try:
                with open(resolved, "r", encoding="utf-8-sig") as f:
                    raw_data = json.load(f)
                    data = (
                        raw_data.get("users", raw_data)
                        if isinstance(raw_data, dict)
                        else {}
                    )
                    if isinstance(data, dict) and data:
                        return data, resolved
            except Exception as e:
                logging.getLogger("uvicorn.error").warning(
                    "Failed parsing user registry at %s: %s", resolved, e
                )

    fallback = {
        "administrator": {
            "display_name": "Administrator",
            "daily_quota": 999,
            "assigned_theme": "obsidian_gold",
            "custom_permissions": ["admin", "unlimited_quota"],
        },
        "admin": {
            "display_name": "Administrator",
            "daily_quota": 999,
            "assigned_theme": "obsidian_gold",
            "custom_permissions": ["admin", "unlimited_quota"],
        },
        "studio": {
            "display_name": "Studio Creator",
            "daily_quota": 50,
            "assigned_theme": "cyber_neon",
            "custom_permissions": [],
        },
        "guest": {
            "display_name": "Guest Explorer",
            "daily_quota": 10,
            "assigned_theme": "sky_peace",
            "custom_permissions": [],
        },
    }
    return fallback, Path("EMBEDDED_MEMORY_FALLBACK")


__all__ = [
    "RUNTIME_DIR",
    "SERVICES_DIR",
    "BACKEND_ROOT",
    "SHADOWBOXER_DIR",
    "ARTIFACTS_DIR",
    "STORAGE_ROOT",
    "CONFIG_DIR",
    "ARTIFACT_TTL_SECONDS",
    "BASELINE_ENGINE_DEFAULTS",
    "DEFAULT_LAYOUT_ITEMS",
    "DEFAULT_PRESET_FILENAME",
    "DEFAULT_SPECULATIVE_COT_TOKENS",
    "GenerationRequest",
    "SUPPORTED_BOX_LABELS",
    "SUPPORTED_EDIT_FORMATS",
    "SUPPORTED_FRAMING_MODES",
    "SUPPORTED_INPUT_ROLES",
    "SUPPORTED_MASK_BLEND_MODES",
    "SUPPORTED_MODELS",
    "SUPPORTED_PROMPT_MODES",
    "SUPPORTED_SOLVERS",
    "get_active_engine_defaults",
    "has_custom_default_preset",
    "parse_layer_list",
    "_FORBIDDEN_SPA_SEGMENTS",
    "_NO_CACHE_HEADERS",
    "get_runtime_override",
    "_get_preview_tokenizer",
    "_close_preview_tokenizer",
    "set_cli_overrides",
    "is_standalone_mode",
    "resolve_site_root",
    "compute_webui_asset_version",
    "render_versioned_index_html",
    "get_themes_root",
    "PollingLogFilter",
    "slugify",
    "get_user_registry_candidates",
    "load_user_registry",
]