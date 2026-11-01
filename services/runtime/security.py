# services/runtime/security.py
from __future__ import annotations

from collections import OrderedDict
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import HTTPException, Request, status

from .env import CONFIG_DIR, get_runtime_override

_SECRETS_FILE = CONFIG_DIR / ".runtime_secrets.json"


def _load_or_create_csprng_secrets() -> Tuple[str, str]:
    env_pow = os.environ.get("VISIONSTACK_POW_SECRET", "").strip()
    env_sess = os.environ.get("VISIONSTACK_SESSION_SECRET", "").strip()
    if env_pow and env_sess:
        return env_pow, env_sess

    stored_pow = ""
    stored_sess = ""
    if _SECRETS_FILE.is_file():
        try:
            data = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                stored_pow = str(data.get("pow_secret", "")).strip()
                stored_sess = str(data.get("session_secret", "")).strip()
        except Exception:
            pass

    pow_sec = env_pow or stored_pow or secrets.token_hex(32)
    sess_sec = env_sess or stored_sess or secrets.token_hex(32)

    if not (_SECRETS_FILE.is_file() and stored_pow and stored_sess):
        try:
            _SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = _SECRETS_FILE.with_suffix(".tmp")
            tmp_file.write_text(
                json.dumps(
                    {
                        "pow_secret": pow_sec,
                        "session_secret": sess_sec,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp_file.replace(_SECRETS_FILE)
        except Exception:
            pass

    return pow_sec, sess_sec


POW_SECRET, SESSION_SECRET = _load_or_create_csprng_secrets()
POW_DIFFICULTY = int(os.environ.get("VISIONSTACK_POW_DIFFICULTY", "4"))
POW_EXPIRATION_SECONDS = int(
    os.environ.get("VISIONSTACK_POW_EXPIRATION_SECONDS", "120")
)
MAX_REPLAY_CACHE_SIZE = 16384
MAX_USER_INFLIGHT_JOBS = int(
    os.environ.get("VISIONSTACK_MAX_USER_INFLIGHT", "4")
)
MAX_GLOBAL_QUEUE_DEPTH = int(
    os.environ.get("VISIONSTACK_MAX_QUEUE_DEPTH", "32")
)

_SECURITY_LOCK = threading.Lock()
_REPLAY_CACHE: OrderedDict[str, float] = OrderedDict()
_RATE_LIMIT_BUCKETS: OrderedDict[str, List[float]] = OrderedDict()


def _is_trusted_proxy_peer(host: str) -> bool:
    clean = host.strip().lower()
    if clean in ("127.0.0.1", "::1", "localhost", "testclient", ""):
        return True
    try:
        ip_obj = ipaddress.ip_address(clean)
        return bool(ip_obj.is_loopback or ip_obj.is_private)
    except ValueError:
        return False


def _normalize_ip_token(raw_ip: str) -> Optional[str]:
    cand = raw_ip.strip()
    if not cand:
        return None
    if cand.startswith("[") and "]" in cand:
        cand = cand[1 : cand.index("]")]
    elif cand.count(":") == 1 and "." in cand:
        cand = cand.split(":", 1)[0]
    if cand.lower() in ("localhost", "testclient"):
        return "127.0.0.1"
    try:
        ip_obj = ipaddress.ip_address(cand)
        if ip_obj.is_loopback:
            return "127.0.0.1"
        return str(ip_obj)
    except ValueError:
        return None


def resolve_client_ip(request: Optional[Request]) -> str:
    if request is None:
        return "127.0.0.1"
    peer_host = (
        request.client.host
        if (request.client is not None and request.client.host)
        else "127.0.0.1"
    )
    if _is_trusted_proxy_peer(peer_host):
        for hdr_name in ("cf-connecting-ip", "x-real-ip", "x-forwarded-for"):
            raw_hdr = request.headers.get(hdr_name)
            if raw_hdr:
                first_hop = raw_hdr.split(",", 1)[0].strip()
                norm_hop = _normalize_ip_token(first_hop)
                if norm_hop:
                    return norm_hop
    return _normalize_ip_token(peer_host) or "127.0.0.1"


def check_rate_limit(
    client_ip: str,
    scope: str = "default",
    max_requests: int = 60,
    window_seconds: float = 60.0,
) -> int:
    now = time.time()
    bucket_key = f"{scope}:{client_ip}"
    cutoff = now - float(window_seconds)
    with _SECURITY_LOCK:
        timestamps = _RATE_LIMIT_BUCKETS.get(bucket_key, [])
        valid_ts = [t for t in timestamps if t > cutoff]
        if len(valid_ts) >= max_requests:
            _RATE_LIMIT_BUCKETS[bucket_key] = valid_ts
            _RATE_LIMIT_BUCKETS.move_to_end(bucket_key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded for {scope}. Please retry shortly.",
            )
        valid_ts.append(now)
        _RATE_LIMIT_BUCKETS[bucket_key] = valid_ts
        _RATE_LIMIT_BUCKETS.move_to_end(bucket_key)
        while len(_RATE_LIMIT_BUCKETS) > 4096:
            _RATE_LIMIT_BUCKETS.popitem(last=False)
        return len(valid_ts)


def create_session_token(slug: str) -> str:
    sess_secret = str(get_runtime_override("SESSION_SECRET", SESSION_SECRET))
    ts = str(int(time.time()))
    payload = f"{slug}:{ts}"
    signature = hmac.new(
        sess_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"


def verify_session_token(token: Optional[str]) -> str:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    sess_secret = str(get_runtime_override("SESSION_SECRET", SESSION_SECRET))
    try:
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError
        slug, ts, sig = parts[0], parts[1], parts[2]
        payload = f"{slug}:{ts}"
        expected = hmac.new(
            sess_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError
        if time.time() - int(ts) > 604800:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired.",
            )
        return slug
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session credentials.",
        )


def _compute_pow_hmac(
    pow_secret: str, challenge: str, client_ip: str
) -> str:
    norm_ip = _normalize_ip_token(client_ip) or "127.0.0.1"
    mac_input = (
        challenge
        if norm_ip == "127.0.0.1"
        else f"{challenge}:{norm_ip}"
    )
    return hmac.new(
        pow_secret.encode("utf-8"),
        mac_input.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def generate_pow_challenge(
    client_ip: str = "127.0.0.1",
    queue_depth: int = 0,
    recent_ip_challenges: int = 0,
) -> Dict[str, Any]:
    pow_secret = str(get_runtime_override("POW_SECRET", POW_SECRET))
    base_diff = int(get_runtime_override("POW_DIFFICULTY", POW_DIFFICULTY))
    eff_diff = base_diff
    if base_diff >= 4:
        if queue_depth >= 2:
            eff_diff += 1
        if recent_ip_challenges >= 10:
            eff_diff += 1
        eff_diff = min(6, eff_diff)

    now = int(time.time())
    nonce = uuid.uuid4().hex
    raw = f"{now}:{nonce}:{eff_diff}"
    sig = _compute_pow_hmac(pow_secret, raw, client_ip)
    return {
        "timestamp": now,
        "nonce": nonce,
        "difficulty": eff_diff,
        "challenge": raw,
        "signature": sig,
    }


def _prune_replay_cache_locked(now: float, ttl: float) -> None:
    expired_keys = [
        k for k, ts in _REPLAY_CACHE.items() if (now - ts) >= ttl
    ]
    for k in expired_keys:
        _REPLAY_CACHE.pop(k, None)
    while len(_REPLAY_CACHE) > MAX_REPLAY_CACHE_SIZE:
        _REPLAY_CACHE.popitem(last=False)


def verify_pow_solution(
    challenge: str,
    signature: str,
    solution_nonce: str,
    client_ip: str = "127.0.0.1",
) -> bool:
    if not challenge or not signature or solution_nonce is None:
        return False
    sol_str = str(solution_nonce).strip()
    if not sol_str or len(sol_str) > 64:
        return False

    pow_secret = str(get_runtime_override("POW_SECRET", POW_SECRET))
    min_diff = max(0, int(get_runtime_override("POW_DIFFICULTY", POW_DIFFICULTY)))
    ttl = float(
        get_runtime_override("POW_EXPIRATION_SECONDS", POW_EXPIRATION_SECONDS)
    )

    try:
        parts = challenge.split(":")
        if len(parts) != 3:
            return False
        ts_str, challenge_nonce, diff_str = parts[0], parts[1], parts[2]
        ts = int(ts_str)
        diff = int(diff_str)
        if not challenge_nonce or len(challenge_nonce) < 8:
            return False
    except Exception:
        return False

    if diff < min_diff or diff > 12:
        return False

    now = time.time()
    if (now - ts) > ttl or ts > (now + 30):
        return False

    expected_sig = _compute_pow_hmac(pow_secret, challenge, client_ip)
    if not hmac.compare_digest(signature, expected_sig):
        return False

    attempt = f"{challenge}:{sol_str}".encode("utf-8")
    digest = hashlib.sha256(attempt).hexdigest()
    if not digest.startswith("0" * diff):
        return False

    pair_key = f"{challenge}:{sol_str}"
    nonce_key = f"nonce:{challenge_nonce}"
    with _SECURITY_LOCK:
        _prune_replay_cache_locked(now, ttl)
        if nonce_key in _REPLAY_CACHE or pair_key in _REPLAY_CACHE:
            return False
        _REPLAY_CACHE[nonce_key] = now
        _REPLAY_CACHE[pair_key] = now
    return True


__all__ = [
    "POW_SECRET",
    "SESSION_SECRET",
    "POW_DIFFICULTY",
    "POW_EXPIRATION_SECONDS",
    "MAX_REPLAY_CACHE_SIZE",
    "MAX_USER_INFLIGHT_JOBS",
    "MAX_GLOBAL_QUEUE_DEPTH",
    "_REPLAY_CACHE",
    "_RATE_LIMIT_BUCKETS",
    "resolve_client_ip",
    "check_rate_limit",
    "create_session_token",
    "verify_session_token",
    "generate_pow_challenge",
    "verify_pow_solution",
]