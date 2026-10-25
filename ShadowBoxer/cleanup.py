#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import stat
import sys

ROOT_DIR = Path(__file__).resolve().parent


def _resolve_cache_dir() -> Path:
    env_hf = os.environ.get("HF_HOME")
    if env_hf and env_hf.strip():
        return Path(env_hf).resolve()
    parent_cache = (ROOT_DIR.parent / ".hf_cache").resolve()
    if parent_cache.exists():
        return parent_cache
    return (ROOT_DIR / ".hf_cache").resolve()


CACHE_DIR = _resolve_cache_dir()
HUB_CACHE_DIR = CACHE_DIR / "hub"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB_CACHE_DIR)

from huggingface_hub import scan_cache_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def remove_readonly_onerror(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_readonly_onexc(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def safe_rmtree(target_path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(target_path, onexc=remove_readonly_onexc)
    else:
        shutil.rmtree(target_path, onerror=remove_readonly_onerror)


def purge_model_cache(repo_pattern: str) -> None:
    if not HUB_CACHE_DIR.exists():
        logging.info("Target cache directory does not exist: %s", HUB_CACHE_DIR)
        return

    logging.info("Scanning anchored cache directory: %s", HUB_CACHE_DIR)
    cache_info = None
    try:
        cache_info = scan_cache_dir(cache_dir=HUB_CACHE_DIR)
    except Exception as e:
        logging.warning(
            "Cache metadata index damaged or inaccessible (%s). Proceeding with direct filesystem pruning.",
            str(e),
        )

    revisions_to_delete = []
    matched_repos = []
    if cache_info is not None:
        for repo in cache_info.repos:
            if repo_pattern.lower() in repo.repo_id.lower():
                matched_repos.append(repo)
                revisions_to_delete.extend(repo.revisions)

    if revisions_to_delete:
        for repo in matched_repos:
            logging.info(
                "Found repository: %s (Revisions: %d, Size on disk: %s)",
                repo.repo_id,
                len(repo.revisions),
                repo.size_on_disk_str,
            )
        commit_hashes = [rev.commit_hash for rev in revisions_to_delete]
        delete_strategy = cache_info.delete_revisions(*commit_hashes)
        logging.info(
            "Expected freed disk space: %s", delete_strategy.expected_freed_size_str
        )
        delete_strategy.execute()
        logging.info("Cache index eviction executed successfully.")

    matching_dirs = list(HUB_CACHE_DIR.glob(f"models--*{repo_pattern}*"))
    if matching_dirs:
        logging.info("Purging model cache paths from filesystem...")
        for target_dir in matching_dirs:
            safe_rmtree(target_dir)
            logging.info("Purged cache path: %s", target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge localized OpenMOSS model revisions and cached blobs."
    )
    parser.add_argument(
        "--repo_pattern",
        type=str,
        default="MOSS-Image",
        help="Sub-string pattern matching repository targets to evict (e.g., MOSS-Image, MOSS-Image-SFT, MOSS-Image-Base, Moss-Image-Edit).",
    )
    args = parser.parse_args()
    try:
        purge_model_cache(args.repo_pattern)
    except Exception as e:
        logging.error("Eviction failure: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()