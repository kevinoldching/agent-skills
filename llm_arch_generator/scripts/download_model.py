#!/usr/bin/env python3
"""Download config.json and model.py from HuggingFace with caching.

model.py can be anywhere in the repo tree. This script uses list_repo_files()
to scan the entire repository (all subdirectories) and locate the modeling file.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

CACHE_DIR = Path.home() / ".cache" / "llm_arch_generator"

def get_cache_path(model_id: str, filename: str) -> Path:
    """Get local cache path for a downloaded file."""
    safe_id = model_id.replace('/', '_').replace('-', '_')
    return CACHE_DIR / safe_id / filename

def find_modeling_files(model_id: str) -> list[str]:
    """Find ALL modeling files by scanning the entire repository.

    Returns a list of all matching file paths, sorted alphabetically.
    The caller (SKILL.md) is responsible for selecting the correct one
    based on config.json model_type / auto_map.
    """
    try:
        all_files = list_repo_files(model_id)
    except Exception:
        return []

    modeling_candidates = []
    for f in all_files:
        filename = os.path.basename(f)
        if filename in ('model.py', 'modeling.py'):
            modeling_candidates.append(f)
        elif filename.startswith('modeling_') and filename.endswith('.py'):
            modeling_candidates.append(f)

    modeling_candidates.sort(key=lambda x: x.split('/')[-1])
    return modeling_candidates

def download_model(model_id: str, output_dir: str = None, use_cache: bool = True) -> tuple[str, list[str]]:
    """Download config.json and model.py from HuggingFace."""
    if output_dir is None:
        out_path = CACHE_DIR / model_id.replace('/', '_').replace('-', '_')
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Download config.json
    config_cache = get_cache_path(model_id, "config.json")
    import shutil
    dest = out_path / "config.json"
    if use_cache and config_cache.exists():
        if config_cache.resolve() != dest.resolve():
            shutil.copy(config_cache, dest)
        config_path = str(dest)
    else:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json", local_dir=str(out_path))
        # If downloaded to same location as cache (output_dir was None), skip copy
        if Path(config_path).resolve() != config_cache.resolve():
            config_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_path, str(config_cache))
        config_path = str(dest)

    # Find and download ALL modeling files
    modeling_filenames = find_modeling_files(model_id)
    model_paths = []
    for modeling_filename in modeling_filenames:
        model_cache = get_cache_path(model_id, modeling_filename.replace('/', '_'))
        dest = out_path / os.path.basename(modeling_filename)
        if use_cache and model_cache.exists():
            if model_cache.resolve() != dest.resolve():
                shutil.copy(model_cache, dest)
            model_paths.append(str(dest))
        else:
            try:
                downloaded = hf_hub_download(repo_id=model_id, filename=modeling_filename, local_dir=str(out_path))
                # If downloaded to same location as cache, skip copy
                if Path(downloaded).resolve() != model_cache.resolve():
                    model_cache.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(downloaded, str(model_cache))
                model_paths.append(str(dest))
            except Exception:
                pass
    return config_path, model_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files from HuggingFace")
    parser.add_argument("model_id", help="e.g., meta-llama/Llama-3-8b")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: cache)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    args = parser.parse_args()
    config, model_paths = download_model(args.model_id, args.output_dir, use_cache=not args.no_cache)
    print(f"config.json: {config}")
    for p in model_paths:
        print(f"modeling_*.py: {p}")