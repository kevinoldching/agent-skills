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

def find_modeling_file(model_id: str) -> str | None:
    """Find the modeling file by scanning the entire repository."""
    try:
        all_files = list_repo_files(model_id)
    except Exception:
        return None

    modeling_patterns = ['model.py', 'modeling.py']
    for f in all_files:
        filename = os.path.basename(f)
        if filename in modeling_patterns:
            return f
        if filename.startswith('modeling_') and filename.endswith('.py'):
            return f
    return None

def download_model(model_id: str, output_dir: str = None, use_cache: bool = True) -> tuple[str, str | None]:
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

    # Find and download modeling file
    modeling_filename = find_modeling_file(model_id)
    model_path = None
    if modeling_filename:
        model_cache = get_cache_path(model_id, modeling_filename.replace('/', '_'))
        dest = out_path / os.path.basename(modeling_filename)
        if use_cache and model_cache.exists():
            if model_cache.resolve() != dest.resolve():
                shutil.copy(model_cache, dest)
            model_path = str(dest)
        else:
            try:
                model_path = hf_hub_download(repo_id=model_id, filename=modeling_filename, local_dir=str(out_path))
                # If downloaded to same location as cache, skip copy
                if Path(model_path).resolve() != model_cache.resolve():
                    model_cache.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(model_path, str(model_cache))
                model_path = str(dest)
            except Exception:
                model_path = None
    return config_path, model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files from HuggingFace")
    parser.add_argument("model_id", help="e.g., meta-llama/Llama-3-8b")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: cache)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    args = parser.parse_args()
    config, model = download_model(args.model_id, args.output_dir, use_cache=not args.no_cache)
    print(f"config.json: {config}")
    print(f"modeling_*.py: {model}")