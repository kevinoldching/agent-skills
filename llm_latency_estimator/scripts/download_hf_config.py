#!/usr/bin/env python3
"""
Download HuggingFace model config.json for LLM Roofline Analysis.

Usage:
    python scripts/download_hf_config.py Qwen/Qwen2.5-72B-Instruct
    python scripts/download_hf_config.py Qwen/Qwen2.5-72B-Instruct --output /path/to/save
"""

import argparse
import json
import os
import sys
from pathlib import Path

def download_config(model_name: str, output_path: str | None = None) -> dict:
    """Download config.json from HuggingFace model hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)

    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    if output_path:
        output_file = Path(output_path) / "config.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {output_file}")
    else:
        print(json.dumps(config, indent=2))

    return config

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model config.json")
    parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-72B-Instruct)")
    parser.add_argument("--output", "-o", help="Output directory to save config.json")
    args = parser.parse_args()

    download_config(args.model, args.output)

if __name__ == "__main__":
    main()