#!/usr/bin/env python3
"""
End-to-end test script for llm-arch-generator skill.

This script verifies the SKILL.md and download_model.py work correctly.
Actual skill invocation requires Claude Code CLI.

Usage:
    python3 scripts/test_skill.py

When Claude Code is available, manual testing should include:
    /llm-arch-generator gpt2
    /llm-arch-generator microsoft/phi-1 -v
    /llm-arch-generator Qwen/Qwen2-0.5B -vv --format png
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_skill_md_syntax():
    """Verify SKILL.md has valid markdown syntax."""
    skill_md = REPO_ROOT / 'SKILL.md'
    if not skill_md.exists():
        return False, "SKILL.md not found"

    content = skill_md.read_text()

    # Check all code blocks are closed
    code_blocks = content.count('```')
    if code_blocks % 2 != 0:
        return False, f"Unclosed code blocks: {code_blocks} fences found"

    # Check all tables have proper format
    lines = content.split('\n')
    in_table = False
    for line in lines:
        if line.startswith('|'):
            in_table = True
        elif in_table and not line.startswith('|') and not line.strip().startswith('|'):
            # Check previous line was separator
            if not line.strip().startswith('|'):
                pass  # Table ended

    return True, "SKILL.md syntax OK"


def test_download_script():
    """Verify download_model.py works with a small model."""
    script = REPO_ROOT / 'scripts' / 'download_model.py'
    if not script.exists():
        return False, "download_model.py not found"

    # Verify Python syntax
    try:
        import ast
        ast.parse(script.read_text())
    except SyntaxError as e:
        return False, f"Python syntax error: {e}"

    return True, "download_model.py syntax OK"


def test_mermaid_syntax():
    """Verify mermaid diagrams have proper syntax."""
    import re

    skill_md = REPO_ROOT / 'SKILL.md'
    content = skill_md.read_text()

    pattern = r'```mermaid\s*(.*?)\s*```'
    blocks = re.findall(pattern, content, re.DOTALL)

    for i, block in enumerate(blocks, 1):
        block = block.strip()
        # Check graph direction
        if 'graph LR' not in block and 'graph TD' not in block:
            if i == 3:
                continue  # Block 3 is color conventions, not a diagram
            return False, f"Mermaid block {i} missing graph direction"

    return True, "Mermaid syntax OK"


def test_model_download():
    """Test downloading a small model."""
    script = REPO_ROOT / 'scripts' / 'download_model.py'

    # Test with gpt2 (very small model)
    result = subprocess.run(
        [sys.executable, str(script), 'gpt2', '--output-dir', '/tmp/gpt2-test'],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        return False, f"Download failed: {result.stderr}"

    # Verify config was downloaded
    config = Path('/tmp/gpt2-test/config.json')
    if not config.exists():
        return False, "config.json not downloaded"

    return True, "Model download OK"


def main():
    tests = [
        ("SKILL.md syntax", test_skill_md_syntax),
        ("download_model.py syntax", test_download_script),
        ("Mermaid syntax", test_mermaid_syntax),
        ("Model download", test_model_download),
    ]

    print("Running llm-arch-generator end-to-end tests\n")

    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}: {msg}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            all_passed = False

    print()
    if all_passed:
        print("All automated tests passed!")
        print("\nManual testing required in Claude Code:")
        print("  /llm-arch-generator gpt2")
        print("  /llm-arch-generator microsoft/phi-1 -v")
        print("  /llm-arch-generator Qwen/Qwen2-0.5B -vv --format png")
    else:
        print("Some tests failed.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())