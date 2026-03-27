#!/bin/bash
# Usage: ./render_mermaid.sh <input.mmd> [output.png] [output.svg]
# Requires: @mermaid-js/mermaid-cli installed globally or npx (optional for PNG/SVG)
# Always runs connectivity check via verify_mermaid.py regardless of render success

INPUT=$1
OUTPUT_PNG=${2:-${INPUT%.mmd}.png}
OUTPUT_SVG=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Find mermaid CLI
if command -v mmdc &> /dev/null; then
    MMDC_CMD="mmdc"
elif command -v npx &> /dev/null; then
    MMDC_CMD="npx @mermaid-js/mermaid-cli"
else
    MMDC_CMD=""
fi

# Render PNG if CLI available
if [ -n "$MMDC_CMD" ]; then
    $MMDC_CMD -i "$INPUT" -o "$OUTPUT_PNG" -b transparent -w 1920
else
    echo "Warning: mermaid-cli not found, skipping PNG render"
fi

# Render SVG if requested and CLI available
if [ -n "$OUTPUT_SVG" ] && [ -n "$MMDC_CMD" ]; then
    $MMDC_CMD -i "$INPUT" -o "$OUTPUT_SVG" -b transparent -w 1920 -f svg
fi

# Always run connectivity check regardless of render success
echo ""
echo "Running connectivity check..."
python "$SCRIPT_DIR/verify_mermaid.py" "$INPUT" --verbose
