#!/bin/bash
# Usage: ./render_mermaid.sh <input.mmd> <output.png> [output.svg]
# Requires: @mermaid-js/mermaid-cli installed globally or npx

INPUT=$1
OUTPUT_PNG=$2
OUTPUT_SVG=$3

if command -v mmdc &> /dev/null; then
    MMDC_CMD="mmdc"
elif command -v npx &> /dev/null; then
    MMDC_CMD="npx @mermaid-js/mermaid-cli"
else
    echo "Error: mermaid-cli not found"
    exit 1
fi

# Render PNG
$MMDC_CMD -i "$INPUT" -o "$OUTPUT_PNG" -b transparent -w 1920

# Render SVG if requested
if [ -n "$OUTPUT_SVG" ]; then
    $MMDC_CMD -i "$INPUT" -o "$OUTPUT_SVG" -b transparent -w 1920 -f svg
fi
