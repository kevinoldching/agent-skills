# Usage: .\render_mermaid.ps1 <input.mmd> <output.png> [output.svg]
# Requires: @mermaid-js/mermaid-cli installed globally or npx

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Input,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$OutputPng,

    [Parameter(Position=2)]
    [string]$OutputSvg
)

# Detect mmdc or npx
if (Get-Command mmdc -ErrorAction SilentlyContinue) {
    $MMDC_CMD = "mmdc"
} elseif (Get-Command npx -ErrorAction SilentlyContinue) {
    $MMDC_CMD = "npx @mermaid-js/mermaid-cli"
} else {
    Write-Error "Error: mermaid-cli not found. Please install Node.js and @mermaid-js/mermaid-cli."
    exit 1
}

# Resolve paths to absolute paths (handles relative paths and Windows paths)
$Input = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Input)
$OutputPng = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPng)

# Render PNG
Write-Host "Rendering PNG: $OutputPng"
& $MMDC_CMD -i $Input -o $OutputPng -b transparent -w 1920

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to render PNG"
    exit 1
}

# Render SVG if requested
if ($OutputSvg) {
    $OutputSvg = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputSvg)
    Write-Host "Rendering SVG: $OutputSvg"
    & $MMDC_CMD -i $Input -o $OutputSvg -b transparent -w 1920 -f svg

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to render SVG"
        exit 1
    }
}

Write-Host "Rendering completed successfully"
