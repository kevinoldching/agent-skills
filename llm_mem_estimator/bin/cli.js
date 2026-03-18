#!/usr/bin/env node

/**
 * llm-mem-estimator CLI
 * Wrapper that invokes calculate_mem.py
 * Can install itself to Claude Code skills directory
 */

const { spawn } = require("child_process");
const { existsSync, rmSync, cpSync, readFileSync } = require("node:fs");
const { resolve, join } = require("node:path");
const { homedir } = require("node:os");

const SKILL_NAME = "llm_mem_estimator";
const SKILL_SRC = resolve(__dirname, "..");
const SKILL_TARGET_DIR = join(homedir(), ".claude", "skills");

function usage() {
  const pkg = JSON.parse(
    readFileSync(resolve(__dirname, "..", "package.json"), "utf8")
  );
  console.log(`${pkg.name} v${pkg.version}\n`);
  console.log("Usage: llm-mem-estimator [options]\n");
  console.log("Options:");
  console.log("  -g, --global    Install to ~/.claude/skills/ (for Claude Code)");
  console.log("  -r, --remove    Remove installed skill");
  console.log("  -h, --help      Show this help");
  console.log("\nMemory estimation options (passed to calculate_mem.py):");
  console.log("  --config CONFIG       Model YAML config file");
  console.log("  --model MODEL         HuggingFace model name");
  console.log("  --generate-config     Generate config from model");
  console.log("  --seq-len LEN         Sequence length");
  console.log("  --batch-size SIZE     Batch size");
  console.log("  --tp N                Tensor parallel size");
  console.log("  --chip CHIP           Hardware chip name");
  console.log("  --find-max-seq-len    Find max sequence length");
  console.log("\nFor full options: llm-mem-estimator --help");
  process.exit(0);
}

function parseArgs(argv) {
  const flags = { global: false, remove: false, estimation: [] };
  for (const arg of argv.slice(2)) {
    if (arg === "-g" || arg === "--global") flags.global = true;
    else if (arg === "-r" || arg === "--remove") flags.remove = true;
    else if (arg === "-h" || arg === "--help") usage();
    else flags.estimation.push(arg);
  }
  return flags;
}

function getTarget() {
  return join(SKILL_TARGET_DIR, SKILL_NAME);
}

function installSkill(target) {
  if (existsSync(target)) {
    rmSync(target, { recursive: true });
    console.log(`Removed existing: ${target}`);
  }
  cpSync(SKILL_SRC, target, {
    recursive: true,
    filter: (src) =>
      !src.includes("node_modules") &&
      !src.includes("__pycache__") &&
      !src.includes(".claude") &&
      !src.endsWith("CLAUDE.md") &&
      !src.includes("/tests"),
  });
  console.log(`Installed to: ${target}`);
}

function removeSkill(target) {
  if (!existsSync(target)) {
    console.log(`Not found: ${target}`);
    return;
  }
  rmSync(target, { recursive: true });
  console.log(`Removed: ${target}`);
}

// Check if Python dependencies are installed
function checkDependencies() {
  return new Promise((resolve) => {
    const check = spawn("python", ["-c", "import yaml"], {
      stdio: "ignore",
      cwd: SKILL_SRC,
    });

    check.on("error", () => resolve(false));
    check.on("close", (code) => resolve(code === 0));
  });
}

// Main function
async function main() {
  const flags = parseArgs(process.argv);
  const target = getTarget();

  // Handle skill installation/removal
  if (flags.remove) {
    removeSkill(target);
    return;
  }

  if (flags.global) {
    installSkill(target);
    return;
  }

  // If no estimation arguments, show help
  if (flags.estimation.length === 0) {
    usage();
    return;
  }

  // Run memory estimation
  const scriptPath = join(SKILL_SRC, "scripts", "calculate_mem.py");

  if (!existsSync(scriptPath)) {
    console.error(`Error: calculate_mem.py not found at ${scriptPath}`);
    process.exit(1);
  }

  // Check dependencies
  const depsInstalled = await checkDependencies();
  if (!depsInstalled) {
    console.log("Note: Python dependencies may not be installed.");
    console.log("Run: pip install -r requirements.txt");
  }

  // Run the Python script
  const pythonArgs = [scriptPath, ...flags.estimation];
  const proc = spawn("python", pythonArgs, {
    stdio: "inherit",
    cwd: SKILL_SRC,
  });

  proc.on("close", (code) => process.exit(code || 0));
  proc.on("error", (err) => {
    console.error("Failed to start Python:", err.message);
    process.exit(1);
  });
}

main();
