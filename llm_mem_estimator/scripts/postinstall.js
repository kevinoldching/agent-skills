#!/usr/bin/env node

/**
 * postinstall.js
 * Runs after npm install to:
 * 1. Install Python dependencies
 * 2. Copy to ~/.claude/skills (if global install)
 */

const { spawn } = require("child_process");
const { existsSync, rmSync, cpSync } = require("node:fs");
const { resolve, join } = require("node:path");
const { homedir } = require("node:os");

const SKILL_NAME = "llm_mem_estimator";
const PKG_ROOT = resolve(__dirname, "..");
const SKILL_TARGET = join(homedir(), ".claude", "skills", SKILL_NAME);

// Check if this is a global installation
function isGlobalInstall() {
  // Check if npm config prefix is in the package root
  // Global installs typically don't have node_modules as a subdirectory
  // or the package is in a global node_modules path
  const npmPrefix = process.env.npm_config_prefix || "";

  if (npmPrefix) {
    return PKG_ROOT.startsWith(npmPrefix) || PKG_ROOT.includes("lib/node_modules");
  }

  // Default: if not in a local node_modules, assume global
  return !PKG_ROOT.includes("node_modules");
}

// Install Python dependencies
function installPythonDeps() {
  return new Promise((resolve) => {
    console.log("Installing Python dependencies...");

    const pip = spawn("pip", ["install", "-r", "requirements.txt"], {
      stdio: "inherit",
      cwd: PKG_ROOT,
    });

    pip.on("close", (code) => {
      if (code === 0) {
        console.log("Python dependencies installed.");
      } else {
        console.log("Warning: Python dependencies installation skipped.");
        console.log("Run manually: pip install -r requirements.txt");
      }
      resolve();
    });

    pip.on("error", () => {
      console.log("Warning: pip not found. Skip Python dependencies.");
      resolve();
    });
  });
}

// Install to Claude Code skills
function installToSkills() {
  console.log("Installing to ~/.claude/skills/ ...");

  if (existsSync(SKILL_TARGET)) {
    rmSync(SKILL_TARGET, { recursive: true });
    console.log(`Removed existing: ${SKILL_TARGET}`);
  }

  cpSync(PKG_ROOT, SKILL_TARGET, {
    recursive: true,
    filter: (src) =>
      !src.includes("node_modules") &&
      !src.includes("__pycache__") &&
      !src.includes(".claude") &&
      !src.endsWith("CLAUDE.md") &&
      !src.endsWith("package-lock.json"),
  });

  console.log(`Installed to: ${SKILL_TARGET}`);
}

// Main
async function main() {
  const isGlobal = isGlobalInstall();

  // Install Python dependencies (both global and local)
  await installPythonDeps();

  // Install to Claude Code skills (global only)
  if (isGlobal) {
    installToSkills();
  }
}

main();
