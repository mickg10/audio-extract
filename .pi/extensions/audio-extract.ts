/**
 * audio-extract typed Pi extension (v2.1 WP10 / §15.9).
 *
 * Exposes ONLY the deterministic JSON CLI as typed tools; the model never runs
 * arbitrary shell. Every tool shells to `uv run audio-extract ... --json` and
 * returns the parsed JSON envelope. Terminal authority stays with the Python
 * selector — no tool here finalizes anything.
 */
import { execFileSync } from "node:child_process";

function cli(args: string[]): unknown {
  const out = execFileSync("uv", ["run", "audio-extract", ...args, "--json"],
    { encoding: "utf8", timeout: 900_000 });
  const lines = out.trim().split("\n");
  return JSON.parse(lines[lines.length - 1]);
}

const runId = { type: "string", description: "run id (track slug)" } as const;

export const tools = {
  inspect_run: {
    description: "Inspect a run: state, source, candidates, rejections.",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["run", "inspect", "--run-id", run_id]),
  },
  inspect_uncertainty: {
    description: "Consolidated report incl. risk bounds and challenge summary.",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["run", "report", "--run-id", run_id]),
  },
  mine_passages: {
    description: "Mine hard passages (miner v2, gated controls).",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["passages", "mine", "--run-id", run_id]),
  },
  render_panel: {
    description: "Render the locked baseline panel (comma-separated logical ids).",
    parameters: {
      type: "object",
      properties: { run_id: runId, models: { type: "string" } },
      required: ["run_id", "models"],
    },
    execute: ({ run_id, models }: { run_id: string; models: string }) =>
      cli(["panel", "render", "--run-id", run_id, "--models", models]),
  },
  build_challenges: {
    description: "Build exact-reference remix challenges from genuine controls.",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["challenges", "build", "--run-id", run_id]),
  },
  run_challenges: {
    description: "Run locked models over built challenges (exact-reference + theft).",
    parameters: {
      type: "object",
      properties: { run_id: runId, models: { type: "string" } },
      required: ["run_id", "models"],
    },
    execute: ({ run_id, models }: { run_id: string; models: string }) =>
      cli(["challenges", "run", "--run-id", run_id, "--models", models]),
  },
  score_candidates: {
    description: "Objective QA scoring over the candidate store.",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["qa", "score", "--run-id", run_id]),
  },
  rescore: {
    description: "Autonomous selection over stored challenge results (selector owns the terminal).",
    parameters: { type: "object", properties: { run_id: runId }, required: ["run_id"] },
    execute: ({ run_id }: { run_id: string }) => cli(["select", "autonomous", "--run-id", run_id]),
  },
} as const;
