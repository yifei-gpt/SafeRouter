#!/usr/bin/env python3
"""Fill remaining (goal x method) holes to a dense 559 x 16 grid, reusing
generate_attacks.py's generators so nothing drifts. Phases: template (no GPU),
rewrite / pair / gptfuzz (vLLM proxy). RandomSearch belongs to
backfill_optimization.py. Appends to fill_attacks.jsonl; resume-safe.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks import generate_attacks as ga   # the exact generators + proxy mgmt

REPO = Path(__file__).resolve().parents[2]
PROBE_INPUT = REPO / "data" / "adversarial" / "probe_input_merged.jsonl"
FILL_OUT = REPO / "data" / "adversarial" / "fill_attacks.jsonl"


def load_canonical_goals():
    """559 unique goals as {goal_id, goal, source} (first occurrence wins)."""
    goals, seen = [], set()
    for line in open(PROBE_INPUT):
        r = json.loads(line)
        g = r["goal"].strip().lower()
        if g in seen:
            continue
        seen.add(g)
        goals.append({"goal_id": r["question_id"], "goal": r["goal"],
                      "source": r.get("source", "")})
    return goals


def load_existing():
    """Current coverage {(goal_lower, method): query} from probe_input + fill file."""
    return ga.read_pairs(PROBE_INPUT, FILL_OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True,
                    choices=["template", "rewrite", "pair", "gptfuzz"])
    ap.add_argument("--no-autostart", action="store_true",
                    help="assume proxy already running (proxy phases)")
    args = ap.parse_args()

    goals = load_canonical_goals()
    existing = load_existing()
    print(f"canonical goals: {len(goals)}   existing (goal,method) pairs: {len(existing)}")

    needs_proxy = args.phase in ("rewrite", "pair", "gptfuzz")
    if needs_proxy:
        if args.no_autostart:
            if not ga._server_ready(ga.PROXY_URL):
                print(f"ERROR: proxy not up at {ga.PROXY_URL}", file=sys.stderr)
                sys.exit(2)
        elif not ga.ensure_proxy_running():
            print("ERROR: could not start proxy", file=sys.stderr)
            sys.exit(1)
        print(f"proxy OK: {ga.PROXY_MODEL} @ {ga.PROXY_URL}")

    with open(FILL_OUT, "a") as out_f:
        if args.phase == "template":
            n = ga.generate_template_attacks(goals, existing, out_f)
        elif args.phase == "rewrite":
            n = ga.generate_rewrite_attacks(goals, existing, out_f)
        elif args.phase == "pair":
            n = ga.generate_pair_attacks(goals, existing, out_f)
        elif args.phase == "gptfuzz":
            n = ga.generate_gptfuzz_attacks(goals, existing, out_f)
    print(f"\n[{args.phase}] wrote {n} new cells -> {FILL_OUT}")


if __name__ == "__main__":
    main()
