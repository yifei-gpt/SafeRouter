#!/usr/bin/env python3
"""A clean, de-duplicated adversarial probe input: drop degenerate rows, recover
distinct goals by (goal_text, source) -> 83 + 124 + 352 = 559, give each one
source-namespaced id (never bare-numeric, which collides with the driver's pb_
prefixing), keep the first valid row per (goal, method). Input is read-only.

    python -m probe.build_probe_input [--in ... --out ...]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]   # saferouter/probe/X.py -> repo root
DEFAULT_IN  = REPO_ROOT / "data" / "adversarial" / "probe_input_merged.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "adversarial" / "probe_input_merged.cleaned.jsonl"

# Source -> namespace prefix used for minting / validating canonical ids.
SOURCE_PREFIX = {
    "pandabench":  "pb_",
    "harmbench":   "hb_",
    "sorry-bench": "sb_",
}
NAMESPACES = ("pb_", "hb_", "sb_")

# Expected distinct-goal breakdown (the adversarial probe set is 559 goals).
EXPECTED_GOALS = {"pandabench": 83, "harmbench": 124, "sorry-bench": 352}
EXPECTED_TOTAL = 559

# Dropped for uniform coverage, leaving 16 methods (15 attacks + Baseline).
EXCLUDE_METHODS = {"GCG", "SCAV", "COLD", "AutoDAN", "TAP"}

REFUSAL_PREFIXES = (
    "i can't",
    "i cannot",
    "i can not",
    "i'm sorry",
    "i am sorry",
    "i'm unable",
    "i am unable",
    "i won't",
    "i will not",
    "sorry, i can",
    "sorry, but i",
)
# Dropped only if the query OPENS with one of these and has no payload.
REFUSAL_MAX_LEN = 600


def is_whole_query_refusal(query: str) -> bool:
    """True iff the WHOLE query is an attacker/target refusal (be conservative:
    must START with a refusal phrase and not carry an embedded attack payload)."""
    # Normalize curly apostrophes so they match REFUSAL_PREFIXES.
    low = (query.lower().lstrip()
           .replace("’", "'").replace("‘", "'")
           .replace("ʼ", "'").replace("′", "'"))
    if not any(low.startswith(p) for p in REFUSAL_PREFIXES):
        return False
    # A second turn means a multi-turn attack wrapper, not a refusal.
    if "\nuser:" in low or "user :" in low:
        return False
    # Pure refusals (often followed by safety-resource text) are short.
    return len(query.strip()) <= REFUSAL_MAX_LEN


def drop_reason(query) -> str:
    """Return the drop reason for a query, or '' if the row is valid."""
    qs = "" if query is None else str(query).strip()
    if not qs:
        return "empty"
    if qs == "[ATTACK FAILED]":
        return "attack_failed"
    if len(qs) < 15:
        return "too_short"
    if is_whole_query_refusal(qs):
        return "refusal"
    return ""


def derive_driver_query_id(question_id: str) -> str:
    """Reproduce probe/adversarial.py's rule: keep namespaced ids, else pb_<id>."""
    qid = str(question_id)
    return qid if qid.startswith(NAMESPACES) else f"pb_{qid}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN),
                    help=f"input probe_input JSONL (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT),
                    help=f"cleaned output JSONL (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    assert in_path.exists(), f"input not found: {in_path}"
    assert in_path.resolve() != out_path.resolve(), \
        "refusing to overwrite the input file; choose a different --out"

    # ---- Pass 1: read, filter, group ----
    n_in = 0
    bad_lines = 0
    drops = Counter()
    # (goal_text, source) -> {goal, source, ids, {method: first valid row}}.
    groups = {}

    with open(in_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                r = json.loads(line)
                qid = str(r["question_id"])
                query = r["query"]
                method = r.get("attack_method", "?")
                goal = r.get("goal", "")
                source = r.get("source", "")
            except (json.JSONDecodeError, KeyError) as e:
                bad_lines += 1
                print(f"  ! malformed line {ln}: {type(e).__name__}: {str(e)[:80]}",
                      file=sys.stderr)
                continue

            if method in EXCLUDE_METHODS:
                drops["excluded_method"] += 1
                continue

            reason = drop_reason(query)
            if reason:
                drops[reason] += 1
                continue

            key = (goal, source)
            g = groups.get(key)
            if g is None:
                g = groups[key] = {
                    "goal": goal,
                    "source": source,
                    "existing_ids": set(),
                    "rows": {},  # attack_method -> emitted row
                }
            # Track any already-namespaced id seen for this goal (canonical pref).
            if qid.startswith(NAMESPACES):
                g["existing_ids"].add(qid)
            # (goal, method) dedup: keep the FIRST valid query.
            if method not in g["rows"]:
                g["rows"][method] = {
                    "question_id": qid,   # provisional; canonicalized in pass 2
                    "query": query,
                    "attack_method": method,
                    "goal": goal,
                    "source": source,
                }

    # Pass 2: assign canonical namespaced ids, distinct goals per source.
    goals_by_source = Counter(src for (_g, src) in groups.keys())

    # Mint stable fresh ids per source, avoiding any already in use there.
    used_ids_by_prefix = defaultdict(set)
    for g in groups.values():
        for eid in g["existing_ids"]:
            for pref in NAMESPACES:
                if eid.startswith(pref):
                    used_ids_by_prefix[pref].add(eid)
    mint_counter = defaultdict(int)

    def mint_id(prefix: str) -> str:
        used = used_ids_by_prefix[prefix]
        while True:
            cand = f"{prefix}{mint_counter[prefix]}"
            mint_counter[prefix] += 1
            if cand not in used:
                used.add(cand)
                return cand

    canonical_id_of = {}  # (goal, source) -> canonical question_id
    for key, g in groups.items():
        source = g["source"]
        prefix = SOURCE_PREFIX.get(source)
        if prefix is None:
            # Unknown source falls back to pb_, still namespaced and collision-free.
            prefix = "pb_"
        existing = sorted(eid for eid in g["existing_ids"] if eid.startswith(prefix))
        canonical_id_of[key] = existing[0] if existing else mint_id(prefix)

    # ---- Emit cleaned rows ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    per_method = Counter()
    emitted_id_for_goal = {}   # canonical question_id -> (goal, source)
    final_query_id_map = defaultdict(set)  # driver query_id -> set of goal texts
    with open(out_path, "w") as out:
        for key, g in groups.items():
            cid = canonical_id_of[key]
            emitted_id_for_goal[cid] = key
            final_qid = derive_driver_query_id(cid)
            final_query_id_map[final_qid].add(key)
            for method in sorted(g["rows"]):
                row = g["rows"][method]
                rec = {
                    "question_id":   cid,
                    "query":         row["query"],
                    "attack_method": method,
                    "goal":          g["goal"],
                    "source":        g["source"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
                per_method[method] += 1

    # Collision gate: after pb_-prefixing, no two goals may share a query_id.
    collisions = {qid: keys for qid, keys in final_query_id_map.items()
                  if len(keys) > 1}
    # Every emitted id must be namespaced, so the driver leaves it untouched.
    non_namespaced = [cid for cid in emitted_id_for_goal
                      if not cid.startswith(NAMESPACES)]
    collision_gate_pass = (not collisions) and (not non_namespaced)

    n_distinct_goals = len(groups)
    goal_gate_pass = (
        n_distinct_goals == EXPECTED_TOTAL
        and all(goals_by_source.get(s, 0) == c for s, c in EXPECTED_GOALS.items())
    )

    # ---- Summary ----
    print("=" * 70)
    print("build_probe_input.py summary")
    print("=" * 70)
    print(f"input file:        {in_path}")
    print(f"output file:       {out_path}")
    print(f"input rows:        {n_in}")
    if bad_lines:
        print(f"malformed lines:   {bad_lines} (skipped)")
    total_dropped = sum(drops.values())
    print(f"dropped rows:      {total_dropped}")
    for reason in ("empty", "attack_failed", "too_short", "refusal"):
        print(f"    {reason:<14} {drops.get(reason, 0)}")
    print("-" * 70)
    print(f"distinct goals:    {n_distinct_goals}  (expected {EXPECTED_TOTAL})")
    for s in ("pandabench", "harmbench", "sorry-bench"):
        got = goals_by_source.get(s, 0)
        exp = EXPECTED_GOALS[s]
        flag = "" if got == exp else f"  <-- MISMATCH (expected {exp})"
        print(f"    {s:<14} {got}{flag}")
    other = {s: c for s, c in goals_by_source.items() if s not in EXPECTED_GOALS}
    if other:
        print(f"    other sources: {other}  <-- UNEXPECTED")
    print(f"GOAL GATE:         {'PASS' if goal_gate_pass else 'FAIL'}")
    print("-" * 70)
    print(f"emitted rows:      {n_out}  (distinct (goal, method) pairs)")
    print(f"attack methods:    {len(per_method)}")
    print("per-method emitted counts:")
    for m in sorted(per_method):
        print(f"    {m:<18} {per_method[m]}")
    print("-" * 70)
    print(f"COLLISION GATE:    {'PASS' if collision_gate_pass else 'FAIL'}")
    if non_namespaced:
        print(f"    non-namespaced canonical ids: {len(non_namespaced)} "
              f"(e.g. {non_namespaced[:5]})")
    if collisions:
        print(f"    final query_id collisions: {len(collisions)}")
        for qid, keys in list(collisions.items())[:5]:
            print(f"      {qid}: {len(keys)} distinct goals -> "
                  f"{[k[0][:40] for k in keys]}")
    print("=" * 70)

    if not goal_gate_pass:
        print("WARNING: distinct-goal count is NOT 559 — see breakdown above.",
              file=sys.stderr)
    if not collision_gate_pass:
        print("WARNING: collision gate FAILED — driver would merge distinct goals.",
              file=sys.stderr)

    # Non-zero exit on a gate failure so callers can detect it.
    return 0 if (goal_gate_pass and collision_gate_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
