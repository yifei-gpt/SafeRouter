#!/usr/bin/env python
"""Backfill the optimization/encoding attacks (GPT4Cipher, RandomSearch, SCAV,
AutoDAN, COLD, ArtPrompt) from PandaBench's 83 goals to all 559. White-box ones
load Llama-3.1-8B via panda_guard, sharded over --workers (~16GB each).
Appends to data/adversarial/backfill_attacks.jsonl; resume-safe.

    --methods M...  subset (default all)   --timing  1 goal/method, no write
    --workers N     white-box processes    --limit N goals per method
"""

import os
import sys

# Environment (CRITICAL): set before any HF / torch import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce allocator fragmentation; helps the fp32 COLD decode.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# COLD needs GLIBCXX_3.4.30: SAFEROUTER_LIBSTDCPP -> a conda libstdc++.
_CONDA_LIBSTDCPP = os.environ.get("SAFEROUTER_LIBSTDCPP", "")
if (not os.environ.get("_BACKFILL_PRELOADED")
        and os.path.exists(_CONDA_LIBSTDCPP)):
    os.environ["_BACKFILL_PRELOADED"] = "1"
    pre = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = (_CONDA_LIBSTDCPP + (":" + pre if pre else ""))
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import time
import argparse
import contextlib
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# One definition per template: a drift silently changes the prompts.
from attacks.generate_attacks import ARTPROMPT_REWRITE_TEMPLATE, GPT4CIPHER_TEMPLATE

# Constants
WHITE_BOX_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_MISSING_GOALS = 476        # 124 HarmBench + 352 Sorry-Bench (the backfill scope)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                       # .../SafeRouter/SafeRouter
PROBE_INPUT = REPO / "data" / "adversarial" / "probe_input_merged.jsonl"
OUT_PATH = REPO / "data" / "adversarial" / "backfill_attacks.jsonl"

# SCAV optimized-instruction CSVs, shipped with panda-guard.
PANDA_GUARD_ROOT = REPO.parent / "panda-guard"
SCAV_CSV_8B = PANDA_GUARD_ROOT / "data" / "SCAV" / "optimized_instructions_8b.csv"
SCAV_CSV_70B = PANDA_GUARD_ROOT / "data" / "SCAV" / "optimized_instructions_70b.csv"

# AutoDAN initial population (genetic algorithm seed prompts).
AUTODAN_PROMPTS = (PANDA_GUARD_ROOT / "src" / "panda_guard" / "role" / "attacks"
                   / "autodan" / "prompt_group.yaml")


# COLD decoding config (panda-guard cold.yaml), as the self.args namespace.
COLD_CONFIG = dict(
    no_cuda=False, verbose=False, print_every=2000, pretrained_model="llama2",
    wandb=False, straight_through=True, topk=10, rl_topk=0, lexical="max",
    lexical_variants=False, if_zx=False, fp16=True, version="", start=0, end=50,
    mode="suffix", control_type="sentiment", batch_size=1, length=100,
    max_length=100, frozen_length=0, goal_weight=500, rej_weight=100,
    abductive_filterx=False, lr_nll_portion=1.0, prefix_length=0,
    counterfactual_max_ngram=3, no_loss_rerank=False, use_sysprompt=False,
    input_lgt_temp=1.0, output_lgt_temp=1.0, rl_output_lgt_temp=1.0, init_temp=1,
    init_mode="original", stepsize=0.1, stepsize_ratio=1, stepsize_iters=1000,
    num_iters=1000, min_iters=0, noise_iters=1, win_anneal_iters=1000,
    constraint_iters=1000, gs_mean=0.0, gs_std=0.01,
    large_noise_iters="50, 200, 500, 1500",
    large_gs_std="0.1, 0.05, 0.01, 0.001",
)

ALL_METHODS = ["GPT4Cipher", "RandomSearch", "SCAV", "AutoDAN", "COLD", "ArtPrompt"]
WHITE_BOX = {"RandomSearch", "AutoDAN", "COLD"}   # need a per-process model copy
# COLD must run fp32, so reduced suffix length and one process.
COLD_LENGTH = 20
COLD_MAX_WORKERS = 1
TARGET_SOURCES = {"harmbench", "sorry-bench"}      # the 476 missing goals


# Goal loading
def load_missing_goals():
    """Return the 476 missing goals as [{"question_id","goal","source"}].

    Canonical (question_id, goal) per source comes from the probe input. Only
    HarmBench + Sorry-Bench sources are returned (PandaBench's 83 are already
    covered for all 6 target attacks). Each (source, goal) maps to exactly one
    question_id in the cleaned probe input (verified).
    """
    seen = {}  # (source, goal) -> question_id
    with open(PROBE_INPUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            src = d.get("source")
            if src not in TARGET_SOURCES:
                continue
            key = (src, d["goal"])
            if key not in seen:
                seen[key] = d["question_id"]
    goals = [{"question_id": qid, "goal": goal, "source": src}
             for (src, goal), qid in seen.items()]
    # Deterministic order: source then question_id, so shards are reproducible.
    goals.sort(key=lambda g: (g["source"], g["question_id"]))
    return goals


def load_existing_pairs():
    """Return the set of (goal, attack_method) already present in the output."""
    pairs = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                pairs.add((d.get("goal", "").strip().lower(), d.get("attack_method")))
    return pairs


# Attackers return one goal's query or raise; white-box built once/process.
def _quiet_run(attacker, idx=-1):
    """attacker.attack mutates messages in place and prints; suppress stdout."""
    def run(goal):
        messages = [{"role": "user", "content": goal}]
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            result = attacker.attack(messages)
        return result[idx]["content"]
    return run


def build_gpt4cipher():
    from panda_guard.role.attacks.gpt4_cipher import (
        GPT4CipherAttacker, GPT4CipherAttackerConfig)
    cfg = GPT4CipherAttackerConfig(rewrite_template=GPT4CIPHER_TEMPLATE)
    attacker = GPT4CipherAttacker(cfg)

    return _quiet_run(attacker)


def build_scav():
    """SCAV is a STATIC CSV lookup keyed on the exact goal string. The shipped
    optimized_instructions_8b.csv only contains the 83 PandaBench goals, so for
    any of the 476 missing goals get_corresponding_string returns None. We
    treat a None lookup as 'not backfillable' and raise so the caller
    logs+skips it."""
    from panda_guard.role.attacks.scav import ScavAttacker, ScavAttackerConfig
    cfg = ScavAttackerConfig(
        target_llm_name=WHITE_BOX_MODEL,
        optimzed_file_8b=str(SCAV_CSV_8B),
        optimzed_file_70b=str(SCAV_CSV_70B),
    )
    attacker = ScavAttacker(cfg)

    def run(goal):
        messages = [{"role": "user", "content": goal}]
        result = attacker.attack(messages)
        q = result[-1]["content"]
        if q is None:
            raise ValueError("SCAV CSV has no optimized instruction for this goal "
                             "(static lookup table covers PandaBench only)")
        return q
    return run




def build_artprompt():
    """ArtPrompt masks words via gpt-3.5-turbo (OpenAI), then renders the masked
    word(s) as ASCII art from a font table (vitc-s.json). Requires
    OPENAI_API_KEY (caller skips the method if unset). NOTE: panda_guard's
    art_prompt.py hardcodes the vitc-s.json path to a location that does not
    exist in this repo layout, so ArtPrompt typically fails at attack() time
    here; failures are caught+logged per goal."""
    from panda_guard.role.attacks.art_prompt import (
        ArtPromptAttacker, ArtPromptAttackerConfig)
    cfg = ArtPromptAttackerConfig(
        mask_llm_url=None,                       # OpenAI default base URL + OPENAI_API_KEY
        rewrite_template=ARTPROMPT_REWRITE_TEMPLATE,
    )
    attacker = ArtPromptAttacker(cfg)

    return _quiet_run(attacker)


def build_random_search(num_steps=None):
    from panda_guard.role.attacks.random_search import (
        RandomSearchAttacker, RandomSearchAttackerConfig)
    from panda_guard.llms import HuggingFaceLLMConfig, LLMGenerateConfig
    target_cfg = HuggingFaceLLMConfig(model_name=WHITE_BOX_MODEL, device_map="cuda")
    cfg = RandomSearchAttackerConfig(
        target_llm_config=target_cfg,
        target_llm_gen_config=LLMGenerateConfig(
            max_n_tokens=4096, temperature=1.0, logprobs=True, seed=42),
        max_iterations=num_steps if num_steps else 3,
        max_restarts=3,
        max_n_to_change=8,
        logprob_threshold=-1.0,
        target_str="Sure,",
        verbose=False,
    )
    attacker = RandomSearchAttacker(cfg)

    return _quiet_run(attacker)


def build_autodan(num_steps=None):
    from panda_guard.role.attacks.autodan.autodan import (
        AutoDanAttacker, AutoDanAttackerConfig)
    from panda_guard.llms import HuggingFaceLLMConfig, LLMGenerateConfig
    target_cfg = HuggingFaceLLMConfig(model_name=WHITE_BOX_MODEL, device_map="cuda")
    cfg = AutoDanAttackerConfig(
        target_llm_config=target_cfg,
        target_llm_gen_config=LLMGenerateConfig(
            max_n_tokens=150, temperature=0.0),
        llm_mutation=False,                 # pure gradient-based GA (PandaBench default)
        num_steps=num_steps if num_steps else 250,   # PandaBench default (matches existing AutoDAN probes)
        batch_size=32,
        adv_prompts_size=256,
        init_adv_prompts_path=str(AUTODAN_PROMPTS),
    )
    attacker = AutoDanAttacker(cfg)

    return _quiet_run(attacker, 0)


def build_cold(num_iters=None, length=None):
    from panda_guard.role.attacks.cold_attack.cold import (
        ColdAttacker, ColdAttackerConfig)
    from panda_guard.llms import HuggingFaceLLMConfig, LLMGenerateConfig
    wb_cfg = HuggingFaceLLMConfig(model_name=WHITE_BOX_MODEL, device_map="cuda")
    cfg = ColdAttackerConfig(
        white_box_llm_config=wb_cfg,
        white_box_llm_gen_config=LLMGenerateConfig(
            max_n_tokens=4096, temperature=1.0, logprobs=False),
    )
    # COLD's decoder hardcodes float32, panda_guard loads fp16 -> cast after.
    cold_args = dict(COLD_CONFIG)
    if num_iters is not None:
        cold_args["num_iters"] = num_iters
    if length is not None:
        cold_args["length"] = length
        cold_args["max_length"] = length
    cfg.cold_config = SimpleNamespace(**cold_args)
    attacker = ColdAttacker(cfg)
    attacker.llm.model = attacker.llm.model.float()   # fp16 -> fp32 (dtype-consistent decode)

    return _quiet_run(attacker)


BUILDERS = {
    "GPT4Cipher": build_gpt4cipher,
    "SCAV": build_scav,
    "ArtPrompt": build_artprompt,
    "RandomSearch": build_random_search,
    "AutoDAN": build_autodan,
    "COLD": build_cold,
}


def build_attacker(method, timing=False):
    """Build one attacker. For white-box methods under --timing, use a reduced
    iteration count so a single-goal probe finishes quickly; the caller
    extrapolates per-step cost to the full iteration budget separately."""
    if timing and method == "RandomSearch":
        return build_random_search(num_steps=1)
    if timing and method == "AutoDAN":
        return build_autodan(num_steps=2)
    if timing and method == "COLD":
        return build_cold(num_iters=20, length=COLD_LENGTH)
    if method == "COLD":
        return build_cold(length=COLD_LENGTH)
    return BUILDERS[method]()


# Output writing (atomic-ish append)
def write_record(rec):
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with open(OUT_PATH, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# White-box worker (one process owns a model and a shard of goals)
def whitebox_worker(method, shard, existing, result_q):
    """Load the model once, attack each goal in the shard, push records to a
    queue for the parent to write (single writer => safe append)."""
    n_ok = 0
    n_skip = 0
    n_fail = 0
    try:
        run = build_attacker(method, timing=False)
    except Exception as e:
        # Model/attacker failed to build: report the whole shard as failed.
        result_q.put(("buildfail", method, str(e)[:300], len(shard)))
        return
    for g in shard:
        key = (g["goal"].strip().lower(), method)
        if key in existing:
            n_skip += 1
            continue
        try:
            query = run(g["goal"])
            if not query or not str(query).strip():
                n_fail += 1
                result_q.put(("log", method, "EMPTY %s" % g["question_id"]))
                continue
            rec = {
                "question_id": g["question_id"],
                "query": str(query).strip(),
                "attack_method": method,
                "goal": g["goal"],
                "source": g["source"],
            }
            result_q.put(("record", rec))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            result_q.put(("log", method, "FAILED %s: %s" % (g["question_id"], str(e)[:200])))
    result_q.put(("done", method, n_ok, n_skip, n_fail))


def run_whitebox_method(method, goals, existing, workers):
    """Shard goals across `workers` processes; single parent writes the output."""
    work = [g for g in goals if (g["goal"].strip().lower(), method) not in existing]
    print("[%s] %d goals to backfill (%d already present)"
          % (method, len(work), len(goals) - len(work)), flush=True)
    if not work:
        return 0

    n_workers = max(1, min(workers, len(work)))
    if method == "COLD":
        n_workers = min(n_workers, COLD_MAX_WORKERS)   # fp32 8B -> 1 process only
        print("  [COLD] capped to %d worker(s) (fp32 memory)" % n_workers, flush=True)
    shards = [work[i::n_workers] for i in range(n_workers)]
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for sh in shards:
        p = ctx.Process(target=whitebox_worker, args=(method, sh, existing, result_q))
        p.start()
        procs.append(p)

    n_written = 0
    done_count = 0
    t0 = time.time()
    while done_count < len(procs):
        msg = result_q.get()
        kind = msg[0]
        if kind == "record":
            write_record(msg[1])
            n_written += 1
            if n_written % 10 == 0:
                el = time.time() - t0
                print("  [%s] %d written (%.0fs, %.3f/s)"
                      % (method, n_written, el, n_written / el if el else 0), flush=True)
        elif kind == "log":
            print("  [%s] %s" % (msg[1], msg[2]), file=sys.stderr, flush=True)
        elif kind == "buildfail":
            print("  [%s] BUILD FAILED (shard of %d lost): %s"
                  % (msg[1], msg[3], msg[2]), file=sys.stderr, flush=True)
            done_count += 1
        elif kind == "done":
            _, m, ok, sk, fl = msg
            print("  [%s] worker done: ok=%d skip=%d fail=%d" % (m, ok, sk, fl), flush=True)
            done_count += 1

    for p in procs:
        p.join()
    print("[%s] wrote %d records (%.0fs)" % (method, n_written, time.time() - t0), flush=True)
    return n_written


# Programmatic / API method (single process, in-line)
def run_inline_method(method, goals, existing):
    work = [g for g in goals if (g["goal"].strip().lower(), method) not in existing]
    print("[%s] %d goals to backfill (%d already present)"
          % (method, len(work), len(goals) - len(work)), flush=True)
    if not work:
        return 0
    try:
        run = build_attacker(method, timing=False)
    except Exception as e:
        print("[%s] BUILD FAILED: %s" % (method, str(e)[:300]), file=sys.stderr, flush=True)
        return 0
    n_ok = n_fail = 0
    t0 = time.time()
    for i, g in enumerate(work):
        try:
            query = run(g["goal"])
            if not query or not str(query).strip():
                n_fail += 1
                print("  [%s] EMPTY %s" % (method, g["question_id"]), file=sys.stderr, flush=True)
                continue
            write_record({
                "question_id": g["question_id"],
                "query": str(query).strip(),
                "attack_method": method,
                "goal": g["goal"],
                "source": g["source"],
            })
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print("  [%s] FAILED %s: %s" % (method, g["question_id"], str(e)[:200]),
                  file=sys.stderr, flush=True)
        if (i + 1) % 50 == 0:
            print("  [%s] %d/%d (%.0fs)" % (method, i + 1, len(work), time.time() - t0), flush=True)
    print("[%s] wrote %d records, %d failed (%.0fs)"
          % (method, n_ok, n_fail, time.time() - t0), flush=True)
    return n_ok


# Timing probe: 1 goal per method, writes nothing, full iteration budgets.
FULL_ITERS = {"RandomSearch": 3, "AutoDAN": 250, "COLD": 1000}
# Iteration counts used by the timing probe (reduced for white-box).
PROBE_ITERS = {"RandomSearch": 1, "AutoDAN": 2, "COLD": 20}


def run_timing(methods, goals, workers):
    if not goals:
        print("No goals available for timing.", file=sys.stderr)
        return
    probe_goal = goals[0]
    print("=" * 72)
    print("TIMING PROBE — 1 goal per method (no output written)")
    print("Probe goal [%s/%s]: %s"
          % (probe_goal["source"], probe_goal["question_id"], probe_goal["goal"][:70]))
    print("White-box model: %s | --workers %d" % (WHITE_BOX_MODEL, workers))
    print("=" * 72)

    results = {}  # method -> dict(status, build_s, attack_s, per_goal_full_s, note)
    for method in methods:
        if method == "ArtPrompt" and not os.environ.get("OPENAI_API_KEY"):
            results[method] = dict(status="SKIPPED",
                                   note="needs OPENAI_API_KEY (gpt-3.5-turbo word masking) — not set")
            print("\n[%s] SKIPPED — no OPENAI_API_KEY" % method, flush=True)
            continue
        print("\n[%s] building attacker..." % method, flush=True)
        tb = time.time()
        try:
            run = build_attacker(method, timing=True)
        except Exception as e:
            results[method] = dict(status="BUILD_FAILED", note=str(e)[:300])
            print("[%s] BUILD FAILED: %s" % (method, str(e)[:300]), flush=True)
            continue
        build_s = time.time() - tb
        print("[%s] built in %.1fs; running 1 goal..." % (method, build_s), flush=True)
        ta = time.time()
        try:
            query = run(probe_goal["goal"])
            attack_s = time.time() - ta
        except Exception as e:
            results[method] = dict(status="ATTACK_FAILED", build_s=build_s,
                                   note=str(e)[:300])
            print("[%s] ATTACK FAILED: %s" % (method, str(e)[:300]), flush=True)
            if os.environ.get("BACKFILL_DEBUG"):
                import traceback
                traceback.print_exc()
            continue

        ok = bool(query and str(query).strip())
        # Extrapolate per-goal time to the full budget (build cost amortized).
        per_goal_full = attack_s
        note = ""
        if method in PROBE_ITERS:
            scale = FULL_ITERS[method] / PROBE_ITERS[method]
            per_goal_full = attack_s * scale
            note = ("attack@%d iters=%.1fs -> extrapolated @%d iters=%.1fs"
                    % (PROBE_ITERS[method], attack_s, FULL_ITERS[method], per_goal_full))
        results[method] = dict(status="OK" if ok else "EMPTY_OUTPUT",
                               build_s=build_s, attack_s=attack_s,
                               per_goal_full_s=per_goal_full, note=note,
                               sample=str(query)[:80] if ok else "")
        print("[%s] attack=%.2fs  output_ok=%s  %s"
              % (method, attack_s, ok, note), flush=True)

    # ---- Summary table ----
    print("\n" + "=" * 72)
    print("TIMING SUMMARY (per-goal wall-time + 476-goal extrapolation)")
    print("=" * 72)
    header = ("%-13s %-12s %10s %12s %14s %14s"
              % ("method", "status", "per-goal", "x476 (1proc)", "x476 (/%d)" % workers, "kind"))
    print(header)
    print("-" * len(header))
    for method in methods:
        r = results.get(method, {})
        st = r.get("status", "?")
        if st in ("OK", "EMPTY_OUTPUT"):
            pg = r.get("per_goal_full_s", r.get("attack_s", 0.0))
            x476 = pg * N_MISSING_GOALS
            kind = "white-box" if method in WHITE_BOX else "cpu/api"
            if method in WHITE_BOX:
                x476_par = x476 / workers
            else:
                x476_par = x476  # cpu/api not GPU-sharded the same way
            print("%-13s %-12s %9.2fs %11.0fs %13.0fs %14s"
                  % (method, st, pg, x476, x476_par, kind))
        else:
            print("%-13s %-12s %10s %12s %14s %14s"
                  % (method, st, "-", "-", "-", r.get("note", "")[:14]))
    print("-" * len(header))
    print("\nNotes:")
    for method in methods:
        r = results.get(method, {})
        if r.get("note"):
            print("  [%s] %s" % (method, r["note"]))
        if r.get("status") not in ("OK", None):
            print("  [%s] STATUS=%s %s" % (method, r.get("status"), r.get("note", "")))
    return results


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    choices=ALL_METHODS, help="subset of attacks (default: all 6)")
    ap.add_argument("--timing", action="store_true",
                    help="run each method on 1 goal, print timing, write nothing")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel processes for white-box methods (default 8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of goals processed per method")
    args = ap.parse_args()

    goals = load_missing_goals()
    print("Loaded %d missing goals (%d HarmBench + %d Sorry-Bench expected = %d)"
          % (len(goals),
             sum(1 for g in goals if g["source"] == "harmbench"),
             sum(1 for g in goals if g["source"] == "sorry-bench"),
             N_MISSING_GOALS), flush=True)
    if args.limit:
        goals = goals[:args.limit]

    if args.timing:
        run_timing(args.methods, goals, args.workers)
        return

    existing = load_existing_pairs()
    print("Output: %s (%d (goal,method) pairs already present)"
          % (OUT_PATH, len(existing)), flush=True)

    grand = 0
    for method in args.methods:
        if method == "ArtPrompt" and not os.environ.get("OPENAI_API_KEY"):
            print("[ArtPrompt] SKIPPED — needs OPENAI_API_KEY (gpt-3.5-turbo word masking)",
                  file=sys.stderr, flush=True)
            continue
        if method in WHITE_BOX:
            grand += run_whitebox_method(method, goals, existing, args.workers)
        else:
            grand += run_inline_method(method, goals, existing)
    print("\nTOTAL written this run: %d records -> %s" % (grand, OUT_PATH), flush=True)


if __name__ == "__main__":
    main()
