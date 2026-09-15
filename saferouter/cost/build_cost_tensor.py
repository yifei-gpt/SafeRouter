#!/usr/bin/env python3
"""The per-probe adversarial cost tensor, from actual token counts rather than
averages. Atom calls, X in {A,B,C} = generation under {S0,S1,S2}: target_X,
harm_X (S4), refusal_X (S5 gate), s5_* (S5 full path); query-level: paraphrase
(S2), qwen3guard (S3).

    python -m cost.build_cost_tensor [--atoms PATH --qatoms PATH --out PATH]
    -> adversarial_cost_tensor.pt: (N, N_MODELS, N_COMPOSITES) $/1000q + keys
"""
import argparse
import json
from pathlib import Path

import torch

from . import (
    MODELS, N_MODELS, COMPOSITES, N_COMPOSITES, COMPOSITE_TO_IDX,
    FULL_TO_SHORT, MODEL_INPUT_PRICES,
    call_cost,
)

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"

# Atom ids must match the embedding keys, or the join yields all zeros.
DEFAULT_ATOMS = DATA / "adversarial" / "probe" / "atoms.jsonl"
DEFAULT_QATOMS = DATA / "adversarial" / "probe" / "query_atoms.jsonl"
DEFAULT_EMB = DATA / "embeddings" / "adversarial_full_embeddings.pt"
DEFAULT_OUT = DATA / "embeddings" / "adversarial_cost_tensor.pt"


def _call_cost_1k(c):
    """call_cost for a single call record, returned in $/1000q."""
    if c is None:
        return 0.0
    return call_cost(c.get("model", ""), c.get("in", 0), c.get("out", 0)) * 1e3


def build(atoms_path, qatoms_path, emb_path, out_path):
    emb_data = torch.load(emb_path, map_location="cpu", weights_only=False)
    keys = emb_data["keys"]  # list of (query_id, attack_method)
    key_to_idx = {k: i for i, k in enumerate(keys)}
    N = len(keys)

    # Query helpers depend on the WRAPPED query -> key on (query_id, method).
    print("Loading query_atoms...")
    query_helper_costs = {}        # (query_id, attack_method) -> {para, guard}
    query_helper_by_qid = {}       # query_id -> any one method's helpers (fallback)
    n_qa = 0
    for line in open(qatoms_path):
        qa = json.loads(line); n_qa += 1
        calls = {c["name"]: c for c in qa.get("calls", [])}
        h = {"para": _call_cost_1k(calls.get("paraphrase")),
             "guard": _call_cost_1k(calls.get("qwen3guard"))}
        query_helper_costs[(qa["query_id"], qa.get("attack_method"))] = h
        query_helper_by_qid.setdefault(qa["query_id"], h)
    print(f"  {n_qa} query atoms ({len(query_helper_by_qid)} distinct query_ids)")

    print("Loading atoms and computing per-probe costs...")
    model_to_idx = {m: i for i, m in enumerate(MODELS)}
    cost_tensor = torch.zeros(N, N_MODELS, N_COMPOSITES)

    # Per-model average S5 cost, applied per-slot only when gated.
    from collections import defaultdict
    _s5bt, _s5rt, _s5ri = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in open(atoms_path):
        a = json.loads(line)
        ms = FULL_TO_SHORT.get(a["target_model"])
        if ms is None:
            continue
        for c in a.get("calls", []):
            nm = c.get("name", "")
            if nm.startswith("s5_backtrans"):
                _s5bt[ms].append(_call_cost_1k(c))
            elif nm.startswith("s5_requery_target"):
                _s5rt[ms].append(_call_cost_1k(c))
            elif nm.startswith("s5_requery_inline"):
                _s5ri[ms].append(_call_cost_1k(c))
    def _avg(d, ms):
        v = d.get(ms)
        return sum(v) / len(v) if v else 0.0
    avg_s5_full = {ms: _avg(_s5bt, ms) + _avg(_s5rt, ms) + _avg(_s5ri, ms)
                   for ms in set(list(_s5bt) + list(_s5rt) + list(_s5ri))}

    n_filled = 0
    n_s5_fallback = 0  # atoms where S5 triple-count != gated-slot-count -> avg fallback
    for line in open(atoms_path):
        atom = json.loads(line)
        qid = atom["query_id"]
        method = atom["attack_method"]
        probe_key = (qid, method)
        pi = key_to_idx.get(probe_key)
        if pi is None:
            continue

        model_full = atom["target_model"]
        model_short = FULL_TO_SHORT.get(model_full)
        if model_short is None:
            continue
        mi = model_to_idx.get(model_short)
        if mi is None:
            continue

        calls = {c["name"]: c for c in atom.get("calls", [])}
        helpers = (query_helper_costs.get((qid, method))
                   or query_helper_by_qid.get(qid)
                   or {"para": 0.0, "guard": 0.0})

        # ---- Generation costs: A = S0, B = S1 (SCR), C = S2 (paraphrase) ----
        gen_A = _call_cost_1k(calls.get("target_A"))
        gen_B = _call_cost_1k(calls.get("target_B"))
        gen_C = _call_cost_1k(calls.get("target_C"))

        # ---- S1 overhead = actual extra input tokens × model input price ----
        s1_extra = max(calls.get("target_B", {}).get("in", 0) -
                       calls.get("target_A", {}).get("in", 0), 0)
        s1_overhead = s1_extra * MODEL_INPUT_PRICES[model_short] / 1e6 * 1e3

        # ---- S4 judge costs, matched to pre-defense (harm_A judges S0) ----
        harm_A = _call_cost_1k(calls.get("harm_A"))
        harm_B = _call_cost_1k(calls.get("harm_B"))
        harm_C = _call_cost_1k(calls.get("harm_C"))

        # S5's gate judge always runs; the rest only when NOT a refusal.
        refusal_A = _call_cost_1k(calls.get("refusal_A"))
        refusal_B = _call_cost_1k(calls.get("refusal_B"))
        refusal_C = _call_cost_1k(calls.get("refusal_C"))

        # ---- S6 PARDEN repeats: one target call per pre-defense response ----
        parden = {"A": _call_cost_1k(calls.get("parden_repeat_A")),
                  "B": _call_cost_1k(calls.get("parden_repeat_B")),
                  "C": _call_cost_1k(calls.get("parden_repeat_C"))}

        # S5 calls are not slot-labeled: map ordered triples in A,B,C order.
        s5_gated = {
            sl: bool(atom.get(f"inline_harm_{sl}")) or (not bool(atom.get(f"inline_refusal_{sl}")))
            for sl in ("A", "B", "C")
        }
        _bt = [c for c in atom.get("calls", []) if c.get("name") == "s5_backtrans"]
        _rt = [c for c in atom.get("calls", []) if c.get("name") == "s5_requery_target"]
        _ri = [c for c in atom.get("calls", []) if c.get("name") == "s5_requery_inline"]
        _gated_slots = [sl for sl in ("A", "B", "C") if s5_gated[sl]]
        s5_tail = {}
        if len(_bt) == len(_rt) == len(_ri) == len(_gated_slots):
            for _i, _sl in enumerate(_gated_slots):
                s5_tail[_sl] = (_call_cost_1k(_bt[_i]) + _call_cost_1k(_rt[_i])
                                + _call_cost_1k(_ri[_i]))
        else:
            _s5_avg = avg_s5_full.get(model_short, 0.0)
            for _sl in _gated_slots:
                s5_tail[_sl] = _s5_avg
            n_s5_fallback += 1

        para_c = helpers["para"]
        guard_c = helpers["guard"]

        # ---- Fill all 16 composites (4 pre × 4 post) ----
        for ci, (pre, post) in enumerate(COMPOSITES):
            # Pre-gen: pick the target generation + matched post-helper slot.
            if pre == "s0":
                pre_ov = 0.0
                gen = gen_A
                harm = harm_A
                refusal_gate = refusal_A
                slot = "A"
            elif pre == "s1":
                # The actual S1 gen (target_B), else S0 gen + input-only overhead.
                if gen_B > 0:
                    pre_ov = 0.0
                    gen = gen_B
                else:
                    pre_ov = s1_overhead
                    gen = gen_A
                harm = harm_B
                refusal_gate = refusal_B
                slot = "B"
            elif pre == "s2":
                pre_ov = para_c
                gen = gen_C
                harm = harm_C
                refusal_gate = refusal_C
                slot = "C"
            elif pre == "s3":
                pre_ov = guard_c
                gen = gen_A  # S3 pass-through: same gen as S0
                harm = harm_A
                refusal_gate = refusal_A
                slot = "A"
            else:
                pre_ov = 0.0
                gen = gen_A
                harm = harm_A
                refusal_gate = refusal_A
                slot = "A"

            # Post-gen
            if post == "s0":
                post_ov = 0.0
            elif post == "s4":
                post_ov = harm                       # S4 = one harmfulness judge
            elif post == "s5":
                # Refusal gate always; full path (this slot's OWN tokens) only when gated.
                post_ov = refusal_gate + s5_tail.get(slot, 0.0)
            elif post == "s6":
                # S6 PARDEN: one target self-repeat call (this slot's OWN tokens).
                post_ov = parden.get(slot, 0.0)
            else:
                post_ov = 0.0

            cost_tensor[pi, mi, ci] = pre_ov + gen + post_ov
            n_filled += 1

    total = N * N_MODELS * N_COMPOSITES
    print(f"  Filled {n_filled}/{total} cells ({n_filled/total*100:.1f}%)")
    print(f"  S5 per-query tail: {n_s5_fallback} atoms used the per-model avg fallback "
          f"(triple/gated-slot count mismatch); rest charged actual per-slot tokens")
    print(f"  Range: [${cost_tensor[cost_tensor > 0].min():.4f}, ${cost_tensor.max():.4f}]/1000q")

    # Spot-check: per-strategy overheads are query-dependent (non-constant).
    s0_idx = COMPOSITE_TO_IDX[("s0", "s0")]
    s1_idx = COMPOSITE_TO_IDX[("s1", "s0")]
    s4_idx = COMPOSITE_TO_IDX[("s0", "s4")]
    s5_idx = COMPOSITE_TO_IDX[("s0", "s5")]
    for label, ci in [("S1", s1_idx), ("S4", s4_idx), ("S5", s5_idx)]:
        diff = cost_tensor[:, 0, ci] - cost_tensor[:, 0, s0_idx]
        nonzero = diff[diff > 0]
        if len(nonzero) > 0:
            print(f"  {label} overhead (model 0): mean=${nonzero.mean():.6f} std=${nonzero.std():.6f}")

    torch.save({
        "cost_tensor": cost_tensor,
        "keys": keys,
        "models": MODELS,
        "composites": COMPOSITES,
    }, out_path)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--atoms", type=str, default=str(DEFAULT_ATOMS))
    p.add_argument("--qatoms", type=str, default=str(DEFAULT_QATOMS))
    p.add_argument("--embeddings", type=str, default=str(DEFAULT_EMB))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()
    build(args.atoms, args.qatoms, args.embeddings, args.out)
