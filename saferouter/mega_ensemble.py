#!/usr/bin/env python3
"""Pool saved per-fold nets across runs and compute the leakage-free honest frontier
(τ* on optval, evaluated on test). All runs must share folds/optval (same
--fold-seed) and be saved with --save-nets.

    python mega_ensemble.py <out_dir> <run_dir1> <run_dir2> ...
"""
import argparse
import json
from pathlib import Path

import torch

from cost import COST_MATRIX
from data_io import load_safety_data
from evaluate import (OP_ASR_TARGET, ensemble_evaluate, ensemble_threshold_sweep,
                      select_tau)
from routers import load_fold_nets

# The scalar test_* fields are written at τ* chosen on optval here, matching
# run_kfold; the stored sweep rebuilds the full frontier.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    adv_embs, keys, safety_tensor, probe_costs = load_safety_data()
    adv_embs = adv_embs.to(dev)
    safety_tensor = safety_tensor.to(dev)
    probe_costs = probe_costs.to(dev) if probe_costs is not None else None

    results, fi = [], 0
    while True:
        nets, test_idx, optval_idx, dropped = load_fold_nets(
            args.runs, fi, device=dev, verbose=True)
        if test_idx is None:                       # no run has this fold -> done
            break
        if not nets:
            raise RuntimeError(f"fold {fi}: every net was non-finite — nothing to ensemble")
        print(f"fold {fi}: pooled {len(nets)} nets from {len(args.runs)} configs"
              + (f"  ({dropped} diverged net(s) dropped)" if dropped else ""))
        sweep = ensemble_threshold_sweep(nets, adv_embs, safety_tensor, test_idx, COST_MATRIX,
                                            device=dev, probe_costs=probe_costs, use_cost_head=True)
        opt_sweep = ensemble_threshold_sweep(nets, adv_embs, safety_tensor, optval_idx, COST_MATRIX,
                                               device=dev, probe_costs=probe_costs, use_cost_head=True)
        # τ* from OPTVAL at the a-priori target, then TEST read once at it (mirrors run_kfold).
        tau_star = select_tau(opt_sweep, OP_ASR_TARGET, len(optval_idx))
        ens = ensemble_evaluate(nets, adv_embs, safety_tensor, test_idx, COST_MATRIX,
                                  device=dev, probe_costs=probe_costs, safety_threshold=tau_star,
                                  use_cost_head=True)
        results.append({"fold": fi, "tau_star": tau_star,
                        "test_asr": ens["asr"], "test_jb": ens["jb"],
                        "test_total": ens["total"], "test_realized_cost": ens["avg_realized_cost"],
                        "threshold_sweep": sweep, "opt_sweep": opt_sweep})
        fi += 1

    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(results, open(Path(args.out) / "sop_results.json", "w"), default=str)
    print(f"saved {len(results)} folds -> {args.out}/sop_results.json")


if __name__ == "__main__":
    main()
