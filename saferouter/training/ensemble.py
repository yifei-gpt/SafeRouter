"""Pool saved per-fold nets across runs into the leakage-free honest frontier.

Every run must share folds and optval (the same --fold-seed) and be saved with
--save-nets. τ* is chosen on optval at the a-priori target and test is read
once at it, as run_kfold does; the stored sweep rebuilds the full frontier.
"""
from cost import COST_MATRIX
from routers import load_fold_nets
from utils.data_io import load_safety_data
from utils.evaluate import (OP_ASR_TARGET, ensemble_evaluate, ensemble_threshold_sweep,
                            select_tau)


def pool_folds(runs, device, verbose=True):
    """-> one record per fold: τ*, the test read at it, and both sweeps."""
    adv_embs, keys, safety_tensor, probe_costs = load_safety_data()
    adv_embs = adv_embs.to(device)
    safety_tensor = safety_tensor.to(device)
    probe_costs = probe_costs.to(device) if probe_costs is not None else None

    results, fi = [], 0
    while True:
        nets, test_idx, optval_idx, dropped = load_fold_nets(runs, fi, device=device,
                                                             verbose=verbose)
        if test_idx is None:                       # no run has this fold -> done
            break
        if not nets:
            raise RuntimeError(f"fold {fi}: every net was non-finite — nothing to ensemble")
        if verbose:
            print(f"fold {fi}: pooled {len(nets)} nets from {len(runs)} configs"
                  + (f"  ({dropped} diverged net(s) dropped)" if dropped else ""))
        sweep = ensemble_threshold_sweep(nets, adv_embs, safety_tensor, test_idx, COST_MATRIX,
                                            device=device, probe_costs=probe_costs, use_cost_head=True)
        opt_sweep = ensemble_threshold_sweep(nets, adv_embs, safety_tensor, optval_idx, COST_MATRIX,
                                               device=device, probe_costs=probe_costs, use_cost_head=True)
        tau_star = select_tau(opt_sweep, OP_ASR_TARGET, len(optval_idx))
        ens = ensemble_evaluate(nets, adv_embs, safety_tensor, test_idx, COST_MATRIX,
                                  device=device, probe_costs=probe_costs, safety_threshold=tau_star,
                                  use_cost_head=True)
        results.append({"fold": fi, "tau_star": tau_star,
                        "test_asr": ens["asr"], "test_jb": ens["jb"],
                        "test_total": ens["total"], "test_realized_cost": ens["avg_realized_cost"],
                        "threshold_sweep": sweep, "opt_sweep": opt_sweep})
        fi += 1
    return results
