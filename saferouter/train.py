#!/usr/bin/env python3
"""Train every router. One subcommand per thing you can train.

    python train.py saferouter [flags]        # ours; see scripts/train.sh
    python train.py baselines [--only mirt]   # RouteLLM, CARROT, IRT-Router
    python train.py ensemble OUT RUN...       # pool saved per-fold nets

The saferouter defaults below are a SMOKE TEST; scripts/train.sh passes the
pre-registered flags.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cost import COST_MATRIX
from routers import SafetyOutcomePredictor
from training import saferouter as SR
from training.bilinear_mf import train_bilinear_mf
from training.carrot_knn import train_carrot_knn
from training.ensemble import pool_folds
from training.mirt import train_mirt
from training.saferouter import run_kfold, train_one_fold
from utils.data_io import (CKPT_DIR, DATA, build_attack_method_index, load_baseline_data,
                           load_benign_data, load_safety_data)
from utils.evaluate import OP_ASR_TARGET, threshold_sweep


def add_saferouter_flags(parser):
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs (paper recipe uses 100)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=3, help="Backbone depth (3=best)")
    parser.add_argument("--dropout", type=float, default=0.30, help="Backbone dropout (0.30=best)")
    parser.add_argument("--focal-gamma", type=float, default=0.5, help="Focal loss gamma (0.5=best)")
    parser.add_argument("--rank-weight", type=float, default=0.3, help="Ranking loss weight")
    parser.add_argument("--cost-train", type=float, default=0.5,
                        help="Cost-aware ranking weight (operates on ground-truth-safe cells only). "
                             "NOTE: this term lives inside loss_rank, which is then scaled by "
                             "--rank-weight, so the EFFECTIVE cost-ranking weight is "
                             "rank_weight * cost_train (e.g. 0.3 * 1.0 = 0.3 with the paper recipe), "
                             "not the raw flag value.")
    parser.add_argument("--kfold", type=int, default=0, help="K-fold CV (0=random split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42,
                        help="Seed for fold assignment (42=original; paper recipe uses 100, which spreads hard attacks)")
    parser.add_argument("--hard-benign-emb", type=str, default=None,
                        help="Path to a .pt of sensitive-but-legitimate query embeddings; "
                             "added to the risk head's benign class (risk-BCE only, never "
                             "safety/quality/cost) so the gate stops flagging sensitive-legit "
                             "content as adversarial. MUST be disjoint from any over-refusal eval set.")
    parser.add_argument("--cost-focal-alpha", type=float, default=0.0,
                        help="Cost-weighted focal: upweight cheap cells")
    parser.add_argument("--asr-target", type=float, default=0.01,
                        help="Target ASR for Lagrangian constraint (ε); paper recipe uses ~0.004")
    parser.add_argument("--lambda-lr", type=float, default=0.01,
                        help="Learning rate for Lagrange multiplier")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds per fold for ensemble")
    # ---- Per-query cost head + calibration + diagram search ----
    parser.add_argument("--safety-arch", type=str, default="flat",
                        choices=["flat", "bilinear"],
                        help="Safety-head diagram (all add a zero-init correction to the flat head)")
    parser.add_argument("--cost-pred-weight", type=float, default=0.0,
                        help="Weight for the per-query cost-regression head (0 disables it)")
    parser.add_argument("--use-cost-head", action="store_true",
                        help="Decide cheap-first selection by predicted per-query cost (needs cost-pred-weight>0)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit a per-net temperature for P(safe) on the val split")
    parser.add_argument("--calib-mode", type=str, default="scalar", choices=["scalar", "vector", "isotonic"],
                        help="Temperature calibration: one shared T or per-(model,defense) T")
    parser.add_argument("--safety-loss", type=str, default="focal",
                        choices=["focal", "bce", "inverse_focal"],
                        help="Safety-head loss reweighting (inverse_focal helps tail calibration)")
    parser.add_argument("--unsafe-weight", type=float, default=1.0,
                        help="Asymmetric upweight on unsafe-cell loss (targets false-safe ASR)")
    parser.add_argument("--ckpt-mode", type=str, default="asr", choices=["asr", "constraint"],
                        help="Checkpoint metric: best val ASR, or min cost s.t. val ASR<ckpt-asr-target")
    parser.add_argument("--ckpt-asr-target", type=float, default=0.008,
                        help="Feasibility threshold for --ckpt-mode constraint")
    parser.add_argument("--op-asr-target", type=float, default=OP_ASR_TARGET,
                        help="Operating-point ASR target; τ* picked on optval (min cost s.t. ASR<this)")
    parser.add_argument("--save-nets", action="store_true",
                        help="Save per-fold ensemble nets (for cross-config mega-ensembling)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default=str(DATA / "checkpoints" / "sop"))


def train_saferouter(args):
    if args.use_cost_head and args.cost_pred_weight <= 0:
        raise SystemExit("--use-cost-head requires --cost-pred-weight > 0 "
                         "(the cost head would be untrained otherwise)")

    print("=" * 60)
    print("Safety-Outcome Predictor")
    print("=" * 60)

    print("\nLoading safety data...")
    adv_embs, keys, safety_tensor, probe_costs = load_safety_data()

    print("Loading benign data...")
    ben_embs, ben_qids, ben_quality, ben_costs, ben_train_idx, ben_test_idx = load_benign_data()

    # Preload: the batch indices below are built on args.device.
    if args.device != "cpu":
        adv_embs = adv_embs.to(args.device)
        safety_tensor = safety_tensor.to(args.device)
        ben_embs = ben_embs.to(args.device)
        ben_quality = ben_quality.to(args.device)
        if probe_costs is not None:
            probe_costs = probe_costs.to(args.device)

    if args.hard_benign_emb:
        hb = torch.load(args.hard_benign_emb, map_location="cpu", weights_only=False)
        SR.HARD_BEN_EMBS = hb["embeddings"].float().to(args.device)
        print(f"  Hard-benign risk negatives: {len(SR.HARD_BEN_EMBS)} sensitive-but-legitimate "
              f"queries added to the gate's benign class ← {args.hard_benign_emb}")

    idx_to_method = build_attack_method_index(keys)

    n_params = sum(p.numel() for p in SafetyOutcomePredictor(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout,
        safety_arch=args.safety_arch).parameters())
    print(f"\nModel: arch={args.safety_arch}, hidden={args.hidden_dim}, "
          f"cost_head={'on' if args.cost_pred_weight>0 else 'off'}, calibrate={args.calibrate}, "
          f"{n_params:,} params")
    print(f"  n_seeds={args.n_seeds}")

    if args.kfold > 0:
        run_kfold(args, adv_embs, safety_tensor, ben_embs, ben_quality,
                  ben_train_idx, ben_test_idx, idx_to_method, probe_costs=probe_costs)
        return

    # Random 85/15 split
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N = len(keys)
    rng = np.random.RandomState(42)
    perm = np.arange(N)
    rng.shuffle(perm)
    split = int(0.85 * N)
    train_idx = perm[:split]
    test_idx = perm[split:]

    net = train_one_fold(
        args, adv_embs, safety_tensor, ben_embs, ben_quality,
        ben_train_idx, train_idx, probe_costs, args.seed)

    # Threshold sweep
    print("\nThreshold sweep:")
    sweep = threshold_sweep(net, adv_embs, safety_tensor, test_idx,
                           COST_MATRIX, device=args.device, probe_costs=probe_costs)
    for t, asr, cost in sweep:
        print(f"  thresh={t:.3f}: ASR={asr*100:.2f}% cost=${cost:.4f}")


def train_baselines(only):
    """The three quality-cost baselines, all on the shared 85/15 benign split."""
    data = load_baseline_data()
    if "bimf" in only:
        (CKPT_DIR / "router").mkdir(exist_ok=True, parents=True)
        train_bilinear_mf(data, save_path=CKPT_DIR / "router" / "quality_router.pt")
    if "carrot" in only:
        train_carrot_knn(data, save_path=CKPT_DIR / "carrot_knn.joblib")
    if "mirt" in only:
        train_mirt(data, save_path=CKPT_DIR / "mirt.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("saferouter", help="train SafeRouter")
    add_saferouter_flags(p)

    p = sub.add_parser("baselines", help="train RouteLLM / CARROT / IRT-Router")
    p.add_argument("--only", nargs="+", choices=["bimf", "carrot", "mirt"],
                   default=["bimf", "carrot", "mirt"])

    p = sub.add_parser("ensemble", help="pool per-fold nets across runs")
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    if args.cmd == "saferouter":
        train_saferouter(args)
    elif args.cmd == "baselines":
        train_baselines(args.only)
        print("\nAll done.")
    else:
        results = pool_folds(args.runs, args.device)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        json.dump(results, open(Path(args.out) / "sop_results.json", "w"), default=str)
        print(f"saved {len(results)} folds -> {args.out}/sop_results.json")


if __name__ == "__main__":
    main()
