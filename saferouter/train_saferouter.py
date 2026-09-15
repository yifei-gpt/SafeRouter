#!/usr/bin/env python3
"""SafeRouter: security-aware routing with cost-conditioned defense selection.
Four heads on a shared frozen Qwen3-Embedding-0.6B backbone -- safety
P(safe|q,m,d,cost), cost, risk P(adversarial|q), quality for benign routing:

    z = backbone(query);  z_cell = concat(z, model_emb, defense_emb, cost_feat)
    safety_logit = cell_mlp(z_cell)        # all 10 x 16 = 160 cells at once

Losses: inverse-focal + pairwise ranking + a Lagrangian constraint (min E[cost]
s.t. ASR < eps); risk BCE; quality MSE. The defaults below are a SMOKE TEST --
reproduce with scripts/train.sh, which passes the pre-registered flags.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model pool, composites and pricing live in cost/; duplicating them here is how
# the reported dollars would silently drift from cost/price.json.
from routers import SafetyOutcomePredictor
from routers.saferouter import _fit_isotonic, _fit_temperature
from data_io import build_attack_method_index, load_benign_data, load_safety_data, make_folds
from evaluate import (OP_ASR_TARGET, ensemble_evaluate, ensemble_threshold_sweep,
                      evaluate, evaluate_risk_gate, select_tau, threshold_sweep)
from cost import COST_MATRIX

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

QUALITY_WEIGHT = 0.5      # benign quality MSE, trained jointly into the backbone


# ─── Data paths ───
# --hard-benign-emb: risk-head negatives teaching that sensitive != adversarial.
HARD_BEN_EMBS = None


# ─── Data loading ───

# ─── Model ───

# ─── Training ───

def train_epoch(net, optimizer, adv_embs, safety_tensor,
                ben_embs, ben_quality, ben_train_idx,
                adv_train_idx, batch_size=512, device="cuda",
                focal_gamma=0.5, rank_weight=0.3, cost_train_weight=0.0,
                probe_costs=None, asr_target=0.01,
                cost_focal_alpha=0.0, cost_pred_weight=0.0, safety_loss="focal",
                unsafe_weight=1.0):
    """One epoch. Loss = focal_safety + ranking + risk_bce + quality_mse
         + λ_detach·max(0, batch_asr−ε)   (primal: θ reduces ASR)
         − λ·max(0, batch_asr−ε)_detach   (dual: λ rises while ASR > ε)
    All data must already be on `device`."""
    net.train()

    # Cost-weighted terms need per-probe costs; fail fast, not a TypeError in the loss.
    if (cost_focal_alpha > 0 or cost_train_weight > 0 or cost_pred_weight > 0) and probe_costs is None:
        raise ValueError(
            "cost_focal_alpha>0 or cost_train_weight>0 or cost_pred_weight>0 requires "
            "probe_costs, but none were loaded. Build it with `python -m cost.build_cost_tensor` "
            "(writes adversarial_cost_tensor.pt), or set the cost weights to 0.")

    # Pre-build shuffled index tensors on GPU — avoids numpy→torch per batch
    adv_perm = torch.tensor(adv_train_idx[np.random.permutation(len(adv_train_idx))],
                            dtype=torch.long, device=device)
    ben_perm = torch.tensor(ben_train_idx[np.random.permutation(len(ben_train_idx))],
                            dtype=torch.long, device=device)

    total_loss = 0.0
    n_steps = 0
    n_pairs = 5  # random pairs per probe for ranking loss
    _zero = torch.tensor(0.0, device=device)

    n_adv_batches = (len(adv_perm) + batch_size - 1) // batch_size
    n_ben_batches = (len(ben_perm) + batch_size - 1) // batch_size
    n_batches = max(n_adv_batches, n_ben_batches)

    for b in range(n_batches):
        # Batch indices (cycle shorter dataset)
        a_start = (b % n_adv_batches) * batch_size
        batch_adv = adv_perm[a_start:a_start + batch_size]
        b_start = (b % n_ben_batches) * batch_size
        batch_ben = ben_perm[b_start:b_start + batch_size]

        B = len(batch_adv)
        if B == 0:
            continue

        # ── Forward pass (adversarial) ──
        safety_logits, risk_logit = net(adv_embs[batch_adv])
        safety_targets = safety_tensor[batch_adv]  # (B, M, D)

        # ── Safety loss (focal / plain BCE / inverse-focal for tail calibration) ──
        valid_mask = (safety_targets >= 0)
        targets_clamped = safety_targets.clamp(0, 1)
        p = torch.sigmoid(safety_logits)
        bce = F.binary_cross_entropy_with_logits(
            safety_logits, targets_clamped, reduction='none')
        pt = p * targets_clamped + (1 - p) * (1 - targets_clamped)
        if safety_loss == "bce":
            focal = bce
        elif safety_loss == "inverse_focal":
            # up-weights confident-correct cells -> better tail calibration (2505.23463)
            focal = ((1 + pt) ** focal_gamma) * bce
        else:  # "focal" (default): down-weights easy cells
            focal = ((1 - pt) ** focal_gamma) * bce
        if cost_focal_alpha > 0:
            cost_w = 1.0 / (1.0 + cost_focal_alpha * probe_costs[batch_adv])
            focal = focal * cost_w
        if unsafe_weight != 1.0:
            # asymmetric: upweight unsafe cells, the false-safe ASR source
            focal = focal * torch.where(targets_clamped < 0.5,
                                        focal.new_tensor(unsafe_weight), focal.new_tensor(1.0))
        loss_safety = focal[valid_mask].mean() if valid_mask.any() else _zero

        # ── Pairwise ranking loss (vectorized) ──
        flat_logits = safety_logits.view(B, -1)  # (B, 120)
        flat_safe = ((safety_targets > 0.5) & valid_mask).view(B, -1)
        flat_unsafe = ((safety_targets < 0.5) & valid_mask).view(B, -1)

        n_safe = flat_safe.sum(dim=1)
        n_unsafe = flat_unsafe.sum(dim=1)
        has_both = (n_safe > 0) & (n_unsafe > 0)

        loss_rank = _zero
        if has_both.any():
            vi = has_both.nonzero(as_tuple=True)[0]
            safe_samples = torch.multinomial(flat_safe[vi].float(), n_pairs, replacement=True)
            unsafe_samples = torch.multinomial(flat_unsafe[vi].float(), n_pairs, replacement=True)
            v_logits = flat_logits[vi]
            margins = F.relu(1.0 - (v_logits.gather(1, safe_samples) -
                                     v_logits.gather(1, unsafe_samples)))
            loss_rank = margins.mean()

        # ── Cost-aware ranking: cheap safe > expensive safe ──
        if cost_train_weight > 0:
            flat_cost = probe_costs[batch_adv].view(B, -1)
            has_multi_safe = (n_safe > 1)
            if has_multi_safe.any():
                mi = has_multi_safe.nonzero(as_tuple=True)[0]
                mi_safe = flat_safe[mi]
                mi_cost = flat_cost[mi].clone()
                mi_cost[~mi_safe] = 1e9
                cheap_idx = mi_cost.argmin(dim=1)

                mi_cost_exp = flat_cost[mi].clone()
                mi_cost_exp[~mi_safe] = -1e9
                expensive_idx = mi_cost_exp.argmax(dim=1)

                mi_logits = flat_logits[mi]
                cheap_logits = mi_logits.gather(1, cheap_idx.unsqueeze(1)).squeeze(1)
                expensive_logits = mi_logits.gather(1, expensive_idx.unsqueeze(1)).squeeze(1)
                cheap_c = flat_cost[mi].gather(1, cheap_idx.unsqueeze(1)).squeeze(1)
                expensive_c = flat_cost[mi].gather(1, expensive_idx.unsqueeze(1)).squeeze(1)

                cost_diff_mask = cheap_c < expensive_c * 0.5
                if cost_diff_mask.any():
                    cost_margins = F.relu(0.5 - (cheap_logits[cost_diff_mask] -
                                                  expensive_logits[cost_diff_mask]))
                    # cost_train_weight applies inside loss_rank: the effective weight is the product.
                    loss_rank = loss_rank + cost_train_weight * cost_margins.mean()

        # ── Risk loss ──
        loss_risk_adv = F.binary_cross_entropy_with_logits(
            risk_logit, torch.ones_like(risk_logit))

        # ── Benign: risk loss + quality loss ──
        loss_risk_ben = loss_quality = _zero
        if len(batch_ben) > 0:
            ben_z = net.backbone(ben_embs[batch_ben])
            loss_quality = F.mse_loss(net.predict_quality(ben_z), ben_quality[batch_ben])
            loss_risk_ben = F.binary_cross_entropy_with_logits(
                net.risk_head(ben_z), torch.zeros(len(batch_ben), 1, device=device))

            # Hard-benign negatives (sensitive but legitimate): risk BCE only.
            if HARD_BEN_EMBS is not None:
                hb_idx = torch.randint(len(HARD_BEN_EMBS), (len(batch_ben),), device=device)
                hb_z = net.backbone(HARD_BEN_EMBS[hb_idx])
                loss_risk_ben = loss_risk_ben + F.binary_cross_entropy_with_logits(
                    net.risk_head(hb_z), torch.zeros(len(hb_idx), 1, device=device))


        # Lagrangian soft-ASR surrogate = mean P(safe) over UNSAFE cells, not 1 - that.
        unsafe_mask = valid_mask & (targets_clamped < 0.5)
        if unsafe_mask.any():
            batch_asr_soft = p[unsafe_mask].mean()
        else:
            batch_asr_soft = _zero
        lam = F.softplus(net.log_lambda).clamp(max=10.0)
        violation = F.relu(batch_asr_soft - asr_target)
        lagrangian_primal = lam.detach() * violation
        lagrangian_dual = -lam * violation.detach()

        # ── Combined loss ──
        loss = (loss_safety + rank_weight * loss_rank
                + 0.5 * (loss_risk_adv + loss_risk_ben) + QUALITY_WEIGHT * loss_quality
                + lagrangian_primal + lagrangian_dual)

        # ── Cost head: per-query pipeline cost, detached backbone ──
        if cost_pred_weight > 0 and valid_mask.any():
            with torch.no_grad():
                z_c = net.backbone(adv_embs[batch_adv])          # frozen features (no backbone grad)
            cost_pred = net.cost_head(z_c).view(B, net.n_models, net.n_composites)
            cost_tgt = net.norm_cost(probe_costs[batch_adv])
            loss = loss + cost_pred_weight * F.smooth_l1_loss(
                cost_pred[valid_mask], cost_tgt[valid_mask])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


# ─── Evaluation ───

# ─── K-fold ───

def train_one_fold(args, adv_embs, safety_tensor, ben_embs, ben_quality,
                   ben_train_idx, train_idx, probe_costs, seed,
                   exclude_idx=None):
    """Train one seed of one fold. Checkpoints are selected on a per-seed val
    split carved from train_idx; test_idx and the operating-point set
    (exclude_idx) are never seen, so neither the checkpoint nor tau* is chosen
    on trained-on data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_idx = np.asarray(train_idx)
    if exclude_idx is not None:                  # fold-level operating-point set
        train_idx = np.setdiff1d(train_idx, np.asarray(exclude_idx))

    # Per-seed validation split from TRAIN rows only (for checkpoint selection).
    _vrng = np.random.RandomState(seed + 9973)
    _perm = _vrng.permutation(len(train_idx))
    _n_val = max(1, int(0.15 * len(train_idx)))
    val_idx = train_idx[_perm[:_n_val]]
    train_sub = train_idx[_perm[_n_val:]]

    net = SafetyOutcomePredictor(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=args.dropout,
        safety_arch=args.safety_arch,
    ).to(args.device)

    # Cost-head target normalization (log1p-standardized), fit on TRAIN rows only.
    if probe_costs is not None:
        _tc = probe_costs[train_sub]
        _pos = torch.log1p(_tc[_tc > 0])
        net.cost_log_mean.fill_(float(_pos.mean()))
        net.cost_log_std.fill_(float(_pos.std().clamp(min=1e-3)))

    # Separate parameter groups: safety and the Lagrange multiplier
    lagrange_params = []
    safety_params = []
    for name, param in net.named_parameters():
        if "log_lambda" in name:
            lagrange_params.append(param)
        else:
            safety_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": safety_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": lagrange_params, "lr": args.lambda_lr, "weight_decay": 0.0},
    ])

    best_asr = float("inf")   # generic checkpoint score (lower = better)
    best_state = None
    eval_every = 5
    patience_evals = 4
    no_improve = 0

    for epoch in range(args.epochs):
        train_epoch(
            net, optimizer, adv_embs, safety_tensor,
            ben_embs, ben_quality, ben_train_idx,
            train_sub, batch_size=args.batch_size, device=args.device,
            focal_gamma=args.focal_gamma, rank_weight=args.rank_weight,
            cost_train_weight=args.cost_train,
            probe_costs=probe_costs, asr_target=args.asr_target,
            cost_focal_alpha=args.cost_focal_alpha,
            cost_pred_weight=args.cost_pred_weight,
            safety_loss=args.safety_loss, unsafe_weight=args.unsafe_weight)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if args.ckpt_mode == "constraint":
                # Lexicographic: feasible (val ASR < target) at min cost, else min achievable ASR.
                pts = [evaluate(net, adv_embs, safety_tensor, val_idx, COST_MATRIX,
                                device=args.device, probe_costs=probe_costs,
                                safety_threshold=tau, use_cost_head=args.use_cost_head)
                       for tau in (0.95, 0.97, 0.98, 0.99)]
                feas = [r["avg_realized_cost"] for r in pts if r["asr"] < args.ckpt_asr_target]
                score = min(feas) if feas else 1.0 + min(r["asr"] for r in pts)
            else:
                res = evaluate(net, adv_embs, safety_tensor, val_idx, COST_MATRIX,
                               device=args.device, probe_costs=probe_costs,
                               safety_threshold=0.95, use_cost_head=args.use_cost_head)
                score = res["asr"]

            if score < best_asr:
                best_asr = score
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience_evals:
                break

    if best_state:
        net.load_state_dict(best_state)
        net.to(args.device)

    # Post-hoc calibration of P(safe) on the val split (val only).
    if args.calibrate:
        if args.calib_mode == "isotonic":
            _fit_isotonic(net, adv_embs, safety_tensor, val_idx, args.device)
        else:
            _fit_temperature(net, adv_embs, safety_tensor, val_idx, args.device,
                             mode=args.calib_mode)

    return net


def run_kfold(args, adv_embs, safety_tensor, ben_embs, ben_quality,
              ben_train_idx, ben_test_idx, idx_to_method, probe_costs=None):
    """Run k-fold attack-method holdout evaluation with multi-seed ensemble."""
    all_methods = sorted(set(idx_to_method.values()))
    folds = make_folds(all_methods, args.kfold, seed=args.fold_seed)

    n_seeds = args.n_seeds
    seeds = [args.seed + s for s in range(n_seeds)]

    print(f"\n{args.kfold} folds, {n_seeds} seeds per fold (fold_seed={args.fold_seed}):")
    for fi, fold in enumerate(folds):
        n = sum(1 for i, m in idx_to_method.items() if m in set(fold))
        print(f"  Fold {fi}: {fold} ({n} probes)")

    all_results = []
    all_fold_nets = []  # store for final ensemble eval

    for fi, fold_methods in enumerate(folds):
        holdout = set(fold_methods)
        train_idx = np.array([i for i, m in idx_to_method.items() if m not in holdout])
        test_idx = np.array([i for i, m in idx_to_method.items() if m in holdout])

        # Operating-point set, excluded from training so τ* is never selected on test.
        _orng = np.random.RandomState(20259 + fi)
        _op = _orng.permutation(len(train_idx))
        optval_idx = train_idx[_op[:max(1, int(0.15 * len(train_idx)))]]

        print(f"\n{'='*60}")
        print(f"Fold {fi}: holdout={fold_methods}, train={len(train_idx)}, "
              f"test={len(test_idx)}, optval={len(optval_idx)}")

        # Train ensemble
        fold_nets = []
        for seed in seeds:
            net = train_one_fold(
                args, adv_embs, safety_tensor, ben_embs, ben_quality,
                ben_train_idx, train_idx, probe_costs, seed,
                exclude_idx=optval_idx)
            # One NaN net makes every `P(safe) > τ` False, degenerating selection to argmax.
            if not all(torch.isfinite(p).all() for p in net.state_dict().values()
                       if torch.is_floating_point(p)):
                print(f"  !! seed={seed} DIVERGED (non-finite weights) — EXCLUDED from the ensemble")
                continue
            fold_nets.append(net)

            # Quick single-seed eval
            res = evaluate(net, adv_embs, safety_tensor, test_idx,
                          COST_MATRIX, device=args.device, probe_costs=probe_costs,
                          safety_threshold=0.95, use_cost_head=args.use_cost_head)
            print(f"  seed={seed}: ASR={res['asr']*100:.2f}% cost=${res['avg_realized_cost']:.4f}")

        if not fold_nets:
            raise SystemExit(f"fold {fi}: every seed diverged -- lower --lr or --lambda-lr, "
                             f"or raise --epochs (the Lagrangian needs room to settle)")
        all_fold_nets.append((test_idx, fold_nets))

        # Sweeps on TEST (frontier) and OPTVAL (τ* selection), same τ grid.
        print(f"  Ensemble ({n_seeds} seeds):")
        sweep = ensemble_threshold_sweep(fold_nets, adv_embs, safety_tensor, test_idx,
                                         COST_MATRIX, device=args.device, probe_costs=probe_costs,
                                         use_cost_head=args.use_cost_head)
        opt_sweep = ensemble_threshold_sweep(fold_nets, adv_embs, safety_tensor, optval_idx,
                                             COST_MATRIX, device=args.device, probe_costs=probe_costs,
                                             use_cost_head=args.use_cost_head)
        for t, asr, cost in sweep:
            marker = " ◀" if asr < 0.008 and cost < 0.10 else ""
            print(f"    thresh={t:.3f}: ASR={asr*100:.2f}% cost=${cost:.4f}{marker}")

        # τ* from OPTVAL via select_tau(), then TEST read once at that τ*.
        tau_star = select_tau(opt_sweep, args.op_asr_target, len(optval_idx))
        ens_res = ensemble_evaluate(fold_nets, adv_embs, safety_tensor, test_idx,
                                    COST_MATRIX, device=args.device, probe_costs=probe_costs,
                                    safety_threshold=tau_star,
                                    use_cost_head=args.use_cost_head)

        # Risk gate eval (from first net in ensemble)
        risk_res = evaluate_risk_gate(fold_nets[0], adv_embs, ben_embs, test_idx, ben_test_idx,
                                      device=args.device)

        result = {
            "fold": fi, "holdout": fold_methods, "tau_star": tau_star,
            "opt_total": int(len(optval_idx)),
            "test_asr": ens_res["asr"], "test_jb": ens_res["jb"], "test_total": ens_res["total"],
            "test_realized_cost": ens_res["avg_realized_cost"],
            "risk_auc": risk_res["auc"], "risk_catch": risk_res["adv_catch"],
            "risk_fpr": risk_res["ben_fpr"],
            "threshold_sweep": sweep, "opt_sweep": opt_sweep,
        }
        all_results.append(result)

        if args.save_nets:
            nets_dir = Path(args.save_dir) / "nets"
            nets_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "states": [{k: v.cpu() for k, v in n.state_dict().items()} for n in fold_nets],
                "arch": {"hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
                         "dropout": args.dropout, "safety_arch": args.safety_arch,
                         },
                "test_idx": np.asarray(test_idx), "optval_idx": np.asarray(optval_idx),
            }, nets_dir / f"fold{fi}.pt")

        print(f"  FINAL (honest τ*={tau_star:.3f} via optval): test_ASR={ens_res['asr']*100:.2f}% "
              f"realized=${ens_res['avg_realized_cost']:.4f}/1000q "
              f"risk_AUC={risk_res['auc']:.3f}")

    # Summary
    total_jb = sum(r["test_jb"] for r in all_results)
    total_n = sum(r["test_total"] for r in all_results)
    fold_asrs = [r["test_asr"] for r in all_results]
    avg_realized = np.mean([r["test_realized_cost"] for r in all_results])

    print(f"\n{'='*60}")
    print(f"SafeRouter {args.kfold}-FOLD SUMMARY ({n_seeds} seeds)")
    print(f"  Micro ASR: {total_jb}/{total_n} = {total_jb/total_n*100:.2f}%")
    print(f"  Macro ASR: {np.mean(fold_asrs)*100:.2f}% ± {np.std(fold_asrs)*100:.2f}%")
    print(f"  Avg realized cost: ${avg_realized:.4f}/1000q")
    print(f"  Risk AUC:  {np.mean([r['risk_auc'] for r in all_results]):.3f}")

    # Global threshold sweep (aggregate over all folds)
    print("\n  Global threshold sweep:")
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87,
                  0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    for t in thresholds:
        total_jb_t = 0
        total_n_t = 0
        total_cost_t = 0.0
        for test_idx_f, fold_nets_f in all_fold_nets:
            r = ensemble_evaluate(fold_nets_f, adv_embs, safety_tensor, test_idx_f,
                                  COST_MATRIX, device=args.device, probe_costs=probe_costs,
                                  safety_threshold=t, use_cost_head=args.use_cost_head)
            total_jb_t += r["jb"]
            total_n_t += r["total"]
            total_cost_t += r["avg_realized_cost"] * r["total"]
        asr_t = total_jb_t / total_n_t
        cost_t = total_cost_t / total_n_t
        marker = " ◀" if asr_t < 0.02 and cost_t < 0.10 else ""
        print(f"    thresh={t:.3f}: ASR={asr_t*100:.2f}% ({total_jb_t}/{total_n_t}) cost=${cost_t:.4f}{marker}")

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    out = Path(args.save_dir) / "sop_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved → {out}")

    return all_results


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Train Safety-Outcome Predictor (SOP)")
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
    # ── Per-query cost head + calibration + diagram search ──
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
    args = parser.parse_args()

    if args.use_cost_head and args.cost_pred_weight <= 0:
        parser.error("--use-cost-head requires --cost-pred-weight > 0 "
                     "(the cost head would be untrained otherwise)")

    print("=" * 60)
    print("Safety-Outcome Predictor")
    print("=" * 60)

    print("\nLoading safety data...")
    adv_embs, keys, safety_tensor, probe_costs = load_safety_data()

    print("Loading benign data...")
    ben_embs, ben_qids, ben_quality, ben_costs, ben_train_idx, ben_test_idx = load_benign_data()

    # Preload once — avoids per-batch CPU->GPU transfers, and the batch indices
    # below are built on args.device, so what they index must live there too.
    if args.device != "cpu":
        adv_embs = adv_embs.to(args.device)
        safety_tensor = safety_tensor.to(args.device)
        ben_embs = ben_embs.to(args.device)
        ben_quality = ben_quality.to(args.device)
        if probe_costs is not None:
            probe_costs = probe_costs.to(args.device)

    if args.hard_benign_emb:
        global HARD_BEN_EMBS
        hb = torch.load(args.hard_benign_emb, map_location="cpu", weights_only=False)
        HARD_BEN_EMBS = hb["embeddings"].float().to(args.device)
        print(f"  Hard-benign risk negatives: {len(HARD_BEN_EMBS)} sensitive-but-legitimate "
              f"queries added to the gate's benign class (risk-BCE only) ← {args.hard_benign_emb}")

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
    else:
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

        # Quality eval


if __name__ == "__main__":
    main()
