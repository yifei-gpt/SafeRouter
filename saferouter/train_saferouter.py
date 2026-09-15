#!/usr/bin/env python3
"""SafeRouter: security-aware routing with cost-conditioned defense selection.
Four heads on a shared frozen Qwen3-Embedding-0.6B backbone -- safety
P(safe|q,m,d,cost), cost, risk P(adversarial|q), plus a loaded quality head:

    z = backbone(query);  z_cell = concat(z, model_emb, defense_emb, cost_feat)
    safety_logit = cell_mlp(z_cell)        # all 10 x 16 = 160 cells at once

Losses: inverse-focal + pairwise ranking + a Lagrangian constraint (min E[cost]
s.t. ASR < eps); risk BCE. The defaults below are a SMOKE TEST --
reproduce with scripts/train.sh, which passes the pre-registered flags.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import beta

# Model pool, composite taxonomy and pricing all live in cost/ -- duplicating them
# here is how the reported dollars would silently drift from cost/price.json.
from cost import (COMPOSITE_TO_IDX, COMPOSITES, FULL_TO_SHORT, GUARD_ABS_COST,
                  GUARD_BLOCK_COST, JUDGED_FILES, MODEL_INPUT_PRICES,
                  MODELS, N_COMPOSITES, N_MODELS, RESPONSE_FILES)
from cost import per_query_costs as cost_per_query

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

# ─── tau* selection (single definition; mega_ensemble imports it) ───
TAU_DELTA = 0.05          # pre-registered confidence level; do not tune
OP_ASR_TARGET = 0.003     # a-priori operating target; mega_ensemble imports it


def cp_upper(k, n, delta=TAU_DELTA):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def select_tau(opt_points, op, n_opt, delta=TAU_DELTA):
    """tau* from [(tau, asr, cost)] measured on OPTVAL: the cheapest tau whose
    ASR upper BOUND clears `op`, else the tau with the tightest bound.

    The bound, not the empirical rate, is what makes this sound: at op=0.003 over
    ~1.1k optval probes the empirical rate carries ~3 expected positives, so
    argmin-cost over the "feasible" set just deploys whichever cheap tau got lucky.
    `n_opt` is required -- a default would turn the stored rates into wrong counts.
    """
    rows = [(t, cp_upper(round(a * n_opt), n_opt, delta), c) for t, a, c in opt_points]
    feas = [r for r in rows if r[1] <= op]
    if feas:
        return min(feas, key=lambda x: x[2])[0]
    return min(rows, key=lambda x: (x[1], x[2]))[0]


# ─── Data paths ───
EMB_ADV = DATA / "embeddings" / "adversarial_full_embeddings.pt"
EMB_BEN = DATA / "embeddings" / "r2bench_28k_embeddings.pt"
# --hard-benign-emb: risk-head negatives teaching that sensitive != adversarial.
HARD_BEN_EMBS = None
COST_TENSOR_PATH = DATA / "embeddings" / "adversarial_cost_tensor.pt"
# Labels use hb_/sb_/pb_ ids matching the embedded keys.
LABELS = DATA / "adversarial" / "probe" / "labels.jsonl"
PROBE_INPUT = DATA / "adversarial" / "probe_input_merged.jsonl"
JUDGED_DIR = DATA / "benign" / "judged"
RESPONSE_DIR = DATA / "benign" / "responses"
# Static cost prior ($/1000q): sets cheap-first ORDER only; reported costs use the tensor.
MODEL_COSTS = {
    "qwen3-0.6b": 0.0054, "qwen3-1.7b": 0.0161, "qwen3-4b": 0.0254,
    "qwen3-8b": 0.2938, "qwen3-14b": 0.1733, "qwen3-30b-a3b": 0.3289,
    "qwen3-32b": 0.2025, "qwen3-coder-next-fp8": 0.8074,
    "nemotron-3-super-120b": 0.5036, "nemotron-3-nano-30b": 0.0655,
}
# Helper costs ($/1000q), token counts measured from probe data.
# Llama-3.3-70B paraphrase: measured avg 289 input + 175 output tokens
PARA_ABS_COST = (289 * 0.10 + 175 * 0.32) / 1e6 * 1e3    # $0.0849/1000q
# S4 gpt-4o-mini harm judge: 326 in + 1.5 out. S5 reuses this twice (harm + refusal judge).
S4_ABS_COST = (326 * 0.15 + 1.5 * 0.60) / 1e6 * 1e3      # $0.0490/1000q
# Llama-3.3-70B backtranslation: avg 350 in + 24 out (N=22645)
S5_BACKTR_COST = (350 * 0.10 + 24 * 0.32) / 1e6 * 1e3    # $0.0427/1000q
# S1 SCR: extra input tokens from 4 retrieved primers, measured as target_B.in - target_A.in.
S1_EXTRA_TOKENS = 447  # average, used in COST_MATRIX for routing


def compute_defense_cost(model, pre, post):
    """pre_overhead + target_generation + post_overhead, $/1000q. For S3 this is
    the pass-through cost; the block case is handled separately."""
    gen_cost = MODEL_COSTS[model]

    # Pre-gen cost
    if pre == "s0":
        pre_ov = 0.0
    elif pre == "s1":
        pre_ov = S1_EXTRA_TOKENS * MODEL_INPUT_PRICES[model] / 1e6 * 1e3
    elif pre == "s2":
        pre_ov = PARA_ABS_COST
    elif pre == "s3":
        pre_ov = GUARD_ABS_COST

    # Post-gen cost
    if post == "s0":
        post_ov = 0.0
    elif post == "s4":
        post_ov = S4_ABS_COST  # 1 GPT-4o-mini harmfulness judge call
    elif post == "s5":
        # S5: two gpt-4o-mini judges (~S4_ABS_COST each) + backtranslation + target re-query.
        post_ov = S4_ABS_COST + S5_BACKTR_COST + gen_cost + S4_ABS_COST
    elif post == "s6":
        # S6 PARDEN: one extra target generation (the repeat); BLEU is free.
        post_ov = gen_cost

    return pre_ov + gen_cost + post_ov


# Precompute cost matrix (N_MODELS, N_COMPOSITES) — absolute $/1000q
COST_MATRIX = torch.zeros(N_MODELS, N_COMPOSITES)
for mi, m in enumerate(MODELS):
    for ci, (pre, post) in enumerate(COMPOSITES):
        COST_MATRIX[mi, ci] = compute_defense_cost(m, pre, post)



# ─── Data loading ───

def cheap_first(p_safe, cost, min_p_safe):
    """Among cells with P(safe) > min_p_safe take the cheapest, else the safest.
    -> (flat cell index per query, the above-threshold mask). Single definition:
    select_cheap_first, ensemble_evaluate and route all resolve cells through it."""
    flat_p, flat_c = p_safe.flatten(1), cost.flatten(1)
    ok = flat_p > min_p_safe
    pick = torch.where(ok.any(1), flat_c.masked_fill(~ok, float("inf")).argmin(1),
                       flat_p.argmax(1))
    return pick, ok


def load_router(ckpt_dir, fold=0, device="cuda", runs="k18_retrain"):
    """Everything routing needs: one fold's net ensemble across every run under
    `runs`, plus the quality head. `load_fold_nets` is the lower-level entry point
    for pooling runs that were trained separately."""
    run_dirs = sorted(p for p in (Path(ckpt_dir) / runs).iterdir() if p.is_dir())
    nets, *_ = load_fold_nets(run_dirs, fold, device=device)
    return nets, load_quality_head(ckpt_dir, device)


def load_quality_head(ckpt_dir=None, device="cuda"):
    """The quality head, loaded rather than trained: SafeRouter routes benign queries
    with the standalone IRT-Router from train_all_routers.py. It scores the raw 1024-d
    embedding, so it is detached from the safety backbone by construction."""
    from train_all_routers import load_mirt
    net, profiles = load_mirt(ckpt_dir or (DATA / "checkpoints"), device=device)
    net.requires_grad_(False)
    return net, profiles


@torch.no_grad()
def route(nets, x, min_p_safe, quality=None):
    """Ensemble cheap-first routing -> (model_idx, composite_idx, p_adversarial, n_safe_cells).
    Mean calibrated P(safe) and mean predicted cost over the nets; among cells above
    min_p_safe take the cheapest, else fall back to the safest cell."""
    p_safe = sum(n.calibrate_psafe(n(x)[0]) for n in nets) / len(nets)
    cost = sum(n.predict_cost(x) for n in nets) / len(nets)
    risk = sum(torch.sigmoid(n(x)[1]).squeeze(-1) for n in nets) / len(nets)
    pick, ok = cheap_first(p_safe, cost, min_p_safe)
    model, defense = pick // N_COMPOSITES, pick % N_COMPOSITES
    if quality is not None:
        # Risk gate: a benign query never pays for a defense, so it goes to the
        # quality head at the S0 cell instead of cheap-first over the 160.
        qnet, profiles = quality
        scores = torch.stack([qnet(profiles[i].expand(len(x), -1), x)
                              for i in range(N_MODELS)], 1)
        benign = risk < 0.5
        model = torch.where(benign, scores.argmax(1), model)
        defense = torch.where(benign, COMPOSITE_TO_IDX[("s0", "s0")], defense)
    return model, defense, risk, ok.sum(1)


def load_fold_nets(run_dirs, fold, device="cuda", verbose=False):
    """Every saved net for one fold across runs -> (nets, test_idx, optval_idx, n_dropped).
    test_idx is None when no run has that fold. Diverged (non-finite) nets are skipped --
    one NaN net makes every `P(safe) > tau` False. Runs must agree on the fold's indices:
    pooling nets trained on each other's test probes would leak."""
    files = [f for f in (Path(r) / "nets" / f"fold{fold}.pt" for r in run_dirs) if f.exists()]
    nets, test_idx, optval_idx, dropped = [], None, None, 0
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if test_idx is not None and (list(blob["test_idx"]) != list(test_idx)
                                     or list(blob["optval_idx"]) != list(optval_idx)):
            raise SystemExit(f"{f} has different test/optval indices than the earlier runs "
                             f"-- all runs must share --fold-seed")
        test_idx, optval_idx = blob["test_idx"], blob["optval_idx"]
        for si, state in enumerate(blob["states"]):
            if not all(torch.isfinite(v).all() for v in state.values()
                       if torch.is_floating_point(v)):
                if verbose:
                    print(f"  !! DROPPED diverged net (non-finite): {f} seed{si}",
                          file=sys.stderr)
                dropped += 1
                continue
            arch = {k: v for k, v in blob["arch"].items() if k != "quality_mode"}
            net = SafetyOutcomePredictor(**arch)
            net.load_state_dict(state, strict=False)   # old ckpts lack calib_iso/iso_y
            nets.append(net.to(device).eval())
    return nets, test_idx, optval_idx, dropped


def probe_query(question_id, attack_method, path=PROBE_INPUT):
    """The attack text of one shipped probe, keyed as in the safety tensor."""
    for line in open(path):
        row = json.loads(line)
        if row["question_id"] == question_id and row["attack_method"] == attack_method:
            return row["query"]
    raise KeyError((question_id, attack_method))


def load_safety_data():
    """-> embs (N,1024), keys [(query_id, attack_method)], safety (N,M,C) 1=safe,
    probe_costs (N,M,C) $/1000q or None."""
    d = torch.load(EMB_ADV, map_location="cpu", weights_only=False)
    embs = d["embeddings"].float()
    keys = d["keys"]
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Build dense safety tensor
    N = len(keys)
    safety = torch.full((N, N_MODELS, N_COMPOSITES), -1, dtype=torch.float32)  # -1 = missing

    model_to_idx = {m: i for i, m in enumerate(MODELS)}
    with open(LABELS) as f:
        for line in f:
            r = json.loads(line)
            short = FULL_TO_SHORT.get(r["target_model"])
            if short is None:
                continue
            mi = model_to_idx.get(short)
            if mi is None:
                continue
            ci = COMPOSITE_TO_IDX.get((r["pre"], r["post"]))
            if ci is None:
                continue
            probe_key = (r["query_id"], r["attack_method"])
            pi = key_to_idx.get(probe_key)
            if pi is None:
                continue
            if r.get("asr_judge_failed"):
                continue  # ASR judge failed -> leave cell missing (-1), never count as safe
            safety[pi, mi, ci] = 1.0 - float(r["is_jailbroken"])  # 1=safe, 0=jailbroken

    # Check coverage
    valid = (safety >= 0).sum().item()
    total = N * N_MODELS * N_COMPOSITES
    print(f"  Safety tensor: {tuple(safety.shape)}, coverage: {valid}/{total} ({valid/total*100:.1f}%)")

    # Load per-probe cost tensor if available
    probe_costs = None
    if COST_TENSOR_PATH.exists():
        ct = torch.load(COST_TENSOR_PATH, map_location="cpu", weights_only=False)
        if "keys" in ct:
            assert ct["keys"] == keys, \
                "cost tensor keys misaligned with embeddings — rerun build_cost_tensor"
        probe_costs = ct["cost_tensor"]  # (N_probes, N_MODELS, N_COMPOSITES) $/1000q
        print(f"  Per-probe costs: {tuple(probe_costs.shape)}, "
              f"range=[${probe_costs[probe_costs>0].min():.4f}, ${probe_costs.max():.4f}]/1000q")
    else:
        print("  Per-probe costs: not found, using average COST_MATRIX")

    return embs, keys, safety, probe_costs


def load_benign_data():
    """-> embs (N,1024), qids, ben_costs (N,M) $/1000q, train_idx, test_idx.
    Judge scores are read only to intersect model coverage and to get the query
    text the duplicate-aware split needs."""
    d = torch.load(EMB_BEN, map_location="cpu", weights_only=False)
    embs = d["embeddings"].float()
    qids = d["question_ids"]
    qid_to_idx = {str(q): i for i, q in enumerate(qids)}

    # Load quality scores (+ query text for the duplicate-aware split)
    scores = {}
    query_text = {}
    for m in MODELS:
        scores[m] = {}
        with open(JUDGED_DIR / JUDGED_FILES[m]) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("judge_score") is not None:
                    scores[m][str(r["id"])] = float(r["judge_score"])
                    if str(r["id"]) not in query_text:
                        query_text[str(r["id"])] = r.get("query") or ""

    # Per-query S0 costs ($/1000q) from the recorded token counts.
    per_query_costs = {m: cost_per_query(RESPONSE_DIR / RESPONSE_FILES[m], m)
                       for m in MODELS}

    # Intersect
    valid_qids = sorted(
        set(qid_to_idx.keys()) &
        set.intersection(*(set(scores[m].keys()) for m in MODELS))
    )
    N = len(valid_qids)

    ben_costs = torch.zeros(N, N_MODELS)
    emb_indices = []
    for i, q in enumerate(valid_qids):
        emb_indices.append(qid_to_idx[q])
        for mi, m in enumerate(MODELS):
            ben_costs[i, mi] = per_query_costs[m].get(q, MODEL_COSTS[m])

    embs_subset = embs[emb_indices]

    # Split — group-aware so duplicate query texts never straddle train/test (H5)
    from benign_split import group_split_indices
    train_idx, test_idx = group_split_indices(valid_qids, query_text, test_frac=0.15)

    print(f"  Benign: {N} queries, train={len(train_idx)}, test={len(test_idx)}")
    return embs_subset, valid_qids, ben_costs, train_idx, test_idx


def build_attack_method_index(keys):
    """Map each probe index to its attack method."""
    return {i: k[1] for i, k in enumerate(keys)}


def make_folds(methods, k, seed=42):
    """Split methods into k folds."""
    rng = np.random.RandomState(seed)
    methods_shuffled = list(methods)
    rng.shuffle(methods_shuffled)
    folds = [[] for _ in range(k)]
    for i, m in enumerate(methods_shuffled):
        folds[i % k].append(m)
    return folds


# ─── Model ───

class SafetyOutcomePredictor(nn.Module):
    """Shared backbone (query -> z) feeding:
      safety   z -> Linear -> n_models x n_composites independent P(safe|q,m,d)
               logits. FLAT (160-way), not a factored model/defense decomposition.
      risk     z -> MLP -> P(adversarial | q)
      quality  NOT trained here -- load_quality_head() returns the standalone
               IRT-Router, which scores the raw 1024-d embedding, not z.

    Loss: focal safety + pairwise ranking (+ optional cost-aware ranking,
    cost-weighted focal, per-query cost regression), plus an always-on Lagrangian
    "min cost s.t. ASR < asr_target" with learned lambda."""

    def __init__(self, input_dim=1024, hidden_dim=256,
                 n_models=N_MODELS, n_composites=N_COMPOSITES,
                 n_layers=2, dropout=0.15, safety_arch="flat"):
        super().__init__()
        self.n_models = n_models
        self.n_composites = n_composites
        self.safety_arch = safety_arch

        # Shared backbone
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                          nn.GELU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*layers)

        # Flat 160-way base; arch variants ADD a zero-init correction, starting at the flat head.
        self.safety_head = nn.Linear(hidden_dim, n_models * n_composites)
        if safety_arch == "bilinear":
            d = 64
            self.s_proj = nn.Linear(hidden_dim, d)
            self.model_emb = nn.Parameter(torch.randn(n_models, d) * 0.02)
            self.defense_emb = nn.Parameter(torch.randn(n_composites, d) * 0.02)
            self.inter_scale = nn.Parameter(torch.zeros(1))      # zero-init → starts at flat
        elif safety_arch == "xattn":
            self.cell_tokens = nn.Parameter(torch.randn(n_models * n_composites, hidden_dim) * 0.02)
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.xattn_out = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.xattn_out.weight); nn.init.zeros_(self.xattn_out.bias)  # zero-init → flat
        elif safety_arch == "factored":
            # IRT/CP-tensor head: logit = a_q·θ_{m,d} − b_q, θ = θ_m + Δ_d + v_m⊙u_d.
            L = 32
            self.f_a = nn.Linear(hidden_dim, L)                              # query discrimination a_q
            self.f_b = nn.Linear(hidden_dim, 1)                             # query difficulty b_q
            self.f_model = nn.Parameter(torch.randn(n_models, L) * 0.05)    # θ_m model ability
            self.f_defense = nn.Parameter(torch.zeros(n_composites, L))     # Δ_d defense shift
            self.f_mv = nn.Parameter(torch.randn(n_models, L) * 0.02)       # v_m
            self.f_dv = nn.Parameter(torch.randn(n_composites, L) * 0.02)   # u_d

        # ── Cost head: per-(model,defense) expected-cost regression (log1p-normalized) ──
        self.cost_head = nn.Linear(hidden_dim, n_models * n_composites)
        # P(safe) temperature per (model, defense) cell, fit on the val split.
        self.register_buffer("temperature", torch.ones(n_models, n_composites))
        # --calib-mode isotonic: calib_iso>0.5 swaps temperature for a 256-pt monotone map.
        self.register_buffer("calib_iso", torch.zeros(1))
        self.register_buffer("iso_y", torch.linspace(0.0, 1.0, 256))
        self.register_buffer("cost_log_mean", torch.zeros(1))   # cost-target normalization
        self.register_buffer("cost_log_std", torch.ones(1))

        # ── Risk head ──
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ── Lagrange multiplier (log-space for positivity) ──
        self.log_lambda = nn.Parameter(torch.tensor(0.0))


    def forward(self, x):
        """
        Returns:
            safety_logits: (batch, n_models, n_composites)
            risk_logit: (batch, 1)
        """
        z = self.backbone(x)
        risk_logit = self.risk_head(z)
        safety_logits = self._safety_logits(z)
        return safety_logits, risk_logit

    def _safety_logits(self, z):
        """Flat logits + (zero-init) arch correction; or the factored IRT head directly."""
        B = z.shape[0]
        if self.safety_arch == "factored":
            a = self.f_a(z)                                                  # (B, L)
            b = self.f_b(z)                                                  # (B, 1)
            theta = (self.f_model.unsqueeze(1) + self.f_defense.unsqueeze(0)
                     + self.f_mv.unsqueeze(1) * self.f_dv.unsqueeze(0))       # (M, D, L)
            return torch.einsum('bl,mdl->bmd', a, theta) - b.unsqueeze(-1)    # (B, M, D)
        logits = self.safety_head(z).view(B, self.n_models, self.n_composites)
        if self.safety_arch == "bilinear":
            zp = self.s_proj(z)                                  # (B, d)
            inter = torch.einsum('bd,md,cd->bmc', zp, self.model_emb, self.defense_emb)
            logits = logits + self.inter_scale * inter
        elif self.safety_arch == "xattn":
            q = self.cell_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 120, h)
            kv = z.unsqueeze(1)                                  # (B, 1, h)
            ctx, _ = self.attn(q, kv, kv)                        # (B, 120, h)
            corr = self.xattn_out(ctx).squeeze(-1).view(B, self.n_models, self.n_composites)
            logits = logits + corr
        return logits

    def calibrate_psafe(self, logits):
        """Calibrated P(safe): isotonic monotone map if fitted (--calib-mode isotonic),
        else temperature scaling. Isotonic uses a fixed 256-pt grid with linear interp."""
        if float(self.calib_iso) > 0.5:
            p = torch.sigmoid(logits).clamp(0.0, 1.0)
            G = self.iso_y.numel()
            pos = p * (G - 1)
            lo = pos.floor().long().clamp(0, G - 2)
            t = (pos - lo.float())
            return (self.iso_y[lo] * (1 - t) + self.iso_y[lo + 1] * t).clamp(0.0, 1.0)
        return torch.sigmoid(logits / self.temperature)

    def predict_safety(self, x):
        """Return P(safe | q, m, d) as probabilities (calibrated)."""
        logits, risk = self.forward(x)
        return self.calibrate_psafe(logits), torch.sigmoid(risk)

    def norm_cost(self, c):
        """Normalize a cost tensor ($/1000q) to the head's log1p-standardized space."""
        return (torch.log1p(c) - self.cost_log_mean) / self.cost_log_std

    @torch.no_grad()
    def predict_cost(self, x):
        """Predict per-query expected cost ($/1000q) for all (model, defense) cells."""
        z = self.backbone(x)
        norm = self.cost_head(z).view(-1, self.n_models, self.n_composites)
        return torch.expm1(norm * self.cost_log_std + self.cost_log_mean).clamp(min=0)

    def select_cheap_first(self, x, cost_matrix, safety_threshold=0.95):
        """Cheap-first selection: among cells predicted safe, pick cheapest."""
        p_safe, _ = self.predict_safety(x)
        B = x.shape[0]
        if cost_matrix.dim() == 2:
            costs = cost_matrix.unsqueeze(0).expand(B, -1, -1).to(x.device)
        else:
            costs = cost_matrix.to(x.device)

        best_flat, _ = cheap_first(p_safe, costs, safety_threshold)
        return best_flat // self.n_composites, best_flat % self.n_composites

# ─── Training ───

def train_epoch(net, optimizer, adv_embs, safety_tensor,
                ben_embs, ben_train_idx,
                adv_train_idx, batch_size=512, device="cuda",
                focal_gamma=0.5, rank_weight=0.3, cost_train_weight=0.0,
                probe_costs=None, asr_target=0.01,
                cost_focal_alpha=0.0, cost_pred_weight=0.0, safety_loss="focal",
                unsafe_weight=1.0):
    """One epoch. Loss = focal_safety + ranking + risk_bce
         + λ_detach·max(0, batch_asr−ε)   (primal: θ reduces ASR)
         − λ·max(0, batch_asr−ε)_detach   (dual: λ rises while ASR > ε)
    All data must already be on `device`."""
    net.train()

    # Cost-weighted terms need per-probe costs; fail fast instead of a TypeError deep in the loss.
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
            # up-weights confident-correct cells -> better tail calibration (arXiv:2505.23463)
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
                    # cost_train_weight applies inside loss_rank, so the effective weight is the product.
                    loss_rank = loss_rank + cost_train_weight * cost_margins.mean()

        # ── Risk loss ──
        loss_risk_adv = F.binary_cross_entropy_with_logits(
            risk_logit, torch.ones_like(risk_logit))

        # ── Benign: risk loss + quality loss ──
        loss_risk_ben = _zero
        if len(batch_ben) > 0:
            ben_z = net.backbone(ben_embs[batch_ben])
            loss_risk_ben = F.binary_cross_entropy_with_logits(
                net.risk_head(ben_z), torch.zeros(len(batch_ben), 1, device=device))

            # Hard-benign negatives (sensitive but legitimate): risk BCE only.
            if HARD_BEN_EMBS is not None:
                hb_idx = torch.randint(len(HARD_BEN_EMBS), (len(batch_ben),), device=device)
                hb_z = net.backbone(HARD_BEN_EMBS[hb_idx])
                loss_risk_ben = loss_risk_ben + F.binary_cross_entropy_with_logits(
                    net.risk_head(hb_z), torch.zeros(len(hb_idx), 1, device=device))


        # Lagrangian soft-ASR surrogate = mean P(safe) over UNSAFE cells, not its complement.
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
                + 0.5 * (loss_risk_adv + loss_risk_ben)
                + lagrangian_primal + lagrangian_dual)

        # ── Cost-prediction head: regress per-query pipeline cost (detached backbone) ──
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

@torch.no_grad()
def evaluate(net, adv_embs, safety_tensor, adv_indices,
             cost_matrix, device="cuda", probe_costs=None,
             safety_threshold=0.95, use_cost_head=False):
    """Cheap-first selection on adversarial probes: among cells with
    P(safe) > threshold pick the cheapest. With use_cost_head, "cheapest" comes
    from the per-query predicted cost rather than the static matrix; realized
    cost is always measured from probe_costs. -> ASR, cost, action stats."""
    net.eval()

    x = adv_embs[adv_indices].to(device)
    targets = safety_tensor[adv_indices]  # (N, M, D)

    # Get model predictions using cheap-first selection
    sel_cost = net.predict_cost(x) if use_cost_head else cost_matrix
    model_idx, defense_idx = net.select_cheap_first(
        x, sel_cost, safety_threshold=safety_threshold)

    model_idx = model_idx.cpu()
    defense_idx = defense_idx.cpu()

    # Look up actual outcomes
    jb = 0
    total = 0
    n_missing = 0
    total_pipeline_cost = 0.0
    total_realized_cost = 0.0
    action_counts = defaultdict(int)

    # Realized cost: an S3 composite that BLOCKED this query for this model costs guard-only.
    s3_s0_ci = COMPOSITE_TO_IDX[("s3", "s0")]
    s0_s0_ci = COMPOSITE_TO_IDX[("s0", "s0")]

    for i in range(len(adv_indices)):
        mi = int(model_idx[i])
        di = int(defense_idx[i])
        idx = adv_indices[i]

        outcome = targets[i, mi, di].item()
        if outcome < 0:
            outcome = 0  # missing = assume jailbroken (conservative)
            n_missing += 1

        total += 1
        if outcome < 0.5:  # jailbroken
            jb += 1

        # Use per-probe cost if available, otherwise average cost matrix
        if probe_costs is not None:
            pipeline_cost = probe_costs[idx, mi, di].item()
        else:
            pipeline_cost = cost_matrix[mi, di].item()
        total_pipeline_cost += pipeline_cost

        # Realized cost: if S3 blocked this query, only guard cost incurred.
        pre, post = COMPOSITES[di]
        if pre == "s3":
            s3_val = targets[i, mi, s3_s0_ci].item()
            s0_val = targets[i, mi, s0_s0_ci].item()
            if s3_val >= 0.5 and s0_val >= 0 and s0_val < 0.5:
                total_realized_cost += GUARD_BLOCK_COST
            else:
                total_realized_cost += pipeline_cost
        else:
            total_realized_cost += pipeline_cost

        action_counts[(MODELS[mi], COMPOSITES[di])] += 1

    asr = jb / max(total, 1)
    avg_pipeline_cost = total_pipeline_cost / max(total, 1)
    avg_realized_cost = total_realized_cost / max(total, 1)

    # Top actions
    top_actions = sorted(action_counts.items(), key=lambda x: -x[1])[:5]

    if n_missing > 0:
        print(f"  WARNING: {n_missing}/{total} outcomes missing (treated as jailbroken)")

    return {
        "asr": asr,
        "jb": jb,
        "total": total,
        "avg_pipeline_cost": avg_pipeline_cost,
        "avg_realized_cost": avg_realized_cost,
        "n_missing": n_missing,
        "top_actions": top_actions,
    }


@torch.no_grad()
def evaluate_risk_gate(net, adv_embs, ben_embs, adv_indices, ben_test_idx, device="cuda"):
    """Evaluate the risk head's discrimination ability."""
    net.eval()

    adv_x = adv_embs[adv_indices].to(device)
    ben_x = ben_embs[ben_test_idx].to(device)

    _, risk_adv = net(adv_x)
    _, risk_ben = net(ben_x)

    risk_adv = torch.sigmoid(risk_adv).cpu().numpy().flatten()
    risk_ben = torch.sigmoid(risk_ben).cpu().numpy().flatten()

    # AUC
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones(len(risk_adv)), np.zeros(len(risk_ben))])
    y_score = np.concatenate([risk_adv, risk_ben])
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:      # single-class y_true
        auc = 0.5

    # At threshold 0.5
    adv_catch = (risk_adv > 0.5).mean()
    ben_fpr = (risk_ben > 0.5).mean()

    return {"auc": auc, "adv_catch": adv_catch, "ben_fpr": ben_fpr}


@torch.no_grad()
@torch.no_grad()
def threshold_sweep(net, adv_embs, safety_tensor, adv_indices,
                    cost_matrix, device="cuda", probe_costs=None, use_cost_head=False):
    """Sweep safety thresholds and report ASR vs cost Pareto points."""
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.945, 0.95,
                  0.96, 0.97, 0.98, 0.99, 0.995]
    results = []
    for t in thresholds:
        res = evaluate(net, adv_embs, safety_tensor, adv_indices,
                      cost_matrix, device=device, probe_costs=probe_costs,
                      safety_threshold=t, use_cost_head=use_cost_head)
        results.append((t, res["asr"], res["avg_realized_cost"]))
    return results


@torch.no_grad()
def ensemble_evaluate(nets, adv_embs, safety_tensor, adv_indices,
                      cost_matrix, device="cuda", probe_costs=None,
                      safety_threshold=0.95, use_cost_head=False):
    """Ensemble: average P(safe) (after per-net calibration), then cheap-first —
    cheapest cell above threshold, else argmax(P(safe))."""
    x = adv_embs[adv_indices].to(device)
    B = x.shape[0]

    # Average calibrated P(safe) (and predicted cost, if requested) across ensemble
    p_safe_sum = torch.zeros(B, N_MODELS, N_COMPOSITES, device=device)
    cost_sum = torch.zeros(B, N_MODELS, N_COMPOSITES, device=device) if use_cost_head else None
    for net in nets:
        net.eval()
        logits, _ = net(x)
        p_safe_sum += net.calibrate_psafe(logits)
        if use_cost_head:
            cost_sum += net.predict_cost(x)
    p_safe_avg = p_safe_sum / len(nets)

    # Cheap-first selection (argmax(P_safe) fallback when no cell is safe enough)
    if use_cost_head:
        costs = cost_sum / len(nets)
    elif cost_matrix.dim() == 2:
        costs = cost_matrix.unsqueeze(0).expand(B, -1, -1).to(device)
    else:
        costs = cost_matrix.to(device)

    best_flat, _ = cheap_first(p_safe_avg, costs, safety_threshold)
    model_idx = (best_flat // N_COMPOSITES).cpu()
    defense_idx = (best_flat % N_COMPOSITES).cpu()

    # Look up actual outcomes
    targets = safety_tensor[adv_indices]
    jb = 0
    total = 0
    total_realized_cost = 0.0
    s3_s0_ci = COMPOSITE_TO_IDX[("s3", "s0")]
    s0_s0_ci = COMPOSITE_TO_IDX[("s0", "s0")]

    for i in range(len(adv_indices)):
        mi = int(model_idx[i])
        di = int(defense_idx[i])
        idx = adv_indices[i]
        outcome = targets[i, mi, di].item()
        if outcome < 0:
            outcome = 0
        total += 1
        if outcome < 0.5:
            jb += 1

        if probe_costs is not None:
            pipeline_cost = probe_costs[idx, mi, di].item()
        else:
            pipeline_cost = cost_matrix[mi, di].item()

        pre, post = COMPOSITES[di]
        if pre == "s3":
            s3_val = targets[i, mi, s3_s0_ci].item()
            s0_val = targets[i, mi, s0_s0_ci].item()
            if s3_val >= 0.5 and s0_val >= 0 and s0_val < 0.5:
                total_realized_cost += GUARD_BLOCK_COST
            else:
                total_realized_cost += pipeline_cost
        else:
            total_realized_cost += pipeline_cost

    asr = jb / max(total, 1)
    avg_realized_cost = total_realized_cost / max(total, 1)

    return {"asr": asr, "jb": jb, "total": total, "avg_realized_cost": avg_realized_cost}


@torch.no_grad()
def ensemble_threshold_sweep(nets, adv_embs, safety_tensor, adv_indices,
                             cost_matrix, device="cuda", probe_costs=None, use_cost_head=False):
    """Sweep the safety threshold for the average-ensemble cheap-first selector."""
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87,
                  0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.975,
                  0.98, 0.982, 0.984, 0.985, 0.986, 0.988, 0.99, 0.991, 0.992, 0.993,
                  0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
    results = []
    for t in thresholds:
        res = ensemble_evaluate(nets, adv_embs, safety_tensor, adv_indices,
                               cost_matrix, device=device, probe_costs=probe_costs,
                               safety_threshold=t, use_cost_head=use_cost_head)
        results.append((t, res["asr"], res["avg_realized_cost"]))
    return results


# ─── K-fold ───

def _fit_isotonic(net, adv_embs, safety_tensor, val_idx, device):
    """Fit a global isotonic calibration of P(safe) on the val split (val only, never
    test). Maps raw sigmoid(logits) -> empirical safe-rate monotonically; stored on a
    fixed 256-pt grid so the ASR/threshold constraint binds the TRUE rate (UCCI-style)."""
    from sklearn.isotonic import IsotonicRegression
    net.eval()
    with torch.no_grad():
        logits, _ = net(adv_embs[val_idx].to(device))
    tgt = safety_tensor[val_idx].to(device)
    mask = (tgt >= 0)
    if int(mask.sum()) < 256:
        return
    p = torch.sigmoid(logits)[mask].float().cpu().numpy()
    y = tgt.clamp(0, 1)[mask].float().cpu().numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    grid = np.linspace(0.0, 1.0, net.iso_y.numel()).astype(np.float32)
    ys = iso.predict(grid).astype(np.float32)
    net.iso_y.copy_(torch.tensor(ys, device=net.iso_y.device))
    net.calib_iso.fill_(1.0)


def _fit_temperature(net, adv_embs, safety_tensor, val_idx, device, mode="scalar"):
    """Fit a temperature on the val split (val only, never test).

    mode='scalar': one shared T. mode='vector': per-(model,defense) T, mildly
    L2-regularized toward the shared scalar so sparse cells don't overfit.
    """
    net.eval()
    with torch.no_grad():
        logits, _ = net(adv_embs[val_idx].to(device))          # (V,10,12) raw
    tgt = safety_tensor[val_idx].to(device)
    mask = (tgt >= 0)
    if mask.sum() < 16:
        return
    tg = tgt.clamp(0, 1)

    # Always fit the shared scalar first (robust anchor).
    s = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([s], lr=0.1, max_iter=50)
    lg_m, tg_m = logits[mask].detach(), tg[mask]

    def cl_s():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg_m / s.exp(), tg_m)
        loss.backward(); return loss
    opt.step(cl_s)
    s_val = float(s.exp().clamp(0.3, 5.0).item())
    net.temperature.fill_(s_val)
    if mode != "vector":
        return

    # Per-cell refinement around the scalar anchor (L2 toward 0 in log space).
    logT = torch.full((net.n_models, net.n_composites), float(s.item()),
                      device=device, requires_grad=True)
    optv = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)
    m = mask.float()

    def cl_v():
        optv.zero_grad()
        scaled = logits.detach() / logT.exp().unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(scaled, tg, reduction="none")
        loss = (bce * m).sum() / m.sum() + 1e-3 * ((logT - s.detach()) ** 2).mean()
        loss.backward(); return loss
    optv.step(cl_v)
    net.temperature.copy_(logT.detach().exp().clamp(0.3, 5.0))


def train_one_fold(args, adv_embs, safety_tensor, ben_embs,
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
            ben_embs, ben_train_idx,
            train_sub, batch_size=args.batch_size, device=args.device,
            focal_gamma=args.focal_gamma, rank_weight=args.rank_weight,
            cost_train_weight=args.cost_train,
            probe_costs=probe_costs, asr_target=args.asr_target,
            cost_focal_alpha=args.cost_focal_alpha,
            cost_pred_weight=args.cost_pred_weight,
            safety_loss=args.safety_loss, unsafe_weight=args.unsafe_weight)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if args.ckpt_mode == "constraint":
                # Lexicographic: prefer feasible (val ASR < target) at min cost, else min achievable ASR.
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


def run_kfold(args, adv_embs, safety_tensor, ben_embs,
              ben_train_idx, ben_test_idx, idx_to_method, probe_costs=None,
              ben_costs=None):
    """Run k-fold attack-method holdout evaluation with multi-seed ensemble."""
    all_methods = sorted(set(idx_to_method.values()))
    folds = make_folds(all_methods, args.kfold, seed=args.fold_seed)

    n_seeds = args.n_seeds
    seeds = [args.seed + s for s in range(n_seeds)]

    print(f"\n{args.kfold} folds, {n_seeds} seeds per fold (fold_seed={args.fold_seed}):")
    for fi, fold in enumerate(folds):
        n = sum(1 for i, m in idx_to_method.items() if m in set(fold))
        print(f"  Fold {fi}: {fold} ({n} probes)")

    # Preload ALL data to GPU once — avoids per-batch CPU→GPU transfers
    if args.device != "cpu":
        print(f"  Preloading data to {args.device}...")
        adv_embs = adv_embs.to(args.device)
        safety_tensor = safety_tensor.to(args.device)
        ben_embs = ben_embs.to(args.device)
        if probe_costs is not None:
            probe_costs = probe_costs.to(args.device)

    all_results = []
    all_fold_nets = []  # store for final ensemble eval

    for fi, fold_methods in enumerate(folds):
        holdout = set(fold_methods)
        train_idx = np.array([i for i, m in idx_to_method.items() if m not in holdout])
        test_idx = np.array([i for i, m in idx_to_method.items() if m in holdout])

        # Operating-point set: excluded from every seed's training so τ* is never selected on test.
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
                args, adv_embs, safety_tensor, ben_embs,
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

        all_fold_nets.append((test_idx, fold_nets))

        # Threshold sweeps on TEST (frontier) and OPTVAL (τ* selection), on the same τ grid.
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

        # Honest operating point: τ* from OPTVAL via select_tau(), then evaluate TEST once at that τ*.
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
    parser.add_argument("--safety-threshold", type=float, default=0.95,
                        help="Default safety threshold for cheap-first selection")
    # ── Per-query cost head + calibration + diagram search ──
    parser.add_argument("--safety-arch", type=str, default="flat",
                        choices=["flat", "bilinear", "xattn", "factored"],
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
    ben_embs, ben_qids, ben_costs, ben_train_idx, ben_test_idx = load_benign_data()

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
    print(f"  n_seeds={args.n_seeds}, safety_threshold={args.safety_threshold}")

    if args.kfold > 0:
        run_kfold(args, adv_embs, safety_tensor, ben_embs,
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
            args, adv_embs, safety_tensor, ben_embs,
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
