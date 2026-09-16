"""The SafeRouter network, and how a trained router resolves a query."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cost import COMPOSITE_TO_IDX, N_COMPOSITES, N_MODELS

class SafetyOutcomePredictor(nn.Module):
    """Four heads on a shared backbone over the frozen embedding:
      safety   Linear -> all 10 x 16 cells at once, FLAT (not factored)
      cost     Linear -> per-cell expected $/1000q
      risk     MLP    -> P(adversarial | q), the benign/adversarial gate
      quality  MLP    -> per-model benign quality; its gradients reach the
                         backbone, so the trunk keeps what benign routing needs
    """

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

        # Flat 160-way base; bilinear ADDS a zero-init correction on top.
        self.safety_head = nn.Linear(hidden_dim, n_models * n_composites)
        if safety_arch == "bilinear":
            d = 64
            self.s_proj = nn.Linear(hidden_dim, d)
            self.model_emb = nn.Parameter(torch.randn(n_models, d) * 0.02)
            self.defense_emb = nn.Parameter(torch.randn(n_composites, d) * 0.02)
            self.inter_scale = nn.Parameter(torch.zeros(1))      # zero-init → starts at flat

        # ---- Cost head: per-cell expected cost, log1p-normalized ----
        self.cost_head = nn.Linear(hidden_dim, n_models * n_composites)
        # Per-cell P(safe) temperature, fit on val.
        self.register_buffer("temperature", torch.ones(n_models, n_composites))
        self.register_buffer("cost_log_mean", torch.zeros(1))   # cost-target normalization
        self.register_buffer("cost_log_std", torch.ones(1))

        # ---- Quality head ----
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, n_models))

        # ---- Risk head ----
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # ---- Lagrange multiplier (log-space for positivity) ----
        self.log_lambda = nn.Parameter(torch.tensor(0.0))


    def forward(self, x):
        """-> safety_logits (batch, n_models, n_composites), risk_logit (batch, 1)."""
        z = self.backbone(x)
        risk_logit = self.risk_head(z)
        safety_logits = self._safety_logits(z)
        return safety_logits, risk_logit

    def _safety_logits(self, z):
        """Flat logits, plus the bilinear head's zero-init (model x defense) correction."""
        logits = self.safety_head(z).view(z.shape[0], self.n_models, self.n_composites)
        if self.safety_arch == "bilinear":
            zp = self.s_proj(z)                                  # (B, d)
            inter = torch.einsum('bd,md,cd->bmc', zp, self.model_emb, self.defense_emb)
            logits = logits + self.inter_scale * inter
        return logits

    def calibrate_psafe(self, logits):
        """P(safe), temperature-scaled per cell by the temperature fit on val."""
        return torch.sigmoid(logits / self.temperature)

    def predict_safety(self, x):
        """Return P(safe | q, m, d) as probabilities (calibrated)."""
        logits, risk = self.forward(x)
        return self.calibrate_psafe(logits), torch.sigmoid(risk)

    def predict_quality(self, z):
        """Per-model benign quality from backbone features. -> (batch, n_models)."""
        return self.quality_head(z)

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
        """Among cells predicted safe the cheapest, else the safest cell."""
        p_safe, _ = self.predict_safety(x)
        B = x.shape[0]
        if cost_matrix.dim() == 2:
            costs = cost_matrix.unsqueeze(0).expand(B, -1, -1).to(x.device)
        else:
            costs = cost_matrix.to(x.device)

        best_flat, _ = cheap_first(p_safe, costs, safety_threshold)
        return best_flat // self.n_composites, best_flat % self.n_composites

def cheap_first(p_safe, cost, min_p_safe):
    """Among cells with P(safe) > min_p_safe the cheapest, else the safest. ->
    (flat cell index, above-threshold mask). The only definition: every caller
    resolves cells through it."""
    flat_p, flat_c = p_safe.flatten(1), cost.flatten(1)
    ok = flat_p > min_p_safe
    pick = torch.where(ok.any(1), flat_c.masked_fill(~ok, float("inf")).argmin(1),
                       flat_p.argmax(1))
    return pick, ok

def _fit_temperature(net, adv_embs, safety_tensor, val_idx, device):
    """Per-cell temperature fitted on val only, L2-regularized toward a shared
    scalar so that sparsely-covered cells cannot overfit."""
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

def load_fold_nets(run_dirs, fold, device="cuda", verbose=False):
    """One fold's nets across runs -> (nets, test_idx, optval_idx, n_dropped);
    test_idx is None when no run has that fold. Diverged (non-finite) nets are
    skipped -- one NaN net makes every `P(safe) > tau` False. Runs must agree
    on the fold indices or pooling would mix nets with each other's test
    probes."""
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
            net.load_state_dict(state, strict=False)   # older ckpts carry extra buffers
            nets.append(net.to(device).eval())
    return nets, test_idx, optval_idx, dropped

def load_router(ckpt_dir, fold=0, device="cuda", runs="k18_retrain"):
    """One fold's nets, pooled across every run under `runs`. `load_fold_nets` is
    the lower-level entry point for runs trained separately."""
    run_dirs = sorted(p for p in (Path(ckpt_dir) / runs).iterdir() if p.is_dir())
    return load_fold_nets(run_dirs, fold, device=device)[0]

@torch.no_grad()
def route(nets, x, min_p_safe):
    """-> (model_idx, composite_idx, p_adversarial, n_safe_cells). Mean calibrated
    P(safe) and mean predicted cost over the nets, then cheap-first."""
    p_safe = sum(n.calibrate_psafe(n(x)[0]) for n in nets) / len(nets)
    cost = sum(n.predict_cost(x) for n in nets) / len(nets)
    risk = sum(torch.sigmoid(n(x)[1]).squeeze(-1) for n in nets) / len(nets)
    pick, ok = cheap_first(p_safe, cost, min_p_safe)
    model, defense = pick // N_COMPOSITES, pick % N_COMPOSITES
    # A benign query never pays for a defense: quality head at the S0 cell.
    quality = sum(n.predict_quality(n.backbone(x)) for n in nets) / len(nets)
    benign = risk < 0.5
    model = torch.where(benign, quality.argmax(1), model)
    defense = torch.where(benign, COMPOSITE_TO_IDX[("s0", "s0")], defense)
    return model, defense, risk, ok.sum(1)
