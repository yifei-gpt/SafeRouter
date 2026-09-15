#!/usr/bin/env python3
"""Train the three quality-cost baseline routers on 28K R2Bench queries: BilinearMF
(RouteLLM), CARROT-KNN, and MIRT (IRT-Router), all on the shared group-aware 85/15
split from benign_split.

    python train_all_routers.py [--only mirt carrot]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# Model pool and pricing come from cost/ so they cannot drift from
# cost/price.json or from train_saferouter's view of the same pool.
from cost import (JUDGED_FILES, MODELS, N_MODELS,
                  RESPONSE_FILES)
from cost import per_query_costs as cost_per_query

# Configuration

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
JUDGED_DIR = DATA / "benign" / "judged"
EMB_PATH = DATA / "embeddings" / "r2bench_28k_embeddings.pt"
LLM_EMB = DATA / "embeddings" / "model_profile_embeddings.pt"
CKPT_DIR = DATA / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CORRECT_THR = 0.7

# Response file mapping (for per-query token costs)
RESPONSE_DIR = DATA / "benign" / "responses"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# Data loading

def load_data():
    """Load 28K embeddings + judge scores for all models."""
    print("Loading R2Bench 28K embeddings + judge labels...")
    emb_data = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    all_embs = emb_data["embeddings"]  # (28020, 1024)
    q_id_list = emb_data["question_ids"]
    q_id_to_idx = {str(qid): i for i, qid in enumerate(q_id_list)}

    # Load judge scores per model (+ query text for the duplicate-aware split)
    correctness = {}
    query_text = {}
    for m in MODELS:
        fname = JUDGED_FILES[m]
        fpath = JUDGED_DIR / fname
        correctness[m] = {}
        with open(fpath) as fh:
            for line in fh:
                r = json.loads(line)
                qid = str(r["id"])
                if r.get("judge_score") is not None:
                    correctness[m][qid] = float(r["judge_score"])
                    if qid not in query_text:
                        query_text[qid] = r.get("query") or ""

    # Intersect: queries that have scores for ALL models AND embeddings
    qids = sorted(
        set(q_id_to_idx.keys()) &
        set.intersection(*(set(correctness[m].keys()) for m in MODELS))
    )
    N = len(qids)
    print(f"  Queries with full model coverage: {N}")

    emb_matrix = np.array([all_embs[q_id_to_idx[q]].numpy() for q in qids])
    score_matrix = np.array([[correctness[m][q] for m in MODELS] for q in qids])

    # Load per-query costs from response data (original CARROT uses actual token costs)
    cost_matrix = np.zeros((N, N_MODELS), dtype=np.float32)
    for mi, m in enumerate(MODELS):
        fpath = RESPONSE_DIR / RESPONSE_FILES[m]
        per_query_cost = cost_per_query(fpath, m, per_1000q=False)
        for qi, qid in enumerate(qids):
            if qid not in per_query_cost:
                raise KeyError(
                    f"Query {qid} has a judge score but no per-query cost in "
                    f"{fpath.name} (missing/empty api_usage). Re-probe this query "
                    f"or drop it from the judged set — no average-cost fallback by design.")
            cost_matrix[qi, mi] = per_query_cost[qid]

    # Group-aware split so duplicate query texts never straddle train/test (H5)
    from benign_split import group_split_indices
    train_idx, test_idx = group_split_indices(qids, query_text, test_frac=0.15)
    print(f"  Total: {N}   train: {len(train_idx)}   test: {len(test_idx)}")
    return {
        "all_embs": all_embs,
        "qids": qids,
        "q_id_to_idx": q_id_to_idx,
        "emb": emb_matrix,
        "score": score_matrix,
        "cost": cost_matrix,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# Router 1 BilinearMF, matching RouteLLM: bias-free P + text_proj + classifier, dim=128.

class BilinearMF(nn.Module):
    def __init__(self, n_models=N_MODELS, dim=128, text_dim=1024):
        super().__init__()
        self.n_models = n_models
        self.P = nn.Embedding(n_models, dim)
        self.text_proj = nn.Linear(text_dim, dim, bias=False)
        self.classifier = nn.Linear(dim, 1, bias=False)

    def project_text(self, q_emb):
        """Project raw embedding into latent routing space."""
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        return self.text_proj(q_emb)

    def forward(self, model_win, model_loss, q_proj):
        """Pairwise scoring on pre-projected embeddings."""
        v_win = F.normalize(self.P(model_win), p=2, dim=-1)
        v_loss = F.normalize(self.P(model_loss), p=2, dim=-1)
        h = v_win - v_loss
        if q_proj.dim() == 1:
            q_proj = q_proj.unsqueeze(0)
        interaction = h * q_proj
        logit = self.classifier(interaction).squeeze(-1)
        return logit

    def score_all(self, q_proj):
        """Return scores for all models given pre-projected embedding(s)."""
        P_all = F.normalize(self.P.weight, p=2, dim=-1)  # (n_models, dim)
        if q_proj.dim() == 1:
            interaction = P_all * q_proj.unsqueeze(0)  # (n_models, dim)
        else:
            interaction = P_all.unsqueeze(0) * q_proj.unsqueeze(1)  # (batch, n_models, dim)
        logits = self.classifier(interaction).squeeze(-1)
        return logits


class PairDataset(Dataset):
    def __init__(self, pairs, embs):
        self.pairs = pairs
        self.embs = embs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        eidx, w, l = self.pairs[i]
        return self.embs[eidx], torch.tensor(w), torch.tensor(l)


def build_quality_pairs(query_indices, score_mat, qids, q_id_to_idx):
    """RouteLLM MF pair construction. Official RouteLLM trains its win-predictor
    on winner/loser preference battles; we have only per-(model, query) judge
    scores, so we instantiate the same objective from correctness: binarize at
    CORRECT_THR and pair every correct model against every incorrect one.

    Do NOT use a naive "argmax-best vs the rest" over CONTINUOUS scores: near-ties
    on easy queries crown a small model the winner and the router collapses toward
    cheap/weak models (~0.53 quality vs ~0.85)."""
    out = []
    for qi in query_indices:
        emb_idx = q_id_to_idx[qids[qi]]
        scores = score_mat[qi]
        winners = [m for m in range(N_MODELS) if scores[m] >= CORRECT_THR]
        losers = [m for m in range(N_MODELS) if scores[m] < CORRECT_THR]
        for w in winners:
            for l in losers:
                out.append((emb_idx, w, l))
    return out


def train_bilinear_mf(data, *, save_path, epochs=30, lr=1e-3, batch_size=64, patience=7):
    """Train BilinearMF with early stopping."""
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)  # deterministic init + shuffle order
    print(f"\n{'='*60}\nTraining BilinearMF (RouteLLM-faithful)\n{'='*60}")
    pair_builder = lambda idx: build_quality_pairs(idx, data["score"], data["qids"], data["q_id_to_idx"])
    # Checkpoints selected on a train-internal val split; test is never touched.
    vrng = np.random.RandomState(SEED + 9973)
    perm = vrng.permutation(len(data["train_idx"]))
    n_val = max(1, int(0.15 * len(data["train_idx"])))
    val_idx = data["train_idx"][perm[:n_val]]
    train_sub = data["train_idx"][perm[n_val:]]
    tr_pairs = pair_builder(train_sub)
    val_pairs = pair_builder(val_idx)
    print(f"  Train pairs: {len(tr_pairs):,}   Val pairs: {len(val_pairs):,}")

    tr_ds = PairDataset(tr_pairs, data["all_embs"])
    val_ds = PairDataset(val_pairs, data["all_embs"])
    tr_ldr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BilinearMF(n_models=N_MODELS).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_acc, pat = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tloss, nb = 0, 0
        for emb, w, l in tr_ldr:
            emb, w, l = emb.to(DEVICE), w.to(DEVICE), l.to(DEVICE)
            q_proj = model.project_text(emb)
            logit = model(w, l, q_proj)
            loss = bce(logit, torch.ones_like(logit))
            optim.zero_grad()
            loss.backward()
            optim.step()
            tloss += loss.item()
            nb += 1

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for emb, w, l in val_ldr:
                emb, w, l = emb.to(DEVICE), w.to(DEVICE), l.to(DEVICE)
                q_proj = model.project_text(emb)
                logit = model(w, l, q_proj)
                correct += (logit > 0).sum().item()
                total += len(logit)
        acc = correct / max(total, 1)
        print(f"  ep={ep:3d}  tr_loss={tloss/nb:.4f}  val_pair_acc={acc*100:.1f}%")
        if acc > best_acc:
            best_acc, pat = acc, 0
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
            if pat >= patience:
                print(f"  early stop ep={ep}")
                break
    print(f"  Best val_pair_acc={best_acc*100:.1f}% → {save_path}")


# Router 2 CARROT-KNN, matching the original: KNeighborsRegressor, cosine metric, tuned k.

def tune_n_neighbors(X_train, Y_train,
                     n_neighbors_range=(2, 4, 8, 16, 32, 64, 128, 256, 512),
                     metric='cosine', cv=5):
    """Tune k via cross-validation (original CARROT logic)."""
    best_score = -float('inf')
    best_k = n_neighbors_range[0]
    for k in n_neighbors_range:
        if k > len(X_train):
            continue
        knn = KNeighborsRegressor(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_train, Y_train, cv=cv, scoring='r2')
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
    return int(best_k), best_score


def train_carrot_knn(data, *, save_path):
    """Train CARROT with quality KNN + cost KNN (matches original two-branch design).

    Original routing: model_idx = ((1-λ) * quality_pred - λ * cost_pred).argmax()
    We save both KNNs; at evaluation time λ controls the quality-cost tradeoff.
    """
    print(f"\n{'='*60}\nTraining CARROT (KNN, cosine)\n{'='*60}")
    X_train = data["emb"][data["train_idx"]]
    Y_train = data["score"][data["train_idx"]]
    X_test = data["emb"][data["test_idx"]]
    Y_test = data["score"][data["test_idx"]]
    C_train = data["cost"][data["train_idx"]]
    C_test = data["cost"][data["test_idx"]]

    # --- Quality branch ---
    print("  [Quality branch]")
    best_k, cv_score = tune_n_neighbors(X_train, Y_train)
    print(f"    Best k={best_k}  CV R²={cv_score:.4f}")

    knn_quality = KNeighborsRegressor(n_neighbors=best_k, metric='cosine')
    knn_quality.fit(X_train, Y_train)

    Y_pred = knn_quality.predict(X_test)
    B_true = (Y_test >= CORRECT_THR).astype(int)
    aucs = []
    for i in range(N_MODELS):
        try:
            aucs.append(roc_auc_score(B_true[:, i], Y_pred[:, i]))
        except ValueError:   # only one class present for this model → AUC undefined
            aucs.append(0.5)
    print(f"    Test mean_auc={np.mean(aucs):.4f}")

    # --- Cost branch (Z-score normalized, matching original) ---
    print("  [Cost branch]")
    cost_mu = C_train.mean(axis=0, keepdims=True)
    cost_std = C_train.std(axis=0, keepdims=True) + 1e-8
    Z_train = (C_train - cost_mu) / cost_std

    best_k_c, cv_score_c = tune_n_neighbors(X_train, Z_train)
    print(f"    Best k={best_k_c}  CV R²={cv_score_c:.4f}")

    knn_cost = KNeighborsRegressor(n_neighbors=best_k_c, metric='cosine')
    knn_cost.fit(X_train, Z_train)

    Z_pred = knn_cost.predict(X_test)
    C_pred = cost_mu + cost_std * Z_pred
    cost_rmse = float(np.sqrt(mean_squared_error(C_test, C_pred)))
    print(f"    Test cost_rmse={cost_rmse:.6f}")

    # Save both branches + normalization params
    carrot_bundle = {
        "knn_quality": knn_quality,
        "knn_cost": knn_cost,
        "cost_mu": cost_mu,
        "cost_std": cost_std,
    }
    joblib.dump(carrot_bundle, save_path)
    print(f"  Saved → {save_path}")


def carrot_route(carrot_bundle, X, lam=0.0):
    """CARROT quality-cost routing:
        model_idx = argmax_m [ (1-λ)·quality_pred_m − λ·zcost_pred_m ]

    Blended in Z-SCORE space, where the 0-1 quality and unit-variance cost are
    comparable and λ sweeps a smooth frontier. Do NOT de-normalize cost to
    dollars here: benign costs are ~1e-3, ~1000x smaller than quality, which
    leaves λ inert until it collapses all-or-nothing at λ→1. cost_mu/cost_std
    stay in the bundle only for reporting realized $."""
    quality_pred = carrot_bundle["knn_quality"].predict(X)
    if lam == 0.0:
        return quality_pred.argmax(axis=1)
    z_cost_pred = carrot_bundle["knn_cost"].predict(X)   # already standardized
    scores = (1 - lam) * quality_pred - lam * z_cost_pred
    return scores.argmax(axis=1)


def load_mirt(ckpt_dir=None, device=DEVICE):
    """The trained IRT-Router plus the model-profile matrix it scores against."""
    net = MIRTNet().to(device).eval()
    net.load_state_dict(torch.load(Path(ckpt_dir or CKPT_DIR) / "mirt.pt",
                                   map_location=device, weights_only=False))
    prof = torch.load(LLM_EMB, map_location="cpu", weights_only=False)
    return net, torch.stack([prof[m] for m in MODELS]).float().to(device)


@torch.no_grad()
def irt_quality(net, llm, x):
    """P(correct) for every model -> (batch, n_models)."""
    return torch.stack([net(llm[i].expand(len(x), -1), x) for i in range(N_MODELS)], 1)


# Router 3 MIRT, matching IRT-Router: sigmoid(sum(a*theta) - b), a = softplus(a_proj(q)),

class MIRTNet(nn.Module):
    def __init__(self, llm_dim=1024, query_dim=1024, latent=25):
        super().__init__()
        self.theta_proj = nn.Linear(llm_dim, latent, bias=False)
        self.a_proj = nn.Linear(query_dim, latent, bias=False)
        self.b_proj = nn.Linear(query_dim, 1, bias=False)

    def forward(self, llm_emb, query_emb):
        theta = self.theta_proj(llm_emb)
        a = F.softplus(self.a_proj(query_emb))
        b = self.b_proj(query_emb).squeeze(-1)
        return torch.sigmoid((a * theta).sum(-1) - b)


def train_mirt(data, *, save_path, epochs=30, lr=1e-3, patience=8):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)  # deterministic init + shuffle order
    print(f"\n{'='*60}\nTraining MIRT\n{'='*60}")
    llm_dict = torch.load(LLM_EMB, map_location="cpu", weights_only=False)
    llm_M = torch.stack([llm_dict[m] for m in MODELS]).to(DEVICE)  # (N_MODELS, 1024)
    Q_emb = data["all_embs"].to(DEVICE)

    # Checkpoints selected on a train-internal val split; test is held out entirely.
    vrng = np.random.RandomState(SEED + 9973)
    perm = vrng.permutation(len(data["train_idx"]))
    n_val = max(1, int(0.15 * len(data["train_idx"])))
    val_qids = set(data["train_idx"][perm[:n_val]].tolist())
    train_sub_qids = set(data["train_idx"][perm[n_val:]].tolist())
    rows_train, rows_val = [], []
    for qi in range(len(data["qids"])):
        if qi in val_qids:
            tgt = rows_val
        elif qi in train_sub_qids:
            tgt = rows_train
        else:
            continue  # test split — never seen during training
        for mi in range(N_MODELS):
            tgt.append((mi, data["q_id_to_idx"][data["qids"][qi]], float(data["score"][qi, mi])))

    def make_loader(rows, shuffle):
        m_t = torch.tensor([r[0] for r in rows], dtype=torch.long)
        q_t = torch.tensor([r[1] for r in rows], dtype=torch.long)
        s_t = torch.tensor([r[2] for r in rows], dtype=torch.float)
        return DataLoader(TensorDataset(m_t, q_t, s_t),
                          batch_size=512, shuffle=shuffle, num_workers=2, pin_memory=True)

    tr_ldr = make_loader(rows_train, True)
    val_ldr = make_loader(rows_val, False)
    print(f"  Train rows: {len(rows_train):,}   Val rows: {len(rows_val):,}")

    model = MIRTNet().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)  # no weight decay (matches original)
    loss_fn = nn.BCELoss()

    best_auc, pat = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tloss, nb = 0, 0
        for mi, qi, s in tr_ldr:
            mi, qi, s = mi.to(DEVICE), qi.to(DEVICE), s.to(DEVICE)
            pred = model(llm_M[mi], Q_emb[qi])
            loss = loss_fn(pred, s)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tloss += loss.item()
            nb += 1

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for mi, qi, s in val_ldr:
                mi, qi, s = mi.to(DEVICE), qi.to(DEVICE), s.to(DEVICE)
                preds.extend(model(llm_M[mi], Q_emb[qi]).cpu().tolist())
                truths.extend(s.cpu().tolist())
        preds, truths = np.array(preds), np.array(truths)
        try:
            # AUC of continuous preds vs the 0.5-binarized label (reference MIRT binarizes both).
            auc = float(roc_auc_score(truths >= 0.5, preds))
        except ValueError:   # only one class present → AUC undefined
            auc = 0.5
        rmse = float(np.sqrt(mean_squared_error(truths, preds)))
        print(f"  ep={ep:3d}  tr_loss={tloss/nb:.4f}  val_rmse={rmse:.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, pat = auc, 0
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
            if pat >= patience:
                print(f"  early stop ep={ep}")
                break
    print(f"  Best auc={best_auc:.4f} → {save_path}")


# CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["bimf", "carrot", "mirt"],
                    help="Train only these routers (default: all 3)")
    args = ap.parse_args()
    selected = args.only or ["bimf", "carrot", "mirt"]

    data = load_data()

    if "bimf" in selected:
        (CKPT_DIR / "router").mkdir(exist_ok=True, parents=True)
        train_bilinear_mf(data, save_path=CKPT_DIR / "router" / "quality_router.pt")

    if "carrot" in selected:
        train_carrot_knn(data, save_path=CKPT_DIR / "carrot_knn.joblib")

    if "mirt" in selected:
        train_mirt(data, save_path=CKPT_DIR / "mirt.pt")

    print("\nAll done.")


if __name__ == "__main__":
    main()
