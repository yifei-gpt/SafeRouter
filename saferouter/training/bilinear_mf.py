"""RouteLLM (BilinearMF): a win-predictor over correct/incorrect model pairs."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cost import N_MODELS
from routers import BilinearMF
from training import CORRECT_THR, DEVICE, SEED
from training.loop import fit


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

    def logits(batch):
        emb, w, l = (x.to(DEVICE) for x in batch)
        return model(w, l, model.project_text(emb))

    def step(batch):
        logit = logits(batch)
        return bce(logit, torch.ones_like(logit))

    def validate(loader):
        # The winner must outrank the loser, so a correct pair is logit > 0.
        correct = total = 0
        for batch in loader:
            logit = logits(batch)
            correct += (logit > 0).sum().item()
            total += len(logit)
        acc = correct / max(total, 1)
        return acc, f"val_pair_acc={acc*100:.1f}%"

    best_acc = fit(model, optim, tr_ldr, val_ldr, step, validate, save_path, epochs, patience)
    print(f"  Best val_pair_acc={best_acc*100:.1f}% → {save_path}")
