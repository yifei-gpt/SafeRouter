"""IRT-Router (MIRT): fit query discrimination/difficulty against model ability."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_squared_error

from cost import MODELS, N_MODELS
from routers import MIRTNet
from utils.data_io import EMB_LLM
from training import DEVICE, SEED
from training.loop import fit


def train_mirt(data, *, save_path, epochs=30, lr=1e-3, patience=8):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)  # deterministic init + shuffle order
    print(f"\n{'='*60}\nTraining MIRT\n{'='*60}")
    llm_dict = torch.load(EMB_LLM, map_location="cpu", weights_only=False)
    llm_M = torch.stack([llm_dict[m] for m in MODELS]).to(DEVICE)  # (N_MODELS, 1024)
    Q_emb = data["all_embs"].to(DEVICE)

    # Checkpoints selected on a train-internal val split; test stays held out.
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

    def step(batch):
        mi, qi, s = (x.to(DEVICE) for x in batch)
        return loss_fn(model(llm_M[mi], Q_emb[qi]), s)

    def validate(loader):
        preds, truths = [], []
        for batch in loader:
            mi, qi, s = (x.to(DEVICE) for x in batch)
            preds.extend(model(llm_M[mi], Q_emb[qi]).cpu().tolist())
            truths.extend(s.cpu().tolist())
        preds, truths = np.array(preds), np.array(truths)
        try:
            # AUC of continuous preds vs the 0.5-binarized label.
            auc = float(roc_auc_score(truths >= 0.5, preds))
        except ValueError:   # only one class present → AUC undefined
            auc = 0.5
        rmse = float(np.sqrt(mean_squared_error(truths, preds)))
        return auc, f"val_rmse={rmse:.4f}  val_auc={auc:.4f}"

    best_auc = fit(model, optim, tr_ldr, val_ldr, step, validate, save_path, epochs, patience)
    print(f"  Best auc={best_auc:.4f} → {save_path}")
