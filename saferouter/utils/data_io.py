"""The shipped probes, judgements and embeddings, plus the fold structure.

Both training drivers read the same three benign sources, so the loader lives
here once and each caller picks the tensor or array flavour it wants.
"""
import json
from pathlib import Path

import numpy as np
import torch

from cost import (COMPOSITE_TO_IDX, FULL_TO_SHORT, JUDGED_FILES, MODEL_COSTS, MODELS,
                  N_COMPOSITES, N_MODELS, RESPONSE_FILES)
from cost import per_query_costs as cost_per_query
from utils.benign_split import group_split_indices

DATA = Path(__file__).resolve().parents[2] / "data"
EMB_ADV = DATA / "embeddings" / "adversarial_full_embeddings.pt"
EMB_BEN = DATA / "embeddings" / "r2bench_28k_embeddings.pt"
EMB_LLM = DATA / "embeddings" / "model_profile_embeddings.pt"
COST_TENSOR_PATH = DATA / "embeddings" / "adversarial_cost_tensor.pt"
LABELS = DATA / "adversarial" / "probe" / "labels.jsonl"
PROBE_INPUT = DATA / "adversarial" / "probe_input_merged.jsonl"
JUDGED_DIR = DATA / "benign" / "judged"
RESPONSE_DIR = DATA / "benign" / "responses"
CKPT_DIR = DATA / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

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

def load_benign_data(per_1000q=True, strict_costs=False, dtype=torch.float32):
    """-> embs (N,1024), qids, quality (N,M) at S0, costs (N,M), train_idx, test_idx,
    over the queries every model judged. costs are $/1000q, or $/query with
    per_1000q=False; strict_costs raises on a query with no recorded api_usage
    instead of falling back to the model's average. dtype float64 keeps the judge
    scores exactly as recorded -- k-NN ties at CORRECT_THR turn on the last bits."""
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
    per_query_costs = {m: cost_per_query(RESPONSE_DIR / RESPONSE_FILES[m], m,
                                        per_1000q=per_1000q) for m in MODELS}

    # Intersect
    valid_qids = sorted(
        set(qid_to_idx.keys()) &
        set.intersection(*(set(scores[m].keys()) for m in MODELS))
    )
    N = len(valid_qids)

    quality = torch.zeros(N, N_MODELS, dtype=dtype)
    ben_costs = torch.zeros(N, N_MODELS, dtype=dtype)
    emb_indices = []
    for i, q in enumerate(valid_qids):
        emb_indices.append(qid_to_idx[q])
        for mi, m in enumerate(MODELS):
            quality[i, mi] = scores[m][q]
            if strict_costs and q not in per_query_costs[m]:
                raise KeyError(f"{q} has a judge score but no per-query cost for {m} "
                               f"(missing/empty api_usage) -- re-probe it or drop it")
            ben_costs[i, mi] = per_query_costs[m].get(q, MODEL_COSTS[m])

    embs_subset = embs[emb_indices]

    # Group-aware: duplicate query texts never straddle train/test.
    train_idx, test_idx = group_split_indices(valid_qids, query_text, test_frac=0.15)

    print(f"  Benign: {N} queries, train={len(train_idx)}, test={len(test_idx)}")
    return embs_subset, valid_qids, quality, ben_costs, train_idx, test_idx

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

def load_baseline_data():
    """The same benign set as numpy arrays and $/query, plus the full 28K embedding
    table the baselines index by question id. A missing cost is fatal here rather
    than averaged over."""
    embs, qids, quality, costs, train_idx, test_idx = load_benign_data(
        per_1000q=False, strict_costs=True, dtype=torch.float64)
    raw = torch.load(EMB_BEN, map_location="cpu", weights_only=False)
    return {"all_embs": raw["embeddings"],
            "q_id_to_idx": {str(q): i for i, q in enumerate(raw["question_ids"])},
            "qids": qids, "emb": embs.numpy(), "score": quality.numpy(),
            "cost": costs.numpy(), "train_idx": train_idx, "test_idx": test_idx}
