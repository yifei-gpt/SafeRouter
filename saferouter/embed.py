#!/usr/bin/env python3
"""Embed adversarial probes, benign queries, or model profiles.

    python embed.py {adversarial|benign|profiles|all} [--out PATH]
"""
import argparse, json
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
MAX_LEN = 4096   # no truncation: max observed 2013 tok (benign) / 1579 (adv cleaned); generous
                 # ceiling; dynamic padding costs nothing unless exceeded. Left-truncation keeps the goal.

MODELS_POOL = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
    "qwen3-14b", "qwen3-30b-a3b", "qwen3-32b",
    "qwen3-coder-next-fp8",
    "nemotron-3-super-120b", "nemotron-3-nano-30b",
]

PROFILES = {
    "qwen3-0.6b": "Qwen3-0.6B is a 0.6B parameter dense Transformer from Alibaba's Qwen team, 2025. Ultra-efficient on-device inference.",
    "qwen3-1.7b": "Qwen3-1.7B is a 1.7B parameter dense Transformer from Alibaba's Qwen team, 2025. Edge deployment with stronger understanding.",
    "qwen3-4b": "Qwen3-4B is a 4B parameter dense Transformer from Alibaba's Qwen team, 2025. Balanced small-scale model.",
    "qwen3-8b": "Qwen3-8B is an 8B parameter dense Transformer from Alibaba's Qwen team, 2025. Strong instruction following and reasoning.",
    "qwen3-14b": "Qwen3-14B is a 14B parameter dense Transformer from Alibaba's Qwen team, 2025. High accuracy on complex reasoning.",
    "qwen3-30b-a3b": "Qwen3-30B-A3B is a 30B MoE model (3B active) from Alibaba's Qwen team, 2025. High quality at low inference cost.",
    "qwen3-32b": "Qwen3-32B is a 32B parameter dense Transformer from Alibaba's Qwen team, 2025. Largest dense Qwen3 model.",
    "qwen3-coder-next-fp8": "Qwen3-Coder-Next is an 80B MoE (3B active) from Alibaba, 2025. Specialized for code generation.",
    "nemotron-3-super-120b": "NVIDIA Nemotron-3-Super-120B is a 120B MoE (12B active) from NVIDIA, 2025. Mamba2 hybrid architecture.",
    "nemotron-3-nano-30b": "NVIDIA Nemotron-3-Nano-30B is a 30B MoE (3B active) from NVIDIA, 2025. Efficient Mamba2 hybrid.",
}


def last_token_pool(last_hidden, attn_mask):
    """Last-token pooling, correct for BOTH left- and right-padded batches.

    The tokenizer here pads LEFT, so the real last token is at index -1; the
    right-padding formula (attn_mask.sum(1) - 1) would index a PAD position for
    any sequence shorter than the batch max. The branch below handles both."""
    if attn_mask[:, -1].sum() == attn_mask.shape[0]:   # left-padded (or unpadded)
        return last_hidden[:, -1]
    seq_len = attn_mask.sum(dim=1) - 1                  # right-padded
    return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_len]


def load_encoder(model_id="Qwen/Qwen3-Embedding-0.6B"):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", truncation_side="left", trust_remote_code=True)
    enc = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True,
                                    attn_implementation="eager").to(DEVICE).eval()
    return tok, enc


def assert_embeddings_distinct(embs, texts, name, min_ratio=0.99):
    """Guard against the truncation/pooling corruption: DISTINCT input texts must
    yield DISTINCT embeddings. (The old bug collapsed e.g. 9756 probes to 6907
    unique rows, or MiniLM to 5143/9756.) Genuinely-identical texts SHOULD share
    a vector, so we compare DISTINCT embeddings vs DISTINCT texts, not raw rows."""
    n_text = len({t for t in texts})
    # Round normalized vectors so float noise doesn't inflate the unique count.
    rounded = (embs.float() * 1e4).round()
    n_emb = int(torch.unique(rounded, dim=0).shape[0])
    print(f"  [{name}] uniqueness: {len(texts)} rows, {n_text} distinct texts, {n_emb} distinct embeddings")
    if n_emb < min_ratio * n_text:
        raise AssertionError(
            f"EMBEDDING CORRUPTION in {name}: {n_text} distinct texts collapsed to only "
            f"{n_emb} distinct embeddings (ratio {n_emb / max(n_text,1):.3f} < {min_ratio}). "
            f"Distinct texts produced identical vectors — the left-truncation/last-token-pool "
            f"fix is not taking effect. NOT saving; investigate before re-embedding.")


def embed_texts(tok, enc, texts, batch_size=BATCH):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embed"):
        # Char pre-guard above any real input so MAX_LEN governs; keeps the LAST chars.
        batch = [t[-20000:] for t in texts[i:i + batch_size]]
        inp = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            h = last_token_pool(enc(**inp).last_hidden_state, inp["attention_mask"])
            embs.append(F.normalize(h, p=2, dim=-1).float().cpu())
    return torch.cat(embs, dim=0)


def _derive_query_id(row):
    """query_id from an atoms row (has `query_id`) OR a probe_input row (`question_id`).
    Mirrors probe/adversarial.py: keep pb_/hb_/sb_-namespaced ids, else prefix pb_.
    Lets embed_adversarial read either atoms.jsonl or the cleaned probe_input."""
    qid = row.get("query_id")
    if qid is None:
        qid = str(row["question_id"])
        if not qid.startswith(("pb_", "hb_", "sb_")):
            qid = f"pb_{qid}"
    return qid


def embed_adversarial(atoms_path, out_path, model_id="Qwen/Qwen3-Embedding-0.6B", batch_size=BATCH):
    key_to_text = {}
    for line in open(atoms_path):
        a = json.loads(line)
        key = (_derive_query_id(a), a["attack_method"])
        if key not in key_to_text:
            key_to_text[key] = a["query"]
    keys = sorted(key_to_text.keys())
    texts = [key_to_text[k] for k in keys]
    print(f"Embedding {len(keys)} adversarial probes")
    tok, enc = load_encoder(model_id)
    embs = embed_texts(tok, enc, texts, batch_size=batch_size)
    assert_embeddings_distinct(embs, texts, "adversarial")
    torch.save({"embeddings": embs, "keys": keys, "key_to_idx": {k: i for i, k in enumerate(keys)}}, out_path)
    print(f"Saved → {out_path} {tuple(embs.shape)}")


def embed_benign(responses_dir, out_path, model_id="Qwen/Qwen3-Embedding-0.6B", batch_size=BATCH):
    response_files = sorted(Path(responses_dir).glob("*.jsonl"))
    rows = [json.loads(l) for l in open(response_files[0])]
    texts = [r["query"] for r in rows]
    ids = [str(r["id"]) for r in rows]
    print(f"Embedding {len(texts)} benign queries")
    tok, enc = load_encoder(model_id)
    embs = embed_texts(tok, enc, texts, batch_size=batch_size)
    assert_embeddings_distinct(embs, texts, "benign")
    torch.save({"embeddings": embs, "question_ids": ids}, out_path)
    print(f"Saved → {out_path} {tuple(embs.shape)}")


def embed_profiles(out_path, model_id="Qwen/Qwen3-Embedding-0.6B"):
    tok, enc = load_encoder(model_id)
    embeddings = {}
    for m in MODELS_POOL:
        inp = tok([PROFILES[m]], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            # Same last-token pooling as embed_texts, so query and model embeddings share one space.
            h = last_token_pool(enc(**inp).last_hidden_state, inp["attention_mask"])
            # .float() to match embed_texts, else fp16 profiles break MIRT's theta_proj.
            embeddings[m] = F.normalize(h, p=2, dim=-1).float().cpu().squeeze(0)
        print(f"  {m}: {tuple(embeddings[m].shape)}")
    torch.save(embeddings, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["adversarial", "benign", "profiles", "all"])
    p.add_argument("--atoms", default="../data/adversarial/probe/atoms.jsonl")
    p.add_argument("--responses-dir", default="../data/benign/responses")
    p.add_argument("--out-dir", default="../data/embeddings")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch", type=int, default=BATCH,
                   help="encoder batch size (32 fits a B200; use 8 on a 24GB card — "
                        "eager attention on ~1.6k-token batches OOMs at 32)")
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode in ("adversarial", "all"):
        embed_adversarial(args.atoms, f"{args.out_dir}/adversarial_full_embeddings.pt", args.model,
                          batch_size=args.batch)
    if args.mode in ("benign", "all"):
        embed_benign(args.responses_dir, f"{args.out_dir}/r2bench_28k_embeddings.pt", args.model,
                     batch_size=args.batch)
    if args.mode in ("profiles", "all"):
        embed_profiles(f"{args.out_dir}/model_profile_embeddings.pt", args.model)
