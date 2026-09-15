#!/usr/bin/env python3
"""Build the SCR pool from WildJailbreak (vanilla_harmful + safe refusals, embedded)
as pool_texts.json + pool_embeddings.npy. Chen et al. (arXiv:2505.15753): 50K entries.

    python safety_strategies/build_scr_pool.py [--max-entries N] [--embedding-model M]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]            # saferouter/


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-entries", type=int, default=50000)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model for retrieval. Default matches "
                             "the rest of SRouter's encoder so SCR shares a "
                             "vector space with the routing/intent heads.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[2] / "data" / "scr_pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    texts_path = out_dir / "pool_texts.json"
    emb_path = out_dir / "pool_embeddings.npy"

    if texts_path.exists() and emb_path.exists():
        with open(texts_path) as f:
            existing = json.load(f)
        print(f"Pool already exists: {len(existing)} entries")
        print("Delete files to rebuild.")
        return

    # Source corpus (WILDJAILBREAK_TSV); only needed to rebuild data/scr_pool/.
    wj_path = Path(os.environ.get(
        "WILDJAILBREAK_TSV",
        ROOT.parent / "data" / "wildjailbreak" / "train" / "train.tsv"))
    if not wj_path.exists():
        print(f"WildJailbreak not found: {wj_path}\n"
              f"Set WILDJAILBREAK_TSV, or use the prebuilt pool in data/scr_pool/.")
        return

    print(f"Loading WildJailbreak from {wj_path}...")
    pool_texts = []
    chunk_size = 50000
    # as floats — matches the dataset card's recommended load.
    for chunk in pd.read_csv(wj_path, sep='\t', chunksize=chunk_size,
                              keep_default_na=False):
        vanilla_harmful = chunk[chunk['data_type'] == 'vanilla_harmful']
        for _, row in vanilla_harmful.iterrows():
            request = str(row.get('vanilla', '')).strip()
            completion = str(row.get('completion', '')).strip()
            if request and completion and len(request) > 10 and len(completion) > 10:
                pool_texts.append({
                    "request": request[:500],     # truncate long entries
                    "response": completion[:500],
                })
                if len(pool_texts) >= args.max_entries:
                    break
        if len(pool_texts) >= args.max_entries:
            break
        print(f"  Loaded {len(pool_texts)} entries so far...")

    print(f"Total pool entries: {len(pool_texts)}")

    # Save texts
    with open(texts_path, "w") as f:
        json.dump(pool_texts, f)
    print(f"Saved texts to {texts_path}")

    # Embed requests for retrieval
    print(f"\nEmbedding with {args.embedding_model}...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    dim = encoder.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {dim}")

    requests = [entry["request"] for entry in pool_texts]
    embeddings = encoder.encode(
        requests,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(emb_path, embeddings)
    print(f"Saved embeddings to {emb_path} ({embeddings.shape})")

    # Save config
    config = {
        "n_entries": len(pool_texts),
        "embedding_model": args.embedding_model,
        "embedding_dim": dim,
        "source": "WildJailbreak vanilla_harmful",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone. Pool: {len(pool_texts)} entries, {dim}-dim embeddings")


if __name__ == "__main__":
    main()
