"""Safety Context Retrieval (S1): retrieve the top K=4 similar safety contexts from
a pre-embedded WildJailbreak pool and prepend them as few-shot examples.

Chen et al., arXiv:2505.15753.
"""

import json
from pathlib import Path

import numpy as np

from .base import (BaseStrategy, StrategyResult, SAFETY_SYSTEM_PROMPT,
                   _add_reasoning_disable)

SSROUTER_DIR = Path(__file__).resolve().parents[1]


def _pick_least_loaded_device() -> str:
    """Pick `cuda:N` with the most free memory; fall back to `cpu`."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    best_idx, best_free = 0, -1
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_idx, best_free = i, free
    return f"cuda:{best_idx}"

# SCR primers AUGMENT S0's prompt rather than replace it, so the worst case is still >= S0.
SCR_PRIMER_BLOCK = """

Below are examples of unsafe requests and the safe responses you should give to similar requests:

{examples}

Apply the same safety reasoning to the following query."""

# Pool lives under <repo>/data/scr_pool (built by build_scr_pool.py).
POOL_DIR = Path(__file__).resolve().parents[2] / "data" / "scr_pool"


class SafetyContextRetrievalStrategy(BaseStrategy):
    """Retrieve relevant safety contexts and prepend to prompt."""

    name = "safety_context_retrieval"
    reference = "Chen et al., arXiv:2505.15753 (preprint, May 2025)"

    def __init__(self, client, model, k: int = 4,
                 embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 embedding_device: str = None, **kwargs):
        super().__init__(client, model)
        self.k = k
        self._load_pool()      # raises if pool not on disk
        from sentence_transformers import SentenceTransformer
        # Disable the cuDNN SDPA path: on sm_100 it has "No valid execution plans" here.
        try:
            import torch
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        if embedding_device is None:
            embedding_device = _pick_least_loaded_device()
        self._encoder = SentenceTransformer(embedding_model, device=embedding_device)
        self.embedding_device = embedding_device

    def _load_pool(self):
        """Load pre-embedded pool from disk. Fails hard if missing —
        run `safety_strategies/build_scr_pool.py` first.

        WildJailbreak occasionally contains raw \\x01/\\x07/\\x0b chars;
        tolerate via errors='replace' + strict=False.
        """
        texts_path = POOL_DIR / "pool_texts.json"
        emb_path = POOL_DIR / "pool_embeddings.npy"

        if not texts_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"SCR pool not found at {POOL_DIR}. "
                f"Build it with: python safety_strategies/build_scr_pool.py"
            )

        with open(texts_path, encoding="utf-8", errors="replace") as f:
            self._pool_texts = json.loads(f.read(), strict=False)
        self._pool_embeddings = np.load(emb_path)

    def _retrieve(self, query: str) -> str:
        query_emb = self._encoder.encode([query], normalize_embeddings=True)[0]
        similarities = self._pool_embeddings @ query_emb
        top_k_idx = np.argsort(similarities)[-self.k:][::-1]
        selected = [self._pool_texts[i] for i in top_k_idx]

        examples = []
        for i, ctx in enumerate(selected, 1):
            req = ctx.get("request", "")
            resp = ctx.get("response", "")
            examples.append(f"Safe Example {i}:\nUser: {req}\nAssistant: {resp}")
        return "\n\n".join(examples)

    def apply(self, query: str, max_tokens: int = 4096) -> StrategyResult:
        try:
            examples = self._retrieve(query)
            # Additive SCR: keep S0's prompt and append the primers, so the worst case is still >= S0.
            system_prompt = SAFETY_SYSTEM_PROMPT + SCR_PRIMER_BLOCK.format(examples=examples)
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=120.0,
            )
            _add_reasoning_disable(kwargs)
            resp = self.client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage

            return StrategyResult(
                response=text,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                strategy_tokens=0,
                strategy_prompt_tokens=0,
                strategy_completion_tokens=0,
                strategy_calls=0,
            )
        except Exception as e:
            return StrategyResult(response="", error=str(e))
