"""Pricing and cost, the single source of truth. Prices from price.json; every
REPORTED cost comes from actual per-call tokens (build_cost_tensor.py). The
lone exception is COST_MATRIX, a prior that only sets cheap-first ORDER.

    call_cost = (in_tok * p_in + out_tok * p_out) / 1e6  [$]  (x1e3 per 1000q)
    S0 = q*p_i + a*p_o                    S3 pass  = Guard(g) + S0
    S1 = (q+s)*p_i + b*p_o                S3 block = Guard(g)   # no generation
    S2 = Llama(h) + (p*p_i + c*p_o)       S4       = S0 + GPT(hm)
    S5 = S0 + GPT(rf)                     # gate always runs
       + Llama(bt) + (r_i*p_i + r_o*p_o) + GPT(rq)    # only when gated

s1/s2 replace the generation, s3 may block before it, s4/s5 append helpers. S4
blocks on a HARMFULNESS judge and S5 on a REFUSAL detector -- not the PAIR
judge, which only produces the ASR label.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# ---- Model pool (10 models) ----

MODELS = [
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b",
    "qwen3-14b", "qwen3-30b-a3b", "qwen3-32b",
    "qwen3-coder-next-fp8",
    "nemotron-3-super-120b", "nemotron-3-nano-30b",
]
N_MODELS = len(MODELS)

SHORT_TO_FULL = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-coder-next-fp8": "Qwen/Qwen3-Coder-Next-FP8",
    "nemotron-3-super-120b": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nemotron-3-nano-30b": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
}
FULL_TO_SHORT = {v: k for k, v in SHORT_TO_FULL.items()}

# probe/ names every file after the full model id, so both filename tables
# derive rather than being another list to keep in sync.
RESPONSE_FILES = {m: SHORT_TO_FULL[m].replace("/", "_").replace(".", "_").lower()
                     + "_r2bench.jsonl" for m in MODELS}
JUDGED_FILES = {m: SHORT_TO_FULL[m].replace("/", "_").replace(".", "_").lower()
                   + "_judged.jsonl" for m in MODELS}

# Composites: 4 pre x 4 post = 16.

PRE = ["s0", "s1", "s2", "s3"]
POST = ["s0", "s4", "s5", "s6"]
COMPOSITES = [(p, q) for p in PRE for q in POST]
N_COMPOSITES = len(COMPOSITES)
COMPOSITE_TO_IDX = {c: i for i, c in enumerate(COMPOSITES)}

# ---- Load pricing from price.json ----

_price_data = json.load(open(HERE / "price.json"))["models"]
_PRICES = {}
for entry in _price_data.values():
    _PRICES[entry["model_id"].lower()] = (entry["input"], entry["output"])

# Per-model pricing ($/M tokens): (input, output)
MODEL_PRICING = {m: (_price_data[_key]["input"], _price_data[_key]["output"])
                 for m, _key in [
                     ("qwen3-0.6b", "qwen3-0.6b"),
                     ("qwen3-1.7b", "qwen3-1.7b"),
                     ("qwen3-4b", "qwen3-4b"),
                     ("qwen3-8b", "qwen3-8b"),
                     ("qwen3-14b", "qwen3-14b"),
                     ("qwen3-30b-a3b", "qwen3-30b-a3b"),
                     ("qwen3-32b", "qwen3-32b"),
                     ("qwen3-coder-next-fp8", "qwen3-coder-next"),
                     ("nemotron-3-super-120b", "nemotron-3-super-120b"),
                     ("nemotron-3-nano-30b", "nemotron-3-nano-30b"),
                 ]}

# Static prior ($/1000q): sets cheap-first ORDER; reported cost is measured.
MODEL_COSTS = {
    "qwen3-0.6b": 0.0054, "qwen3-1.7b": 0.0161, "qwen3-4b": 0.0254,
    "qwen3-8b": 0.2938, "qwen3-14b": 0.1733, "qwen3-30b-a3b": 0.3289,
    "qwen3-32b": 0.2025, "qwen3-coder-next-fp8": 0.8074,
    "nemotron-3-super-120b": 0.5036, "nemotron-3-nano-30b": 0.0655,
}

# Input-only prices ($/M tokens) — for computing S1 extra-token overhead
MODEL_INPUT_PRICES = {m: p[0] for m, p in MODEL_PRICING.items()}

def per_query_costs(path, model, per_1000q=True):
    """query id -> cost, from the api_usage recorded in a response file.
    $/1000q by default; per_1000q=False gives $/query. Queries whose api_usage is
    missing or empty are absent from the result -- callers decide whether that is
    fatal rather than getting a silent average."""
    inp, out = MODEL_PRICING[model]
    scale = 1e3 if per_1000q else 1.0
    costs = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            usage = row.get("api_usage") or {}
            if usage:
                costs[str(row["id"])] = (usage.get("prompt_tokens", 0) * inp
                                         + usage.get("completion_tokens", 0) * out) / 1e6 * scale
    return costs


# Guard cost ($/1000q); Qwen3Guard-0.6B at 529 in + 10 out per call.
GUARD_ABS_COST = (529 * 0.02 + 10 * 0.05) / 1e6 * 1e3       # $0.0111/1000q

# Realized cost when S3 blocks: guard call, no generation.
GUARD_BLOCK_COST = GUARD_ABS_COST


def call_cost(model_id, in_tok, out_tok):
    """Dollar cost of one API call from actual token counts. `model_id` is matched
    case-insensitively against price.json. Returns raw dollars, not $/1000q."""
    p = _PRICES.get(model_id.lower(), (0.0, 0.0))
    return (in_tok * p[0] + out_tok * p[1]) / 1e6


# ---- Static routing cost matrix ($/1000q per (model, composite)) ----
# A prior for cheap-first ORDERING only; every reported dollar comes from
# the realized per-probe tensor, not from here.
import torch  # noqa: E402

# Helper costs ($/1000q), token counts measured from probe data.
# Llama-3.3-70B paraphrase: measured avg 289 input + 175 output tokens
PARA_ABS_COST = (289 * 0.10 + 175 * 0.32) / 1e6 * 1e3    # $0.0849/1000q
# S4 gpt-4o-mini harm judge: 326 in + 1.5 out. S5 pays it twice.
S4_ABS_COST = (326 * 0.15 + 1.5 * 0.60) / 1e6 * 1e3      # $0.0490/1000q
# Llama-3.3-70B backtranslation: avg 350 in + 24 out (N=22645)
S5_BACKTR_COST = (350 * 0.10 + 24 * 0.32) / 1e6 * 1e3    # $0.0427/1000q
# S1 SCR: extra input tokens from 4 primers, measured as target_B.in - A.in.
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
        # S5: two gpt-4o-mini judges + backtranslation + the target re-query.
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
