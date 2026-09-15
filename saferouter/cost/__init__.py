"""Pricing and cost computation -- single source of truth. Prices from
cost/price.json; every REPORTED cost comes from actual per-call tokens
(build_cost_tensor.py). The one exception is train_saferouter's static COST_MATRIX,
a prior that only sets cheap-first ORDER and feeds no reported number.

    call_cost = (in_tok * p_in + out_tok * p_out) / 1e6   [$]   (x1e3 for $/1000q)
    S0 = q*p_i + a*p_o                    S3 pass  = Guard(g) + S0
    S1 = (q+s)*p_i + b*p_o                S3 block = Guard(g)        # no generation
    S2 = Llama(h) + (p*p_i + c*p_o)       S4       = S0 + GPT(hm)
    S5 = S0 + GPT(rf)                     # gate always runs
       + Llama(bt) + (r_i*p_i + r_o*p_o) + GPT(rq)    # only when gated

Composite (pre x post): s1/s2 replace the generation, s3 may block before it,
s4/s5 append helper calls. S4 blocks on a HARMFULNESS judge and S5 on a REFUSAL
detector -- not the PAIR judge, which only produces the ASR label.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# ─── Model pool (10 models) ───

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

# probe/ names every response and judgement file after the full model id, so the
# two filename tables are derivable rather than another list to keep in sync.
RESPONSE_FILES = {m: SHORT_TO_FULL[m].replace("/", "_").replace(".", "_").lower()
                     + "_r2bench.jsonl" for m in MODELS}
JUDGED_FILES = {m: SHORT_TO_FULL[m].replace("/", "_").replace(".", "_").lower()
                   + "_judged.jsonl" for m in MODELS}

# Composites (4 pre x 4 post = 16). s6 = PARDEN: repeat the response, block on low BLEU.

PRE = ["s0", "s1", "s2", "s3"]
POST = ["s0", "s4", "s5", "s6"]
COMPOSITES = [(p, q) for p in PRE for q in POST]
N_COMPOSITES = len(COMPOSITES)
COMPOSITE_TO_IDX = {c: i for i, c in enumerate(COMPOSITES)}

# ─── Load pricing from price.json ───

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


# Guard helper cost ($/1000q); Qwen3Guard-0.6B measured at 529 in + 10 out per call.
GUARD_ABS_COST = (529 * 0.02 + 10 * 0.05) / 1e6 * 1e3       # $0.0111/1000q

# Realized cost when S3 blocks: guard call, no generation.
GUARD_BLOCK_COST = GUARD_ABS_COST


def call_cost(model_id, in_tok, out_tok):
    """Dollar cost of one API call from actual token counts. `model_id` is matched
    case-insensitively against price.json. Returns raw dollars, not $/1000q."""
    p = _PRICES.get(model_id.lower(), (0.0, 0.0))
    return (in_tok * p[0] + out_tok * p[1]) / 1e6
