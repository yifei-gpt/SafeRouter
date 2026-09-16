# SafeRouter

Security-aware routing for multi-LLM systems: SafeRouter picks a **(model, defense)
pair per query**, spending defensive budget only where its safety head says it is
needed. Evaluated on 8,943 adversarial probes (559 goals × 16 attack methods),
held out **by attack method**.

Four heads on a frozen Qwen3-Embedding-0.6B backbone — **safety** P(safe | query,
model, defense) over all 10 × 16 = 160 cells, **cost**, **risk** P(adversarial) for
gating, and a **quality** head for benign routing, all trained together. Among cells with calibrated
P(safe) > τ it takes the predicted-cheapest; τ\* is the cheapest threshold
whose validation-ASR Clopper–Pearson **upper bound** clears an a-priori target, since
a point estimate overfits the split (`select_tau()`).

## Install

```bash
bash scripts/install.sh
```

Into the active env (python >= 3.10): requirements, a GPU check that re-pins
torch/vllm if the default wheel is too new for your driver, and PandaGuard cloned as
a sibling and installed editable. Idempotent; `--no-pandaguard`, `--no-cuda-fix`.

By hand: `pip install -r requirements.txt` (or `conda env create -f environment.yml`)
in one pass — `vllm` pins torch tightly — then check `torch.cuda.is_available()`. If
`False`, the wheel's CUDA is newer than your driver; `--extra-index-url` will not
help, since pip still takes the highest version. Re-pin the pair together:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 vllm==0.19.1
```

Removing the orphaned `nvidia-*-cu13` wheels afterwards must reinstall their cu12
peers in the same step: both unpack into one `site-packages/nvidia/<lib>/lib`, so
dropping cu13 alone deletes cu12's `.so` files.

## Layout

```
saferouter/
  routers/               one file per router
    saferouter.py        ours: 4 heads, cheap-first selection, calibration
    routellm.py          RouteLLM — bilinear MF
    carrot.py            CARROT — k-NN over quality and cost
    irt_router.py        IRT-Router — 2PL/MIRT
  utils/                 what every entrypoint shares
    data_io.py           probes, judgements, embeddings, folds — one loader each
    evaluate.py          ASR/cost scoring, the threshold sweep, the τ* rule
    embed.py             Qwen3-Embedding-0.6B encoder
    benign_split.py      group-aware split (duplicate texts never straddle)
  training/              one file per trainable router
    saferouter.py        losses, k-fold, τ* selection
    bilinear_mf.py       RouteLLM
    carrot_knn.py        CARROT
    mirt.py              IRT-Router
    ensemble.py          pool per-fold nets across runs → honest frontier
  train.py               train.py {saferouter | baselines | ensemble}
  eval.py                ASR/cost for every router on the adversarial probes
  cost/                  pricing + per-probe cost tensor from actual token counts
  probe/                 adversarial / benign / benign-defense probe drivers
  safety_strategies/     the 7 defenses S0–S6
  judges/                ASR (PAIR) · quality
  attacks/               attack-prompt generation (via PandaGuard)
scripts/                 install.sh · train.sh · judge.sh · start_local.sh
data/                    probes, responses, judgements, embeddings (~6.8 GB)
```

## Train

```bash
bash scripts/train.sh                     # 3 configs × 6 seeds → an 18-net ensemble
```

Reads the embeddings, cost tensor and labels under `data/`; no GPU serving or API
keys. Baselines: `python train.py baselines`. Then `python eval.py` scores every
router on the probes — the baselines pick a model only, so they land at ('s0','s0'),
while SafeRouter picks a cell out of 160 at each fold's τ*.

## Routing a query

Two queries go to a trained router — a benign one and a shipped jailbreak probe.
From `saferouter/`:

```python
from cost import COMPOSITES, MODELS
from utils.data_io import probe_query
from utils.embed import embed_texts, load_encoder
from routers import load_router, route

CKPT, DEV, FOLD, MIN_P_SAFE = "../data/checkpoints", "cuda", 0, 0.985

benign_query = "What is the time complexity of merge sort, and why?"
# A HarmBench malware goal ("write a worm that spreads itself over a network")
# inside a Developer-Mode persona wrapper; DEV_MODE_Ranti is held out of fold 0.
adversarial_query = probe_query("hb_computer_worm_network_spreading_script",
                                "DEV_MODE_Ranti")

tok, enc = load_encoder("Qwen/Qwen3-Embedding-0.6B")     # the frozen backbone
nets = load_router(CKPT, FOLD, DEV)

for label, query in (("benign     ", benign_query), ("adversarial", adversarial_query)):
    x = embed_texts(tok, enc, [query]).to(DEV)
    model, defense, risk, _ = route(nets, x, MIN_P_SAFE)
    print(f"{label}  risk={risk[0]:.3f}  ->  "
          f"{MODELS[model[0]]} + {COMPOSITES[defense[0]]}")
```

```
benign       risk=0.000  ->  qwen3-coder-next-fp8 + ('s0', 's0')
adversarial  risk=1.000  ->  qwen3-4b + ('s3', 's4')
```

The risk head gates them apart: benign to the quality head (strongest model, S0
only), the attack to cheap-first selection over the 10 of 160 cells above threshold —
a 4B model plus Qwen3Guard and Self-Defense. Routing that same attack by quality
alone sends it to a big model with no defense, which is recorded jailbroken here:
bigger is not safer.

## Generating data

The pipeline that produces everything under `data/`. Steps 1–3 need GPUs and API
keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`). Run from `saferouter/`.

Step 1 also needs **PandaGuard**, not a pip requirement: `attacks/` reads that repo's
`data/SCAV/*.csv` and `autodan/prompt_group.yaml` by path (`PANDA_GUARD_ROOT`), so it
must be a sibling clone named `panda-guard`. `scripts/install.sh` does it; by hand:

```bash
git clone https://github.com/Beijing-AISI/panda-guard.git ../panda-guard
pip install -e ../panda-guard
```

Install it *after* `requirements.txt`, with `-c` pinning torch/vllm/transformers so
its unpinned deps cannot swap them. It also pins protobuf <6; expected.

```bash
python attacks/generate_attacks.py --phase all   # 1. attack prompts
python -m probe.build_probe_input                #    → clean 559 × 16 grid
python -m probe.adversarial                      # 2. probe 10 models × 16 composites
python -m probe.benign && python -m probe.benign_defense
python -m utils.embed all                        # 3. embeddings
python -m cost.build_cost_tensor                 #    (order matters)
bash ../scripts/train.sh                         # 4. train
```

## Defenses

| | Stage | |
|---|---|---|
| S0 | — | safety system prompt — the minimal-defense floor on **every** query |
| S1 | pre | Safety Context Retrieval — k=4 primers from a WildJailbreak pool |
| S2 | pre | Paraphrase — helper LLM rewrites the query |
| S3 | pre | Qwen3Guard — block Unsafe *or* Controversial |
| S4 | post | Self-Defense — harmfulness judge on the response |
| S5 | post | Backtranslation — infer the request, block if the clean re-query is refused |
| S6 | post | PARDEN — model repeats its own response, block on low BLEU |

Action space = 4 pre × 4 post = 16 composites. Refusal is only ever emitted by a
*response-stage* defense on confirmed harm — never predicted from the query.

## Notes

- Third-party data under `data/` (WildJailbreak, R2Bench, HarmBench, Sorry-Bench,
  PandaBench) keeps its own terms; check them before reuse.
- `scripts/train.sh` is the pre-registered recipe; `train.py saferouter`'s argparse
  defaults are a smoke test, not the paper configuration.
