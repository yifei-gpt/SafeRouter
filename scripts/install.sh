#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_DIR="$(dirname "$REPO")/panda-guard"
PG_URL="https://github.com/Beijing-AISI/panda-guard.git"
PIN=(torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 vllm==0.19.1)

WANT_PG=1; CUDA_FIX=1
for a in "$@"; do case "$a" in
  --no-pandaguard) WANT_PG=0 ;;
  --no-cuda-fix)   CUDA_FIX=0 ;;
  -h|--help) echo "usage: install.sh [--no-pandaguard] [--no-cuda-fix]"; exit 0 ;;
  *) echo "unknown flag: $a" >&2; exit 2 ;;
esac; done

PY="$(command -v python3 || command -v python)"
say(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }
gpu(){ "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
gpuinfo(){ "$PY" -c 'import torch; print(f"torch {torch.__version__}, {torch.cuda.device_count()} GPU(s)")'; }

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || die "need python >= 3.10"

say "pip"
"$PY" -m pip install -q -U pip setuptools wheel
"$PY" -m pip -V

say "requirements"
"$PY" -m pip install -r "$REPO/requirements.txt"

say "gpu"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "no nvidia-smi -- SafeRouter needs an NVIDIA GPU"
elif gpu; then
  gpuinfo
elif [ "$CUDA_FIX" = 0 ]; then
  die "torch cannot reach the GPU; re-pin with: pip install ${PIN[*]}"
else
  echo "driver too old for this wheel -- re-pinning torch+vllm"
  "$PY" -m pip install "${PIN[@]}"
  gpu || die "still no GPU; pick a build at https://pytorch.org/get-started/locally/"

  ORPHANS="$("$PY" - <<'EOF'
from importlib.metadata import distributions
print(" ".join(sorted(
    d.metadata["Name"] for d in distributions()
    if (d.metadata["Name"] or "").startswith("nvidia-")
    and (d.metadata["Name"] or "").endswith("-cu13"))))
EOF
)"
  if [ -n "$ORPHANS" ]; then
    REPAIR="$("$PY" - $ORPHANS <<'EOF'
import sys
from importlib.metadata import version, PackageNotFoundError
out = []
for name in sys.argv[1:]:
    peer = name[:-len("-cu13")] + "-cu12"
    try: out.append(f"{peer}=={version(peer)}")
    except PackageNotFoundError: pass
print(" ".join(out))
EOF
)"
    echo "removing orphaned cu13 wheels: $ORPHANS"
    "$PY" -m pip uninstall -q -y $ORPHANS
    [ -n "$REPAIR" ] && "$PY" -m pip install -q --force-reinstall --no-deps $REPAIR
    gpu || die "GPU broke while removing cu13 wheels; repair with: pip install --force-reinstall --no-deps $REPAIR"
  fi
  gpuinfo
fi

if [ "$WANT_PG" = 1 ]; then
  say "panda-guard"
  if [ -d "$PG_DIR/.git" ]; then
    echo "already at $PG_DIR"
  elif [ -e "$PG_DIR" ]; then
    die "$PG_DIR exists and is not a git clone"
  else
    git clone --quiet "$PG_URL" "$PG_DIR"
    echo "cloned to $PG_DIR"
  fi
  CONS="$(mktemp)"; trap 'rm -f "$CONS"' EXIT
  "$PY" - "$CONS" <<'EOF'
import sys
from importlib.metadata import version, PackageNotFoundError
with open(sys.argv[1], "w") as f:
    for pkg in ("torch", "torchvision", "torchaudio", "vllm", "transformers", "numpy"):
        try: f.write(f"{pkg}=={version(pkg)}\n")
        except PackageNotFoundError: pass
EOF
  "$PY" -m pip install -e "$PG_DIR" -c "$CONS"
fi

say "verify"
REPO="$REPO" PG="$WANT_PG" "$PY" - <<'EOF'
import importlib, os, sys
from pathlib import Path

repo = Path(os.environ["REPO"])
sys.path.insert(0, str(repo / "saferouter"))
bad = []

def check(label, mods):
    start = len(bad)
    for m in mods:
        try: importlib.import_module(m)
        except Exception as e: bad.append(f"{m}: {e}")
    print(f"{label}: {'ok' if len(bad) == start else 'FAILED'}")

check("core", ["torch", "numpy", "scipy", "sklearn", "joblib", "transformers",
               "sentence_transformers", "pandas", "tqdm", "openai", "httpx", "nltk"])
check("saferouter", ["train_saferouter", "mega_ensemble", "train_all_routers", "embed"])

if os.environ["PG"] == "1":
    check("panda_guard", ["panda_guard.llms", "panda_guard.role.attacks.pair",
                          "panda_guard.role.attacks.autodan.autodan",
                          "panda_guard.role.attacks.scav",
                          "panda_guard.role.judges.llm_based"])
    pg = repo.parent / "panda-guard"
    for rel in ("data/SCAV/optimized_instructions_8b.csv",
                "data/SCAV/optimized_instructions_70b.csv",
                "src/panda_guard/role/attacks/autodan/prompt_group.yaml"):
        if not (pg / rel).exists(): bad.append(f"missing {pg / rel}")

need = ["embeddings/adversarial_full_embeddings.pt",
        "embeddings/adversarial_cost_tensor.pt",
        "adversarial/probe/labels.jsonl"]
have = sum((repo / "data" / r).exists() for r in need)
print(f"data: {have}/{len(need)} training inputs"
      + ("" if have == len(need) else "  -> generate it, see README"))

if bad:
    print("\nFAILED:")
    for b in bad: print(" ", b)
    sys.exit(1)
EOF

say "done"
echo "generate data:  cd saferouter && python attacks/generate_attacks.py --phase all"
echo "train:          bash scripts/train.sh"
