#!/bin/bash
# Unified judge entry point. Two distinct judge passes, one script.
#
#   ./judge.sh quality [--check|--resume]   Benign QUALITY judge (Qwen3.5-122B,
#                                           in-process) over data/benign/responses
#                                           -> data/benign/judged.
#
#   ./judge.sh asr [flags]                  Adversarial ASR/PAIR judge (DeepSeek-
#                                           V4-Flash, local vLLM server) re-scoring
#                                           atoms.jsonl -> regenerates labels.jsonl.
#       asr flags: --only-failed  (just empty/unparseable slots)
#                  --dry-run      (plan only, no server/GPU)
#                  --keep-server  (leave the judge server up afterward)
#       asr env:   PROBE_DIR=... CONCURRENCY=64 GPU_MEM=0.95 ASR_JUDGE_PORT=8010
#
# The two passes use different judges, models, and mechanisms (in-process model
# load vs. vLLM server); they are kept as separate subcommands on purpose.

set -e
cd "$(dirname "$0")"
VLLM_PYTHON="${VLLM_PYTHON:-python3}"

MODE="${1:-}"; shift || true

case "$MODE" in
# =========================================================================
quality)
    JUDGE_SCRIPT="../saferouter/quality_judge.py"
    INPUT_DIR="../data/benign/responses"
    OUTPUT_DIR="../data/benign/judged"
    JUDGE_MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
    TP=1; GPU_MEM=0.90; BATCH_SIZE=256

    if [ "${1:-}" == "--check" ]; then
        $VLLM_PYTHON $JUDGE_SCRIPT --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --check
        exit 0
    fi
    RESUME_FLAG=""; [ "${1:-}" == "--resume" ] && RESUME_FLAG="--resume"

    echo "============================================"
    echo "Benign quality judging  —  $JUDGE_MODEL"
    echo "  input=$INPUT_DIR  output=$OUTPUT_DIR  tp=$TP  gpu-mem=$GPU_MEM  batch=$BATCH_SIZE"
    echo "============================================"
    mkdir -p "$OUTPUT_DIR"
    $VLLM_PYTHON $JUDGE_SCRIPT \
        --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" \
        --judge-model "$JUDGE_MODEL" --tp $TP --gpu-mem $GPU_MEM \
        --batch-size $BATCH_SIZE $RESUME_FLAG
    echo "Judging complete."
    $VLLM_PYTHON $JUDGE_SCRIPT --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --check
    ;;
# =========================================================================
asr)
    REJUDGE="../saferouter/jailbreak_judge.py"   # ASR re-judge CLI (module §3)
    PROBE_DIR="${PROBE_DIR:-../data/adversarial/probe}"   # fresh re-probe output
    JUDGE_MODEL="${ASR_JUDGE_MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
    PORT="${ASR_JUDGE_PORT:-8010}"
    GPU_MEM="${GPU_MEM:-0.99}"
    MAX_LEN="${ASR_MAX_LEN:-4096}"
    CONCURRENCY="${CONCURRENCY:-32}"
    LOG_DIR="/tmp/vllm_logs"; mkdir -p "$LOG_DIR"
    ASR_BASE="http://localhost:${PORT}/v1"
    ATOMS="$PROBE_DIR/atoms.jsonl"; LABELS="$PROBE_DIR/labels.jsonl"; QATOMS="$PROBE_DIR/query_atoms.jsonl"

    PASS=""; KEEP_SERVER=0; DRY=0
    for a in "$@"; do
        case "$a" in
            --keep-server) KEEP_SERVER=1 ;;
            --dry-run)     DRY=1; PASS="$PASS --dry-run" ;;
            --only-failed) PASS="$PASS --only-failed" ;;
            *)             PASS="$PASS $a" ;;
        esac
    done

    echo "============================================"
    echo "Adversarial ASR (PAIR) re-judge  —  local $JUDGE_MODEL"
    echo "  probe=$PROBE_DIR  port=$PORT  gpu-mem=$GPU_MEM  max-len=$MAX_LEN  conc=$CONCURRENCY"
    echo "============================================"

    if [ "$DRY" == "1" ]; then
        $VLLM_PYTHON "$REJUDGE" --atoms "$ATOMS" --labels "$LABELS" --qatoms "$QATOMS" $PASS
        exit 0
    fi
    [ -f "$ATOMS" ] || { echo "ERROR: $ATOMS not found — run the target probe first."; exit 1; }

    cleanup() {
        [ "$KEEP_SERVER" == "1" ] && { echo "Leaving ASR judge server up on :$PORT (--keep-server)."; return; }
        local pids; pids=$(lsof -ti:"$PORT" 2>/dev/null || true)
        [ -n "$pids" ] && { echo "Stopping ASR judge server on :$PORT..."; kill $pids 2>/dev/null || true; }
    }
    trap cleanup EXIT

    if curl -sf "$ASR_BASE/models" >/dev/null 2>&1; then
        echo "ASR judge already running at $ASR_BASE — reusing it."
    else
        echo "Starting $JUDGE_MODEL on :$PORT ..."
        nohup vllm serve "$JUDGE_MODEL" \
            --port "$PORT" --host 0.0.0.0 --gpu-memory-utilization "$GPU_MEM" \
            --kv-cache-dtype fp8 --max-model-len "$MAX_LEN" --served-model-name "$JUDGE_MODEL" \
            > "$LOG_DIR/asr-judge.log" 2>&1 &
        SERVER_PID=$!
        echo -n "  waiting for server (pid $SERVER_PID, log $LOG_DIR/asr-judge.log)"
        UP=0
        for i in $(seq 1 120); do          # up to 20 min for the model to load
            if curl -sf "$ASR_BASE/models" >/dev/null 2>&1; then echo " UP (~$((i*10))s)"; UP=1; break; fi
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo " FAILED (process exited)"; tail -20 "$LOG_DIR/asr-judge.log" | sed 's/^/    /'; exit 1
            fi
            sleep 10
        done
        [ "$UP" == "1" ] || { echo " TIMEOUT"; tail -20 "$LOG_DIR/asr-judge.log" | sed 's/^/    /'; exit 1; }
    fi

    export ASR_JUDGE_BASE_URL="$ASR_BASE"
    export ASR_JUDGE_MODEL="$JUDGE_MODEL"
    $VLLM_PYTHON "$REJUDGE" \
        --atoms "$ATOMS" --labels "$LABELS" --qatoms "$QATOMS" \
        --concurrency "$CONCURRENCY" --overwrite $PASS
    echo "ASR re-judge complete -> labels regenerated in $PROBE_DIR"
    echo "Next: rebuild the cost tensor  (cd ../saferouter && python -m cost.build_cost_tensor)"
    ;;
# =========================================================================
*)
    echo "usage: $0 {quality|asr} [flags]" >&2
    echo "  quality [--check|--resume]            benign quality judge (Qwen3.5-122B)" >&2
    echo "  asr [--only-failed|--dry-run|--keep-server]   adversarial ASR/PAIR re-judge (DeepSeek)" >&2
    exit 2
    ;;
esac
