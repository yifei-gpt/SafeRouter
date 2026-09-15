#!/usr/bin/env bash
# Start local vLLM servers for SafeRouter's 10-model target pool + Qwen3Guard.
#
# B200 (183GB) can't fit all simultaneously, so we run in batches:
#
#   Batch 1:  Guard + 0.6B + 1.7B + 4B + 8B + 14B   (0.95 total)  Qwen3 small/medium
#   Batch 2:  Guard + 30B-A3B alone                  (0.95 total)  Qwen3 large MoE
#   Batch 3:  Guard + 32B alone                      (0.95 total)  Qwen3 large dense
#   Batch 4:  Guard + Qwen3-Coder-Next-FP8           (0.95 total)  Code-specialized
#   Batch 5:  Guard + Nemotron-3-Super-120B-NVFP4    (0.95 total)  NVIDIA (Mamba2 hybrid)
#   Batch 6:  Guard + Nemotron-3-Nano-30B-NVFP4      (0.95 total)  NVIDIA (Mamba2 hybrid)
#
# Usage:
#   ./start_local.sh [--batchN]     # start Batch N (1-6; default 1)
#   ./start_local.sh --judge        # DeepSeek-V4-Flash ASR judge, port 8010
#   ./start_local.sh --quality-judge  # Qwen3.5-122B quality judge, port 8000
#   ./start_local.sh --status       # check which servers are up
#   ./start_local.sh --stop         # kill all vLLM servers
#   ./start_local.sh --test         # quick inference test on all running servers
#   ./start_local.sh --clean        # clean vLLM/torch/triton caches (new node)

set -e

LOG_DIR="/tmp/vllm_logs"
mkdir -p "$LOG_DIR"

# ---- Cache cleanup ----

clean_cache() {
    echo "=== Cleaning vLLM / torch caches ==="
    local cleaned=0
    if [ -d "$HOME/.cache/vllm/torch_compile_cache" ]; then
        local size=$(du -sh "$HOME/.cache/vllm/torch_compile_cache" 2>/dev/null | cut -f1)
        rm -rf "$HOME/.cache/vllm/torch_compile_cache"
        echo "  Removed ~/.cache/vllm/torch_compile_cache ($size)"
        cleaned=1
    fi
    if [ -d "$HOME/.cache/vllm/modelinfos" ]; then
        rm -rf "$HOME/.cache/vllm/modelinfos"
        echo "  Removed ~/.cache/vllm/modelinfos"
        cleaned=1
    fi
    if [ -d "$HOME/.cache/torch_extensions" ]; then
        local size=$(du -sh "$HOME/.cache/torch_extensions" 2>/dev/null | cut -f1)
        rm -rf "$HOME/.cache/torch_extensions"
        echo "  Removed ~/.cache/torch_extensions ($size)"
        cleaned=1
    fi
    if [ -d "$HOME/.triton/cache" ]; then
        local size=$(du -sh "$HOME/.triton/cache" 2>/dev/null | cut -f1)
        rm -rf "$HOME/.triton/cache" 2>/dev/null || true
        echo "  Removed ~/.triton/cache ($size)"
        cleaned=1
    fi
    if [ $cleaned -eq 0 ]; then
        echo "  No caches found to clean."
    fi
    echo
}

# ═══════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════

# Batch 1: Guard + 0.6B + 1.7B + 4B + 8B + 14B (total 0.95)
declare -A BATCH1_MODELS=(
    ["qwen3guard"]="Qwen/Qwen3Guard-Gen-0.6B"
    ["qwen3-0.6b"]="Qwen/Qwen3-0.6B"
    ["qwen3-1.7b"]="Qwen/Qwen3-1.7B"
    ["qwen3-4b"]="Qwen/Qwen3-4B"
    ["qwen3-8b"]="Qwen/Qwen3-8B"
    ["qwen3-14b"]="Qwen/Qwen3-14B"
)
declare -A BATCH1_PORTS=(
    ["qwen3guard"]=8001
    ["qwen3-0.6b"]=8002
    ["qwen3-1.7b"]=8003
    ["qwen3-4b"]=8004
    ["qwen3-8b"]=8005
    ["qwen3-14b"]=8006
)
declare -A BATCH1_MEM=(
    ["qwen3guard"]=0.05
    ["qwen3-0.6b"]=0.06
    ["qwen3-1.7b"]=0.08
    ["qwen3-4b"]=0.12
    ["qwen3-8b"]=0.20
    ["qwen3-14b"]=0.44
)

# Batch 2: Guard + 30B-A3B alone (total 0.95)
declare -A BATCH2_MODELS=(
    ["qwen3-30b-a3b"]="Qwen/Qwen3-30B-A3B"
)
declare -A BATCH2_PORTS=(
    ["qwen3-30b-a3b"]=8007
)
declare -A BATCH2_MEM=(
    ["qwen3-30b-a3b"]=0.90
)

# Batch 3: Guard + 32B alone (total 0.95)
declare -A BATCH3_MODELS=(
    ["qwen3-32b"]="Qwen/Qwen3-32B"
)
declare -A BATCH3_PORTS=(
    ["qwen3-32b"]=8008
)
declare -A BATCH3_MEM=(
    ["qwen3-32b"]=0.90
)

# Batch 4: Guard + Qwen3-Coder-Next-FP8 alone (total 0.95)
declare -A BATCH4_MODELS=(
    ["qwen3-coder-next"]="Qwen/Qwen3-Coder-Next-FP8"
)
declare -A BATCH4_PORTS=(
    ["qwen3-coder-next"]=8013
)
declare -A BATCH4_MEM=(
    ["qwen3-coder-next"]=0.90
)

# Batch 5: Guard + Nemotron-3-Super-120B-NVFP4 alone (total 0.95)
declare -A BATCH5_MODELS=(
    ["nemotron-120b"]="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
)
declare -A BATCH5_PORTS=(
    ["nemotron-120b"]=8014
)
declare -A BATCH5_MEM=(
    ["nemotron-120b"]=0.90
)

# Batch 6: Guard + Nemotron-3-Nano-30B-NVFP4 alone (total 0.95)
declare -A BATCH6_MODELS=(
    ["nemotron-nano-30b"]="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)
declare -A BATCH6_PORTS=(
    ["nemotron-nano-30b"]=8015
)
declare -A BATCH6_MEM=(
    ["nemotron-nano-30b"]=0.90
)

ALL_PORTS="8000 8001 8002 8003 8004 8005 8006 8007 8008 8010 8013 8014 8015"

# ---- Helpers ----

start_and_wait() {
    local name=$1 model=$2 port=$3 mem=$4 timeout=${5:-300} extra_args=${6:-}
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "  [skip] :${port} ${name} already running"
        return 0
    fi
    echo -n "  [start] :${port} ${name} (${model}, mem=${mem}) ..."
    # --enforce-eager skips torch.compile: startup drops from >300s (cold-cache
    # recompile → timeout → set -e aborts the batch) to ~25s. Inference is a hair
    # slower but this is a batched generation probe, not a latency benchmark.
    nohup vllm serve "$model" \
        --port "$port" \
        --host 0.0.0.0 \
        --gpu-memory-utilization "$mem" \
        --max-model-len 16384 \
        --trust-remote-code \
        --enforce-eager \
        --served-model-name "$model" \
        $extra_args \
        > "${LOG_DIR}/${name}.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 $((timeout / 2))); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo " UP ($((i*2))s, pid=${pid})"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo " FAILED (process exited)"
            tail -5 "${LOG_DIR}/${name}.log" | sed 's/^/    /'
            return 1
        fi
        sleep 2
    done
    echo " TIMEOUT (${timeout}s)"
    tail -5 "${LOG_DIR}/${name}.log" | sed 's/^/    /'
    return 1
}

start_nemotron() {
    # Nemotron models need special vLLM flags: Mamba2 hybrid, NVFP4/FP8, fp8 KV cache
    local name=$1 model=$2 port=$3 mem=$4 timeout=${5:-600}
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "  [skip] :${port} ${name} already running"
        return 0
    fi
    echo -n "  [start] :${port} ${name} (${model}, mem=${mem}) ..."
    nohup vllm serve "$model" \
        --port "$port" \
        --host 0.0.0.0 \
        --served-model-name "$model" \
        --gpu-memory-utilization "$mem" \
        --max-model-len 16384 \
        --trust-remote-code \
        --dtype auto \
        --kv-cache-dtype fp8 \
        --enforce-eager \
        --enable-chunked-prefill \
        --mamba-ssm-cache-dtype float16 \
        > "${LOG_DIR}/${name}.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 $((timeout / 2))); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo " UP ($((i*2))s, pid=${pid})"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo " FAILED (process exited)"
            tail -5 "${LOG_DIR}/${name}.log" | sed 's/^/    /'
            return 1
        fi
        sleep 2
    done
    echo " TIMEOUT (${timeout}s)"
    return 1
}

kill_port() {
    local port=$1
    local pid=$(lsof -ti:${port} 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  [kill] :${port} (pid=${pid})"
        kill "$pid" 2>/dev/null || true
    fi
}

kill_all_targets() {
    echo "Killing all target servers..."
    for port in $ALL_PORTS; do
        kill_port "$port"
    done
    sleep 3
}

ensure_guard() {
    echo "Ensuring Qwen3Guard is running on :8001..."
    start_and_wait "qwen3guard" "Qwen/Qwen3Guard-Gen-0.6B" 8001 0.05 900
}

# ---- Commands ----

cmd_batch1() {
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 1: Guard + 0.6B + 1.7B + 4B + 8B + 14B (~75GB) ==="
    for name in qwen3guard qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b; do
        start_and_wait "$name" "${BATCH1_MODELS[$name]}" "${BATCH1_PORTS[$name]}" "${BATCH1_MEM[$name]}" 1800 || true
    done
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_batch2() {
    kill_all_targets
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 2: Guard + 30B-A3B alone (0.90 GPU) ==="
    ensure_guard
    start_and_wait "qwen3-30b-a3b" "${BATCH2_MODELS[qwen3-30b-a3b]}" "${BATCH2_PORTS[qwen3-30b-a3b]}" "${BATCH2_MEM[qwen3-30b-a3b]}" 2400
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_batch3() {
    kill_all_targets
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 3: Guard + 32B alone (0.90 GPU) ==="
    ensure_guard
    start_and_wait "qwen3-32b" "${BATCH3_MODELS[qwen3-32b]}" "${BATCH3_PORTS[qwen3-32b]}" "${BATCH3_MEM[qwen3-32b]}" 2400
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_batch4() {
    kill_all_targets
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 4: Guard + Qwen3-Coder-Next-FP8 (0.95 total) ==="
    ensure_guard
    start_and_wait "qwen3-coder-next" "${BATCH4_MODELS[qwen3-coder-next]}" "${BATCH4_PORTS[qwen3-coder-next]}" "${BATCH4_MEM[qwen3-coder-next]}" 2400
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_batch5() {
    kill_all_targets
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 5: Guard + Nemotron-3-Super-120B-NVFP4 (0.95 total) ==="
    ensure_guard
    start_nemotron "nemotron-120b" "${BATCH5_MODELS[nemotron-120b]}" "${BATCH5_PORTS[nemotron-120b]}" "${BATCH5_MEM[nemotron-120b]}" 3000
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_batch6() {
    kill_all_targets
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Batch 6: Guard + Nemotron-Nano-30B-NVFP4 (0.95 total) ==="
    ensure_guard
    start_nemotron "nemotron-nano-30b" "${BATCH6_MODELS[nemotron-nano-30b]}" "${BATCH6_PORTS[nemotron-nano-30b]}" "${BATCH6_MEM[nemotron-nano-30b]}" 2400
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_status() {
    echo "=== Server Status ==="
    for port in $ALL_PORTS; do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            model=$(curl -sf "http://localhost:${port}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
            echo "  :${port}  UP  ${model}"
        else
            echo "  :${port}  DOWN"
        fi
    done
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_stop() {
    echo "=== Stopping all vLLM servers ==="
    for port in $ALL_PORTS; do
        kill_port "$port"
    done
    sleep 2
    echo "Done."
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_test() {
    echo "=== Quick inference test ==="
    for port in $ALL_PORTS; do
        if ! curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            continue
        fi
        model=$(curl -sf "http://localhost:${port}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        resp=$(curl -sf "http://localhost:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":16,\"temperature\":0}" 2>&1)
        content=$(echo "$resp" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:50])" 2>/dev/null)
        if [ -n "$content" ]; then
            echo "  :${port}  ${model}  ->  ${content}"
        else
            echo "  :${port}  ${model}  ->  FAILED"
        fi
    done
}

# ---- Judge model ----

cmd_judge() {
    cmd_stop
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== DeepSeek-V4-Flash (ASR Judge, port 8010) ==="
    start_and_wait "deepseek-v4-flash" "deepseek-ai/DeepSeek-V4-Flash" 8010 0.99 900 \
        "--kv-cache-dtype fp8 --max-model-len 4096"
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

cmd_quality_judge() {
    cmd_stop
    # (cache-clean removed from auto-start: --enforce-eager writes no compile
    #  cache, so purging it just added startup latency. Use `--clean` explicitly
    #  on a genuinely fresh node.)
    echo "=== Qwen3.5-122B-A10B-FP8 (Quality Judge, port 8000) ==="
    start_and_wait "qwen3.5-122b-judge" "Qwen/Qwen3.5-122B-A10B-FP8" 8000 0.95 900 \
        "--dtype bfloat16 --max-model-len 8192 --max-num-seqs 512"
    echo
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/  GPU: /'
}

# ---- Main ----

case "${1:-batch1}" in
    --batch1|batch1)  cmd_batch1 ;;
    --batch2|batch2)  cmd_batch2 ;;
    --batch3|batch3)  cmd_batch3 ;;
    --batch4|batch4)  cmd_batch4 ;;
    --batch5|batch5)  cmd_batch5 ;;
    --batch6|batch6)  cmd_batch6 ;;
    --judge|judge)    cmd_judge ;;
    --quality-judge|quality-judge)  cmd_quality_judge ;;
    --status|status)  cmd_status ;;
    --stop|stop)      cmd_stop ;;
    --test|test)      cmd_test ;;
    --clean|clean)    clean_cache ;;
    *)
        echo "Usage: $0 [batch1|...|batch6|judge|status|stop|test|clean]"
        echo
        echo "  === Qwen3 pool (all 0.95 total GPU) ==="
        echo "  batch1   Guard + 0.6B + 1.7B + 4B + 8B + 14B"
        echo "  batch2   Guard + 30B-A3B alone"
        echo "  batch3   Guard + 32B alone"
        echo
        echo "  === Cross-family models (all 0.95 total GPU) ==="
        echo "  batch4   Guard + Qwen3-Coder-Next-FP8"
        echo "  batch5   Guard + Nemotron-3-Super-120B-NVFP4"
        echo "  batch6   Guard + Nemotron-Nano-30B-NVFP4"
        echo
        echo "  === Judge models ==="
        echo "  judge    DeepSeek-V4-Flash (ASR judge, port 8010)"
        echo
        echo "  === Utilities ==="
        echo "  status   Check which servers are running"
        echo "  stop     Kill all vLLM servers"
        echo "  test     Quick inference test on running servers"
        echo "  clean    Clean vLLM/torch/triton caches"
        exit 1
        ;;
esac
