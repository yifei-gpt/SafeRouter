#!/usr/bin/env python3
"""Probe drivers.

    python -m probe.adversarial      # attacks x models x 16 composites
    python -m probe.benign           # raw S0 benign responses
    python -m probe.benign_defense   # benign under every defense S0-S6
"""
LOCAL_TARGET_PORTS = {
    "Qwen/Qwen3-0.6B":     8002,
    "Qwen/Qwen3-1.7B":     8003,
    "Qwen/Qwen3-4B":       8004,
    "Qwen/Qwen3-8B":       8005,
    "Qwen/Qwen3-14B":      8006,
    "Qwen/Qwen3-30B-A3B":  8007,
    "Qwen/Qwen3-32B":      8008,
    "Qwen/Qwen3-Coder-Next-FP8":                            8013,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4":       8014,
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4":  8015,
}
QWEN3GUARD_PORT = 8001
