"""Qwen3Guard (S3): block when the guard reports Safety=Unsafe or Controversial,
else generate normally. The ~0.6B classifier is served via vLLM at an
OpenAI-compatible endpoint (default :8001, auto-started).
https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
"""

import fcntl
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
import openai

from .base import BaseStrategy, StrategyResult, judge_call


def _resolve_vllm_paths() -> Tuple[str, str]:
    """Resolve (vllm_bin, vllm_lib) with this priority:
       1. `QWEN3GUARD_VLLM_BIN` / `QWEN3GUARD_VLLM_LIB` env vars
       2. `shutil.which("vllm")` for the bin + sibling `lib/`
       3. Hardcoded fallback (may need updating per environment)
    """
    vllm_bin = os.environ.get("QWEN3GUARD_VLLM_BIN")
    vllm_lib = os.environ.get("QWEN3GUARD_VLLM_LIB")

    if vllm_bin is None:
        which = shutil.which("vllm")
        if which:
            vllm_bin = which
            if vllm_lib is None:
                # vllm bin lives in <env>/bin/vllm; the env's lib is at <env>/lib
                vllm_lib = str(Path(which).resolve().parent.parent / "lib")
    if vllm_bin is None:
        vllm_bin = shutil.which("vllm") or "vllm"
    if vllm_lib is None:
        vllm_lib = ""
    return vllm_bin, vllm_lib


DEFAULT_GUARD_BASE_URL = os.environ.get("QWEN3GUARD_BASE_URL", "http://localhost:8001/v1")
DEFAULT_GUARD_MODEL    = os.environ.get("QWEN3GUARD_MODEL",    "Qwen/Qwen3Guard-Gen-0.6B")
DEFAULT_GUARD_GPU      = os.environ.get("QWEN3GUARD_GPU",      "0")
DEFAULT_VLLM_BIN, DEFAULT_VLLM_LIB = _resolve_vllm_paths()
LOCK_PATH = Path("/tmp/qwen3guard_autostart.lock")
LOG_PATH  = Path(os.environ.get("QWEN3GUARD_LOG", "/tmp/qwen3guard_server.log"))


def _server_ready(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(base_url.rstrip("/") + "/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_ready(base_url: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_ready(base_url):
            return True
        time.sleep(2)
    return False


def _start_guard_server(model: str, port: int, gpu: str) -> subprocess.Popen:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(LOG_PATH, "ab")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["LD_LIBRARY_PATH"] = DEFAULT_VLLM_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        DEFAULT_VLLM_BIN, "serve", model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.30",
        "--max-model-len", "8192",
        "--host", "0.0.0.0",
        "--trust-remote-code",
        "--served-model-name", model,
    ]
    return subprocess.Popen(
        cmd, stdout=fout, stderr=fout, env=env,
        start_new_session=True,
    )


def ensure_guard_running(base_url: str = DEFAULT_GUARD_BASE_URL,
                         model: str    = DEFAULT_GUARD_MODEL,
                         gpu: str      = DEFAULT_GUARD_GPU,
                         autostart: bool = True,
                         startup_timeout: float = 300.0) -> bool:
    """Ensure a Qwen3Guard server is up. Idempotent + multi-process-safe via flock."""
    if _server_ready(base_url):
        return True
    if not autostart:
        return False

    parsed = urlparse(base_url)
    port = parsed.port or 8001

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if _server_ready(base_url):
            return True
        print(f"[Qwen3Guard] starting vLLM on GPU {gpu} port {port} "
              f"({model}) — log: {LOG_PATH}", flush=True)
        proc = _start_guard_server(model, port, gpu)
        ok = _wait_for_ready(base_url, timeout=startup_timeout)
        if ok:
            print(f"[Qwen3Guard] server ready at {base_url} (pid={proc.pid})", flush=True)
        else:
            print(f"[Qwen3Guard] server failed to come up within "
                  f"{startup_timeout}s — see {LOG_PATH}", flush=True)
        return ok


def _parse_guard_output(text: str) -> dict:
    out = {"safety": None, "categories": [], "refusal": None, "raw": text.strip()}
    if not text:
        return out

    m_safety = re.search(r"Safety\s*:[ \t]*(\w+)", text, re.IGNORECASE)
    if m_safety:
        out["safety"] = m_safety.group(1).strip().lower()

    m_cats = re.search(r"Categories?\s*:[ \t]*([^\n]+)", text, re.IGNORECASE)
    if m_cats:
        cats = m_cats.group(1).strip()
        out["categories"] = [c.strip().lower() for c in re.split(r"[,;]", cats) if c.strip()]

    m_ref = re.search(r"Refusal\s*:[ \t]*(\w+)", text, re.IGNORECASE)
    if m_ref:
        out["refusal"] = m_ref.group(1).strip().lower()

    return out


class Qwen3GuardStrategy(BaseStrategy):
    """Qwen3Guard input classifier: block when Safety=Unsafe or Controversial."""

    name = "qwen3guard"
    reference = "Qwen3Guard-Gen-0.6B (Alibaba 2025)"

    def __init__(self, client: openai.OpenAI, model: str,
                 guard_base_url: Optional[str] = None,
                 guard_api_key: Optional[str] = None,
                 guard_model: Optional[str] = None,
                 guard_gpu: Optional[str] = None,
                 autostart_guard: bool = True,
                 startup_timeout: float = 300.0,
                 **kwargs):
        super().__init__(client, model)
        base_url = guard_base_url or DEFAULT_GUARD_BASE_URL
        guard_model = guard_model or DEFAULT_GUARD_MODEL
        gpu = guard_gpu or DEFAULT_GUARD_GPU

        autostart = autostart_guard and (
            os.environ.get("QWEN3GUARD_AUTOSTART", "1") != "0"
        )
        if not ensure_guard_running(base_url=base_url, model=guard_model,
                                    gpu=gpu, autostart=autostart,
                                    startup_timeout=startup_timeout):
            raise RuntimeError(
                f"Qwen3Guard server not available at {base_url} and "
                "auto-start failed (or was disabled). Set QWEN3GUARD_BASE_URL "
                "or run `vllm serve Qwen/Qwen3Guard-Gen-0.6B --port 8001` manually."
            )

        self.guard_client = openai.OpenAI(
            base_url=base_url,
            api_key=guard_api_key or os.environ.get("QWEN3GUARD_API_KEY") or "EMPTY",
        )
        self.guard_model = guard_model

    def _moderate_prompt(self, query: str) -> Tuple[dict, int, int]:
        # No truncation: a low cap hides the goal at the END of long prompts.
        messages = [{"role": "user", "content": query[:20000]}]
        text, p, c = judge_call(self.guard_client, self.guard_model, messages,
                                max_tokens=64, temperature=0.0)
        return _parse_guard_output(text), p, c

    def apply(self, query: str, max_tokens: int = 4096) -> StrategyResult:
        try:
            prompt_verdict, p, c = self._moderate_prompt(query)

            # Qwen3Guard-Gen emits Safe/Controversial/Unsafe; S3 blocks the latter two.
            if (prompt_verdict.get("safety") or "").lower() in ("unsafe", "controversial"):
                return StrategyResult(
                    response="I cannot assist with this request.",
                    blocked=True,
                    strategy_tokens=p + c,
                    strategy_prompt_tokens=p,
                    strategy_completion_tokens=c,
                    strategy_calls=1,
                    metadata={
                        "blocked_at": "input",
                        "prompt_verdict": prompt_verdict["raw"][:200],
                    },
                )

            response_text, usage = self._generate(query, max_tokens)
            return StrategyResult(
                response=response_text,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                strategy_tokens=p + c,
                strategy_prompt_tokens=p,
                strategy_completion_tokens=c,
                strategy_calls=1,
                metadata={"prompt_verdict": prompt_verdict["raw"][:200]},
            )

        except Exception as e:
            return StrategyResult(response="", error=str(e))
