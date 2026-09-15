"""The 7 anchored defenses; the action space is the 4x4 factored policy
pre {s0,s1,s2,s3} x post {s0,s4,s5,s6} = 16 composites, resolved by get_strategy().

    s0 baseline safety prompt        s4 SelfDefense (harmfulness judge)   post
    s1 SafetyContextRetrieval  pre   s5 Backtranslation                   post
    s2 Paraphrase              pre   s6 PARDEN (repeat-and-BLEU)          post
    s3 Qwen3Guard              pre

These are REFERENCE implementations: the reported numbers come from the inline
re-implementations in probe/, which import only SCR and shared prompts from here.
"""

from .base import (
    BaseStrategy,
    StrategyResult,
    SAFETY_SYSTEM_PROMPT,
    DEFAULT_JUDGE_MODEL,
)
from .safety_context_retrieval import SafetyContextRetrievalStrategy
from .paraphrase import ParaphraseStrategy
from .qwen3guard import Qwen3GuardStrategy
from .self_defense import SelfDefenseStrategy, REJECT_RESPONSE
from .backtranslation import BacktranslationStrategy
from .parden import PardenStrategy


# Canonical 7-anchor taxonomy.
ANCHOR_REGISTRY = {
    "s0": BaseStrategy,                     # None
    "s1": SafetyContextRetrievalStrategy,   # Prompt Augmentation
    "s2": ParaphraseStrategy,               # Input Transformation
    "s3": Qwen3GuardStrategy,               # Input Classification
    "s4": SelfDefenseStrategy,              # Output Verification
    "s5": BacktranslationStrategy,          # Intent Backtranslation
    "s6": PardenStrategy,                   # Output Verification via Repetition (PARDEN)
}

PRE_GEN_ANCHORS = ("s0", "s1", "s2", "s3")
POST_GEN_ANCHORS = ("s0", "s4", "s5", "s6")

# The action space is exactly the 7 anchors.
STRATEGY_REGISTRY = dict(ANCHOR_REGISTRY)

# Human-readable aliases → canonical keys.
NAME_ALIASES = {
    "none": "s0",
    "standard": "s0",
    "safety_context_retrieval": "s1",
    "scr": "s1",
    "paraphrase": "s2",
    "qwen3guard": "s3",
    "selfdefense": "s4",
    "self_defense": "s4",
    "self_examine": "s4",   # back-compat for callers of the old class name
    "backtranslation": "s5",
    "parden": "s6",
    "repeat": "s6",
}


def get_strategy(key: str):
    """Resolve a strategy key (anchor ID or alias) to a class."""
    resolved = NAME_ALIASES.get(key, key)
    if resolved not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{key}'. "
            f"Anchors: {list(ANCHOR_REGISTRY)}; "
            f"available: {list(STRATEGY_REGISTRY)}; "
            f"aliases: {list(NAME_ALIASES)}."
        )
    return STRATEGY_REGISTRY[resolved]


__all__ = [
    "BaseStrategy",
    "StrategyResult",
    "SAFETY_SYSTEM_PROMPT",
    "DEFAULT_JUDGE_MODEL",
    "REJECT_RESPONSE",
    "SafetyContextRetrievalStrategy",
    "ParaphraseStrategy",
    "Qwen3GuardStrategy",
    "SelfDefenseStrategy",
    "BacktranslationStrategy",
    "PardenStrategy",
    "ANCHOR_REGISTRY",
    "STRATEGY_REGISTRY",
    "NAME_ALIASES",
    "PRE_GEN_ANCHORS",
    "POST_GEN_ANCHORS",
    "get_strategy",
]
