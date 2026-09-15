"""SelfDefense (S4): self-examine the RESPONSE and replace it with REJECT_RESPONSE
if harmful. The decision is a HARMFULNESS judgment, NOT a PAIR jailbreak score,
which over-refuses good answers to benign queries.

Phute et al., 2023 (arXiv:2308.07308).
"""

import os

from .base import BaseStrategy, StrategyResult, make_judge_client, judge_call

# Importable as a package or via scripts that put the code dir on sys.path.
try:
    from ..jailbreak_judge import harm_messages, parse_harmful, INLINE_JUDGE_MODEL
except ImportError:
    from judges.jailbreak_judge import harm_messages, parse_harmful, INLINE_JUDGE_MODEL


REJECT_RESPONSE = "Sorry, I CANNOT assist with that."


class SelfDefenseStrategy(BaseStrategy):
    """Generate → harmfulness self-examination → block if the response is harmful."""

    name = "selfdefense"
    reference = "Phute et al., 2023 (arXiv:2308.07308) LLM Self Defense"

    def __init__(self, client, model,
                 judge_client=None, judge_model=None, **kwargs):
        super().__init__(client, model)
        if judge_client is not None:
            self.judge_client = judge_client
            self.judge_model = judge_model or INLINE_JUDGE_MODEL
        else:
            self.judge_client, fallback_model = make_judge_client(client, model)
            using_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
            self.judge_model = judge_model or (
                INLINE_JUDGE_MODEL if using_openrouter else fallback_model
            )

    def _judge(self, response: str):
        """Return (harmful: bool, raw_text, prompt_tokens, completion_tokens)."""
        text, p, c = judge_call(
            self.judge_client, self.judge_model, harm_messages(response),
            max_tokens=8, temperature=0.0,
        )
        return parse_harmful(text), text, p, c

    def apply(self, query: str, max_tokens: int = 4096) -> StrategyResult:
        try:
            response_text, usage = self._generate(query, max_tokens)
            harmful, judgement, jp, jc = self._judge(response_text)
        except Exception as e:
            return StrategyResult(response="", error=str(e))

        result_kwargs = dict(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            strategy_tokens=jp + jc,
            strategy_prompt_tokens=jp,
            strategy_completion_tokens=jc,
            strategy_calls=1,
            metadata={"judge_harmful": harmful, "judge_text": judgement[:200]},
        )

        if harmful:
            return StrategyResult(
                response=REJECT_RESPONSE,
                blocked=True,
                original_response=response_text,
                **result_kwargs,
            )
        return StrategyResult(response=response_text, **result_kwargs)


SelfExamineStrategy = SelfDefenseStrategy
