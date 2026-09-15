"""Paraphrase (S2): a helper LLM rewrites the query in natural language,
destroying token-level adversarial suffixes but preserving benign semantics.

Jain et al., 2023 (arXiv:2309.00614); PandaGuard ParaphraseDefender.
"""

import os

from .base import BaseStrategy, StrategyResult, make_judge_client, judge_call


PARAPHRASE_PROMPT = 'paraphrase the following paragraph: \n"{prompt}"\n\n'

# PandaBench's standardized helper; smaller ones comply with DAN attacks.
PARAPHRASE_HELPER_MODEL = "meta-llama/llama-3.3-70b-instruct"


class ParaphraseStrategy(BaseStrategy):
    name = "paraphrase"
    reference = "Jain et al., 2023 (arXiv:2309.00614); PandaGuard ParaphraseDefender"

    def __init__(self, client, model,
                 judge_client=None, judge_model=None,
                 paraphrase_max_tokens: int = 2048, **kwargs):
        # PandaGuard's default 512 truncates ~28% of attacks.
        super().__init__(client, model)
        self.paraphrase_max_tokens = paraphrase_max_tokens

        if judge_client is not None:
            self.para_client = judge_client
            self.para_model = judge_model or PARAPHRASE_HELPER_MODEL
        else:
            # With OPENROUTER_API_KEY use the 70B helper, else fall back to the target.
            self.para_client, fallback_model = make_judge_client(client, model)
            using_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
            self.para_model = judge_model or (
                PARAPHRASE_HELPER_MODEL if using_openrouter else fallback_model
            )

    def _paraphrase(self, query: str):
        text, p, c = judge_call(
            self.para_client, self.para_model,
            [{"role": "user", "content": PARAPHRASE_PROMPT.format(prompt=query)}],
            max_tokens=self.paraphrase_max_tokens, temperature=0.0,
        )
        # PandaGuard strips bracket wrappers and a preamble line.
        text = text.strip().strip("]").strip("[").strip()
        if "\n" in text:
            text = "\n".join(text.split("\n")[1:]).strip()
        return text, p, c

    def apply(self, query: str, max_tokens: int = 4096) -> StrategyResult:
        try:
            paraphrased, sp, sc = self._paraphrase(query)
            paraphrase_error = None
        except Exception as e:
            paraphrased, sp, sc = "", 0, 0
            paraphrase_error = str(e)

        # Empty paraphrase / error → use original (PandaGuard behaviour).
        used_query = paraphrased if paraphrased else query

        try:
            response_text, usage = self._generate(used_query, max_tokens)
        except Exception as e:
            return StrategyResult(response="", error=str(e))

        meta = {
            "paraphrased_query": (paraphrased or "")[:200],
            "fell_back_to_original": not bool(paraphrased),
        }
        if paraphrase_error:
            meta["paraphrase_error"] = paraphrase_error[:200]

        return StrategyResult(
            response=response_text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            strategy_tokens=sp + sc,
            strategy_prompt_tokens=sp,
            strategy_completion_tokens=sc,
            strategy_calls=1,
            metadata=meta,
        )
