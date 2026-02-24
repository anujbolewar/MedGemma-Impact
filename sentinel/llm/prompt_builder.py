"""
sentinel/llm/prompt_builder.py — Structured prompt construction for MedGemma intent reconstruction.

Builds safety-constrained prompts that instruct MedGemma to reconstruct exactly
one expressive sentence from intent tokens. No diagnosis or medical advice is
ever requested from the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, field_validator

from sentinel.intent.classifier import IntentBundle

logger = logging.getLogger(__name__)

# Hard-coded safety instruction that is always appended — never removed
_SAFETY_SUFFIX: str = (
    "IMPORTANT: Output only the single sentence the patient is expressing. "
    "Do not provide any diagnosis, medical advice, or treatment recommendations. "
    "Do not add explanation or context. Output only the sentence."
)


@dataclass(frozen=True)
class BuiltPrompt:
    """
    The output of the PromptBuilder.

    Attributes:
        system_prompt: System-role instruction for the model.
        user_prompt: The formatted user-turn message with intent tokens.
        full_text: Combined prompt string for models that accept a single input.
        token_count_estimate: Rough token estimate (characters / 4).
    """

    system_prompt: str
    user_prompt: str
    full_text: str
    token_count_estimate: int


class PromptTokens(BaseModel):
    """
    Pydantic-validated input tokens for the prompt builder.

    Ensures that all required fields are non-empty strings before
    the prompt is constructed, preventing malformed LLM inputs.
    """

    body_part: str
    sensation: str
    urgency: str
    intensity: str

    @field_validator("body_part", "sensation", "urgency", "intensity")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        """
        Validate that a token field is a non-empty string.

        Args:
            v: Field value to validate.

        Returns:
            The validated string.

        Raises:
            ValueError: If the string is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Intent token must not be empty")
        return v.strip().lower()


class PromptBuilder:
    """
    Constructs clinically-safe MedGemma prompts from intent token bundles.

    The prompt template is deliberately minimal and constrained to prevent
    the model from producing diagnostic content. It uses the instruct-format
    chat template expected by medgemma-4b-it.

    Example output sentence given these tokens:
        BODY_PART=chest, SENSATION=pressure, URGENCY=right now, INTENSITY=moderate
        → "I have moderate pressure in my chest right now."
    """

    _SYSTEM_PROMPT: str = (
        "You are an assistive communication system helping a patient with ALS "
        "or locked-in syndrome express themselves. Your only task is to reconstruct "
        "a single, natural, first-person sentence based on the patient's intent tokens. "
        "Never diagnose. Never recommend treatment. Output exactly one sentence."
    )

    _USER_TEMPLATE: str = (
        "My intent tokens are:\n"
        "- BODY_PART: {body_part}\n"
        "- SENSATION: {sensation}\n"
        "- URGENCY: {urgency}\n"
        "- INTENSITY: {intensity}\n\n"
        "Please reconstruct the single sentence I am trying to express.\n\n"
        "{safety_suffix}"
    )

    def build(self, bundle: IntentBundle) -> str:
        """
        Build a complete prompt string from an :class:`IntentBundle`.

        Args:
            bundle: A complete or partial intent bundle. Values are validated
                    and non-empty; missing values fall back to "unknown".

        Returns:
            A formatted prompt ready for MedGemma tokenisation.

        Raises:
            ValueError: If any intent token value fails Pydantic validation.
        """
        raw_tokens = bundle.to_prompt_tokens()
        tokens = PromptTokens(**raw_tokens)

        user_content = self._USER_TEMPLATE.format(
            body_part=tokens.body_part,
            sensation=tokens.sensation,
            urgency=tokens.urgency,
            intensity=tokens.intensity,
            safety_suffix=_SAFETY_SUFFIX,
        )

        logger.debug("PromptBuilder: user content (%d chars)", len(user_content))
        return user_content

    def build_chat_messages(self, bundle: IntentBundle) -> list[dict[str, str]]:
        """
        Build the prompt as a list of chat messages for transformers.

        This is the preferred format for models loaded with AutoModelForCausalLM
        and a chat template (medgemma-4b-it uses Gemma chat format).

        Args:
            bundle: Complete intent bundle.

        Returns:
            List of role/content dicts: [{'role': 'system', ...}, {'role': 'user', ...}]
        """
        user_content = self.build(bundle)
        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def estimate_input_tokens(self, bundle: IntentBundle) -> int:
        """
        Estimate the number of input tokens for budget planning.

        Uses a character / 4 heuristic (good approximation for English text
        with SentencePiece tokenisers like Gemma).

        Args:
            bundle: The intent bundle to estimate for.

        Returns:
            Estimated integer token count.
        """
        prompt = self.build(bundle)
        return max(1, len(self._SYSTEM_PROMPT + prompt) // 4)
