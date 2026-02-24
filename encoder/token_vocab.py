"""
encoder/token_vocab.py — Complete 50-token intent vocabulary for NeuroWeave Sentinel.

Tokens are organised across 9 clinical categories. Each token carries a
natural_hint phrase that is injected directly into the LLM prompt so the
model understands the patient's intended communication without seeing raw codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────
# IntentToken dataclass
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntentToken:
    """
    A single entry in the NeuroWeave intent vocabulary.

    Attributes:
        id: Unique integer token ID in [0, 49].
        code: Canonical uppercase code string (e.g. ``'SENS_PAIN'``).
        category: Category group string (e.g. ``'SENSATION'``).
        description: Full English description for UI tooltips and logs.
        natural_hint: Short phrase injected into the LLM prompt.
    """

    id: int
    code: str
    category: str
    description: str
    natural_hint: str


# ──────────────────────────────────────────────────────────────
# Vocabulary definition — 50 tokens across 9 categories
# ──────────────────────────────────────────────────────────────
# Columns: id, code, category, description, natural_hint

_RAW: list[tuple[int, str, str, str, str]] = [

    # ── BODY (8) ─────────────────────────────────────────────
    ( 0, "BODY_HEAD",        "BODY",      "Head or neck region",          "head or neck"),
    ( 1, "BODY_CHEST",       "BODY",      "Chest or upper torso",         "chest or upper body"),
    ( 2, "BODY_ABDOMEN",     "BODY",      "Abdomen or stomach area",      "abdomen or stomach"),
    ( 3, "BODY_BACK",        "BODY",      "Back or spine area",           "back or spine"),
    ( 4, "BODY_ARM",         "BODY",      "Arm, hand, or shoulder",       "arm, hand, or shoulder"),
    ( 5, "BODY_LEG",         "BODY",      "Leg, foot, or hip",            "leg, foot, or hip"),
    ( 6, "BODY_WHOLE",       "BODY",      "Whole body or general",        "whole body or general"),
    ( 7, "BODY_UNSPECIFIED", "BODY",      "Body area not specified",       "unspecified body area"),

    # ── SENSATION (6) ────────────────────────────────────────
    ( 8, "SENS_PAIN",        "SENSATION", "Pain or hurting",              "pain or hurting"),
    ( 9, "SENS_PRESSURE",    "SENSATION", "Pressure, tightness or weight","pressure or tightness"),
    (10, "SENS_ITCH",        "SENSATION", "Itching or irritation",        "itching or irritation"),
    (11, "SENS_NAUSEA",      "SENSATION", "Nausea or feeling sick",       "nausea or feeling sick"),
    (12, "SENS_DISCOMFORT",  "SENSATION", "General discomfort or unease", "discomfort or unease"),
    (13, "SENS_COMFORT",     "SENSATION", "Feeling comfortable or better","comfortable or feeling better"),

    # ── INTENSITY (4) ────────────────────────────────────────
    (14, "INT_MILD",         "INTENSITY", "Mild — barely noticeable",     "mild"),
    (15, "INT_MODERATE",     "INTENSITY", "Moderate — noticeably present","moderate"),
    (16, "INT_SEVERE",       "INTENSITY", "Severe — strongly felt",       "severe"),
    (17, "INT_CRITICAL",     "INTENSITY", "Critical — unbearable",        "critical or unbearable"),

    # ── TEMPORAL (4) ─────────────────────────────────────────
    (18, "TIME_NOW",         "TEMPORAL",  "Happening right now",          "right now"),
    (19, "TIME_RECENT",      "TEMPORAL",  "Started recently",             "just started or recent"),
    (20, "TIME_ONGOING",     "TEMPORAL",  "Ongoing or persistent",        "ongoing or persistent"),
    (21, "TIME_WORSENING",   "TEMPORAL",  "Getting worse over time",      "getting worse"),

    # ── NEEDS (8) ────────────────────────────────────────────
    (22, "NEED_WATER",       "NEEDS",     "Needs water or is thirsty",    "thirsty, needs water"),
    (23, "NEED_FOOD",        "NEEDS",     "Hungry or needs food",         "hungry, needs food"),
    (24, "NEED_TOILET",      "NEEDS",     "Needs toilet or bathroom",     "needs toilet or bathroom"),
    (25, "NEED_REPOSITION",  "NEEDS",     "Needs repositioning in bed",   "needs to be repositioned"),
    (26, "NEED_MEDICATION",  "NEEDS",     "Needs medication or pain relief","needs medication"),
    (27, "NEED_FAMILY",      "NEEDS",     "Wants family or loved ones",   "wants family nearby"),
    (28, "NEED_REST",        "NEEDS",     "Needs rest or sleep",          "needs rest or sleep"),
    (29, "NEED_NURSE",       "NEEDS",     "Needs the nurse or carer",     "needs nurse or carer"),

    # ── COGNITIVE (6) ────────────────────────────────────────
    (30, "COG_YES",          "COGNITIVE", "Yes / agree / correct",        "yes"),
    (31, "COG_NO",           "COGNITIVE", "No / disagree / incorrect",    "no"),
    (32, "COG_UNSURE",       "COGNITIVE", "Unsure or uncertain",          "unsure"),
    (33, "COG_REPEAT",       "COGNITIVE", "Please repeat that",           "please repeat"),
    (34, "COG_UNDERSTAND",   "COGNITIVE", "I understand",                 "I understand"),
    (35, "COG_CONFUSED",     "COGNITIVE", "I am confused or lost",        "confused"),

    # ── EMOTIONAL (6) ────────────────────────────────────────
    (36, "EMO_ANXIOUS",      "EMOTIONAL", "Feeling anxious or worried",   "anxious or worried"),
    (37, "EMO_CALM",         "EMOTIONAL", "Feeling calm",                 "calm"),
    (38, "EMO_DISTRESSED",   "EMOTIONAL", "Feeling distressed or upset",  "distressed or upset"),
    (39, "EMO_OKAY",         "EMOTIONAL", "Feeling okay overall",         "feeling okay"),
    (40, "EMO_HAPPY",        "EMOTIONAL", "Feeling happy or positive",    "happy"),
    (41, "EMO_SAD",          "EMOTIONAL", "Feeling sad or low",           "sad or low"),

    # ── EMERGENCY (4) ────────────────────────────────────────
    (42, "EMRG_PAIN_SEVERE", "EMERGENCY", "Severe unexpected pain",       "severe unexpected pain"),
    (43, "EMRG_BREATHING",   "EMERGENCY", "Breathing difficulty",         "difficulty breathing"),
    (44, "EMRG_CALL_NURSE",  "EMERGENCY", "Call nurse immediately",       "call nurse immediately"),
    (45, "EMRG_URGENT",      "EMERGENCY", "Something urgent is happening","something urgent"),

    # ── MODIFIERS (4) ────────────────────────────────────────
    (46, "MOD_MORE",         "MODIFIERS", "More / increase / stronger",   "more"),
    (47, "MOD_LESS",         "MODIFIERS", "Less / decrease / weaker",     "less"),
    (48, "MOD_SAME",         "MODIFIERS", "Same as before / unchanged",   "same as before"),
    (49, "MOD_STOP",         "MODIFIERS", "Stop / end / no more",         "stop"),
]

assert len(_RAW) == 50, f"Vocabulary size is {len(_RAW)}, expected 50"

# ──────────────────────────────────────────────────────────────
# Build the VOCAB dict and lookup index
# ──────────────────────────────────────────────────────────────

VOCAB: dict[str, IntentToken] = {
    row[1]: IntentToken(id=row[0], code=row[1], category=row[2],
                        description=row[3], natural_hint=row[4])
    for row in _RAW
}

_BY_ID: dict[int, IntentToken] = {t.id: t for t in VOCAB.values()}
_BY_CATEGORY: dict[str, list[IntentToken]] = {}
for _t in VOCAB.values():
    _BY_CATEGORY.setdefault(_t.category, []).append(_t)

# Categories whose tokens alone do not constitute a meaningful utterance
_MODIFIER_CATEGORIES: frozenset[str] = frozenset({"MODIFIERS", "COGNITIVE"})


# ──────────────────────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────────────────────

def get_by_id(token_id: int) -> Optional[IntentToken]:
    """
    Return the :class:`IntentToken` with the given integer ID.

    Args:
        token_id: Integer ID in [0, 49].

    Returns:
        The matching token, or ``None`` if not found.
    """
    return _BY_ID.get(token_id)


def get_by_category(category: str) -> list[IntentToken]:
    """
    Return all tokens belonging to the given category.

    Args:
        category: Category string (e.g. ``'SENSATION'``, ``'NEEDS'``).
                  Case-sensitive.

    Returns:
        List of :class:`IntentToken` objects; empty list if category unknown.
    """
    return list(_BY_CATEGORY.get(category, []))


def tokens_to_prompt_string(tokens: list[IntentToken]) -> str:
    """
    Render a list of tokens as a comma-separated natural-language hint string
    suitable for injection into the MedGemma prompt.

    Example::

        tokens_to_prompt_string([VOCAB['BODY_CHEST'], VOCAB['SENS_PAIN'], VOCAB['INT_SEVERE']])
        # → 'chest or upper body, pain or hurting, severe'

    Args:
        tokens: Ordered list of :class:`IntentToken` objects.

    Returns:
        Comma-and-space-joined ``natural_hint`` values.
        Returns an empty string if ``tokens`` is empty.
    """
    return ", ".join(t.natural_hint for t in tokens)


def validate_sequence(tokens: list[IntentToken]) -> bool:
    """
    Check that a token sequence constitutes a meaningful utterance.

    A sequence is considered valid if it contains at least one token whose
    category is **not** in ``{MODIFIERS, COGNITIVE}``. Pure modifier or
    cognitive-only selections (e.g. just ``COG_YES`` or ``MOD_MORE``) are
    ambiguous without a substantive token to modify.

    Args:
        tokens: Token list to validate.

    Returns:
        ``True`` if at least one substantive (non-modifier, non-cognitive)
        token is present; ``False`` otherwise.
    """
    return any(t.category not in _MODIFIER_CATEGORIES for t in tokens)
