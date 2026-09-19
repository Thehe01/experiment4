"""Document-level context that is independent of entity-type decisions.

This module only recovers abbreviation definitions that are explicit in the
source document (``Long Form (SF)`` or ``SF (Long Form)``).  It deliberately
does not decide whether either side is a Configuration, and it does not add
entities to model output.  The resulting lexical context is fixed for P0 and
all ProTeGi candidates in the same run.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


DOCUMENT_CONTEXT_VERSION = "explicit-abbreviation-context-v1"
NO_EXPLICIT_ABBREVIATIONS = "(none detected)"

_SHORT_FORM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/-]{1,14}$")
_LONG_BEFORE_SHORT_RE = re.compile(
    r"(?P<long>[A-Za-z0-9][^()\n]{2,160}?)\s*"
    r"\((?P<short>[A-Za-z][A-Za-z0-9._/-]{1,14})\)"
)
_SHORT_BEFORE_LONG_RE = re.compile(
    r"\b(?P<short>[A-Za-z][A-Za-z0-9._/-]{1,14})\s*"
    r"\((?P<long>[^()\n]{3,160})\)"
)


def _looks_like_short_form(value: str) -> bool:
    value = value.strip()
    if not _SHORT_FORM_RE.fullmatch(value):
        return False
    alnum = [char for char in value if char.isalnum()]
    if len(alnum) < 2:
        return False
    letters = [char for char in value if char.isalpha()]
    uppercase = [char for char in letters if char.isupper()]
    return len(uppercase) >= 2 or (uppercase and any(char.isdigit() for char in value))


def _candidate_long_form_tail(value: str, short_form: str) -> str:
    """Limit a regex candidate to the plausible words immediately before ``(SF)``."""
    value = re.split(r"[.!?;:\n]", value)[-1].strip(" \t,;:-")
    words = value.split()
    max_words = min(len(short_form) + 5, max(2 * len(short_form), 4), 12)
    return " ".join(words[-max_words:])


def _align_long_form(short_form: str, long_candidate: str) -> Optional[str]:
    """Return the shortest aligned suffix using the Schwartz--Hearst scan."""
    short_index = len(short_form) - 1
    long_index = len(long_candidate) - 1

    while short_index >= 0:
        short_char = short_form[short_index]
        if not short_char.isalnum():
            short_index -= 1
            continue
        target = short_char.casefold()
        while long_index >= 0:
            char_matches = long_candidate[long_index].casefold() == target
            first_char_at_word_start = (
                short_index != 0
                or long_index == 0
                or not long_candidate[long_index - 1].isalnum()
            )
            if char_matches and first_char_at_word_start:
                break
            long_index -= 1
        if long_index < 0:
            return None
        long_index -= 1
        short_index -= 1

    aligned = long_candidate[long_index + 1 :].strip(" \t,;:-")
    if len(aligned.split()) < 2 or len(aligned) <= len(short_form):
        return None
    return aligned


def extract_explicit_abbreviation_pairs(text: str) -> List[Dict[str, object]]:
    """Recover explicit abbreviation definitions without assigning entity types."""
    pairs: List[Dict[str, object]] = []
    seen = set()

    for match in _LONG_BEFORE_SHORT_RE.finditer(text):
        short_form = match.group("short").strip()
        if not _looks_like_short_form(short_form):
            continue
        candidate = _candidate_long_form_tail(match.group("long"), short_form)
        long_form = _align_long_form(short_form, candidate)
        if not long_form:
            continue
        long_start = match.start("long") + match.group("long").rfind(long_form)
        key = (short_form.casefold(), long_form.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "short_form": short_form,
            "long_form": long_form,
            "definition_start": long_start,
            "definition_end": match.end(),
            "pattern": "long_form_then_short_form",
        })

    for match in _SHORT_BEFORE_LONG_RE.finditer(text):
        short_form = match.group("short").strip()
        long_form = match.group("long").strip(" \t,;:-")
        if not _looks_like_short_form(short_form):
            continue
        aligned = _align_long_form(short_form, long_form)
        if not aligned or aligned.casefold() != long_form.casefold():
            continue
        key = (short_form.casefold(), long_form.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "short_form": short_form,
            "long_form": long_form,
            "definition_start": match.start(),
            "definition_end": match.end(),
            "pattern": "short_form_then_long_form",
        })

    return sorted(
        pairs,
        key=lambda item: (
            int(item["definition_start"]),
            str(item["short_form"]).casefold(),
        ),
    )


def format_abbreviation_context(
    pairs: List[Dict[str, object]],
    *,
    max_pairs: int = 24,
) -> str:
    """Serialize explicit pairs deterministically for the fixed runtime preamble."""
    if not pairs:
        return NO_EXPLICIT_ABBREVIATIONS
    return "\n".join(
        f"- {item['short_form']} = {item['long_form']}"
        for item in pairs[:max_pairs]
    )


def render_document_context(context: str) -> str:
    """Render the non-optimizable context contract shared by P0 and candidates."""
    return (
        f"<DOCUMENT_CONTEXT version=\"{DOCUMENT_CONTEXT_VERSION}\">\n"
        "The following pairs were recovered mechanically only from explicit "
        "abbreviation-definition patterns in the full source document. They "
        "provide lexical equivalence only: a listed form is not automatically "
        "a Configuration or any other entity. For every occurrence, apply the "
        "frozen entity definition and exact source-span rules to its local context.\n"
        f"{context or NO_EXPLICIT_ABBREVIATIONS}\n"
        "</DOCUMENT_CONTEXT>"
    )
