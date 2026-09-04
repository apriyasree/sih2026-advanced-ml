"""Immediate-safety language detection for a support tool, not diagnosis."""

import re
from typing import Dict, List


SAFETY_PATTERNS = {
    "self_harm_imminent": [
        "kill myself", "end my life", "want to die", "hurt myself",
        "harm myself", "have a plan to die", "cannot keep myself safe",
        "i want to commit suicide", "i am going to commit suicide",
    ],
    "immediate_physical_danger": [
        "they are here", "someone is attacking me", "being attacked", "he has a weapon",
        "she has a weapon", "i am in danger now", "locked me in", "cannot get out",
        "threatening to kill me", "following me right now",
    ],
}


def assess_immediate_safety(text: str) -> Dict[str, List[str]]:
    text_lower = (text or "").casefold()

    def contains_phrase(phrase: str) -> bool:
        # Word boundaries prevent substring mistakes such as matching a keyword
        # inside an unrelated word. This remains a transparent, conservative
        # language flag and is never a determination of intent.
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        return re.search(pattern, text_lower) is not None

    matches = {
        category: [phrase for phrase in phrases if contains_phrase(phrase)]
        for category, phrases in SAFETY_PATTERNS.items()
    }
    matches = {category: phrases for category, phrases in matches.items() if phrases}
    return {"flags": list(matches.keys()), "matched_phrases": matches}
