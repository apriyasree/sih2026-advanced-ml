"""General Indian legal-support routing. This is information, not legal advice."""

from typing import Any, Dict, List


LEGAL_KEYWORDS = {
    "sexual_violence_support": ("sexual", "assault", "molest", "rape", "harass", "stalk"),
    "caste_or_tribal_harm_support": ("caste", "dalit", "tribal", "adivasi", "atrocity", "social boycott"),
    "threat_or_intimidation_support": ("threat", "threatened", "weapon", "following", "stalking", "attack", "unsafe"),
    "general_victim_rights": ("fir", "police", "complaint", "case", "court", "lawyer", "legal aid", "compensation"),
}

BASE_RESOURCES: List[Dict[str, str]] = [
    {"name": "Emergency response", "number": "112", "when": "Immediate danger or urgent police, fire or medical assistance."},
    {"name": "Tele-MANAS mental-health support", "number": "14416 / 1800-89-14416", "when": "24-hour emotional-support referral in India."},
    {"name": "NALSA legal-aid helpline", "number": "15100", "when": "Free legal-aid information and connection to the Legal Services Authorities."},
]


def legal_guidance_for(text: str, safety_flags: List[str], support_score: float) -> Dict[str, Any]:
    lowered = (text or "").casefold()
    categories = [category for category, keywords in LEGAL_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]
    if safety_flags and "threat_or_intimidation_support" not in categories:
        categories.append("threat_or_intimidation_support")

    steps: List[str] = []
    if "immediate_physical_danger" in safety_flags or support_score >= 75:
        steps.append("If there is immediate danger, use 112 or move to a safer place if doing so is safe. This tool cannot contact services for you.")
    if categories:
        steps.append("You may ask the District Legal Services Authority (DLSA) for free legal aid and information about victim-support and compensation processes.")
        steps.append("If you choose to report, a lawyer or DLSA can explain complaint/FIR options, including whether a Zero FIR may be appropriate; procedures vary by facts and location.")
    if "sexual_violence_support" in categories:
        steps.append("For sexual-violence concerns, consider a trusted support person and local one-stop or women-support services. Preserve only information you feel safe preserving; do not put yourself at further risk.")
    if "caste_or_tribal_harm_support" in categories:
        steps.append("For caste- or tribal-targeted harm, a legal-aid provider can explain protections and local support routes relevant to the facts.")

    return {
        "categories": categories,
        "resources": BASE_RESOURCES if categories or safety_flags or support_score >= 55 else [],
        "next_steps": steps,
        "disclaimer": "General information only, not legal advice. A qualified lawyer or legal-aid authority should review the facts.",
    }
