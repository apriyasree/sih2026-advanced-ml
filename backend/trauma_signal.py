"""
Explainable, non-diagnostic trauma-language screening for AMMA.
Grounded in the TRACE Trauma Language Corpus (EMNLP Findings 2024 / MiriamSchirmer/trauma-language).

Maps conversational language to the 4 DSM-5 trauma symptom clusters:
1. Intrusion: Vivid re-experiencing, flashbacks, trauma nightmares, somatic cue reactivity.
2. Avoidance: Cognitive thought suppression, behavioral avoidance of places/reminders, social isolation.
3. Hyperarousal & Reactivity: Hypervigilance, exaggerated startle, insomnia, irritability.
4. Negative Cognition & Mood: Self-blame, survivor guilt, emotional numbing, alienated worldview.

Strictly non-diagnostic: Identifies transparent linguistic signals to provide context
for a human counsellor. It never generates diagnostic labels (e.g., PTSD).
"""

import re
from typing import Any, Dict, List, Tuple

# TRACE EMNLP 2024 Taxonomy Mappings
TRACE_SYMPTOM_CLUSTERS: Dict[str, Dict[str, Any]] = {
    "intrusion": {
        "label": "Intrusive Trauma Memories & Re-experiencing",
        "dsm5_cluster": "Cluster B",
        "description": "Involuntary intrusive memories, vivid flashbacks, nightmares, and somatic trauma cue reactivity.",
        "phrases": [
            "flashback", "flashbacks", "nightmare", "nightmares", "night terror", "night terrors",
            "reliving", "reliving it", "intrusive memory", "intrusive thoughts", "bad dreams",
            "seeing his face", "seeing their faces", "can't unsee", "replaying in my head",
            "came flooding back", "smell reminds me", "sound took me back", "frozen with memory",
            "woke up screaming", "haunting me", "visual replay"
        ]
    },
    "avoidance": {
        "label": "Cognitive & Situational Avoidance",
        "dsm5_cluster": "Cluster C",
        "description": "Cognitive avoidance of trauma memories, behavioral avoidance of external reminders, social withdrawal.",
        "phrases": [
            "avoid", "avoiding", "cannot go there", "can't go near", "won't leave", "hiding",
            "staying away", "locked my door", "locked indoors", "shut myself in", "try not to think",
            "push it out of my mind", "blocking it out", "don't want to remember", "forcing myself to forget",
            "staying in room", "avoiding everyone", "don't want to see anyone", "crossing the street",
            "shut off my phone", "isolated myself"
        ]
    },
    "hyperarousal": {
        "label": "Hyperarousal & Threat Reactivity",
        "dsm5_cluster": "Cluster E",
        "description": "Hypervigilance, autonomic arousal, exaggerated startle, insomnia, irritability, safety scanning.",
        "phrases": [
            "hypervigilant", "on edge", "startled", "jumpy", "watching me", "watching my back",
            "looking over my shoulder", "heart pounding", "heart racing", "cannot sleep", "can't sleep",
            "sleeping with lights on", "awake all night", "tossing and turning", "on guard",
            "can't let my guard down", "flinching", "jumping at every noise", "chest tight",
            "can't breathe", "suffocating", "shaking", "trembling"
        ]
    },
    "cognition_mood": {
        "label": "Negative Alterations in Cognition & Mood",
        "dsm5_cluster": "Cluster D",
        "description": "Distorted self-blame, survivor guilt, pervasive negative affect, emotional detachment, alienation.",
        "phrases": [
            "numb", "feeling numb", "feel nothing", "detached", "guilt", "guilty", "ashamed",
            "shame", "my fault", "should have stopped it", "dirty", "humiliated", "not real",
            "outside my body", "no hope", "hopeless", "broken forever", "nobody understands",
            "no one cares", "alienated", "robot", "empty inside", "void", "worthless",
            "future is ruined", "lost my soul"
        ]
    }
}

def assess_trauma_signal(text: str) -> Dict[str, Any]:
    """
    Analyzes text against the TRACE trauma taxonomy.
    Extracts matched symptom clusters and exact context quotes for counsellor review.
    """
    text_clean = (text or "").lower()
    matched_clusters: Dict[str, List[str]] = {}
    evidence_snippets: List[Dict[str, str]] = []

    for cluster_id, cluster_meta in TRACE_SYMPTOM_CLUSTERS.items():
        cluster_matches = []
        for phrase in cluster_meta["phrases"]:
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            match = re.search(pattern, text_clean)
            if match:
                cluster_matches.append(phrase)
                # Extract surrounding context snippet (+/- 30 chars)
                start = max(0, match.start() - 25)
                end = min(len(text), match.end() + 25)
                snippet = text[start:end].strip()
                evidence_snippets.append({
                    "cluster": cluster_id,
                    "cluster_label": cluster_meta["label"],
                    "dsm5_cluster": cluster_meta["dsm5_cluster"],
                    "phrase": phrase,
                    "context_snippet": f"...{snippet}..."
                })
        if cluster_matches:
            matched_clusters[cluster_id] = cluster_matches

    cluster_count = len(matched_clusters)
    phrase_count = sum(len(m) for m in matched_clusters.values())
    
    # Transparent weighted signal calculation (0 - 100)
    # 25 points per unique DSM-5 cluster active + 5 points per matched phrase
    signal_score = min(100.0, cluster_count * 25.0 + phrase_count * 5.0)
    signal_score = round(signal_score, 1)

    review_recommended = cluster_count >= 2 or phrase_count >= 3 or signal_score >= 50.0

    return {
        "screening_type": "TRACE Trauma Language Corpus (EMNLP Findings 2024)",
        "signal_score": signal_score,
        "matched_clusters_count": cluster_count,
        "matched_clusters": list(matched_clusters.keys()),
        "matched_phrases_by_cluster": matched_clusters,
        "evidence_snippets": evidence_snippets,
        "clinician_review_recommended": review_recommended,
        "not_a_diagnosis": True,
        "model_version": "trace-trauma-linguistic-v2",
        "taxonomy_reference": "https://github.com/MiriamSchirmer/trauma-language"
    }
