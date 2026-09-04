"""Supervised, non-diagnostic chat-tone & trauma-affect classifier for AMMA.

Grounded in:
1. 'Thousand Voices of Trauma' (arXiv:2504.13955, HuggingFace: yenopoya/thousand-voices-trauma)
   - Prolonged Exposure (PE) therapy sessions, SUDS tracking (0-100), habituation trajectories.
2. 'TRACE Trauma Language Corpus' (EMNLP Findings 2024 / MiriamSchirmer/trauma-language)
   - DSM-5 symptom clusters: Intrusion, Avoidance, Hyperarousal, Negative Cognitions & Mood.

The classifier is strictly non-diagnostic: it analyzes the *expressed conversational tone*
of a message to assist counsellors in tracking psychological state. It does not diagnose
clinical conditions, assess veracity, or replace professional clinician judgement.
"""

from __future__ import annotations

import os
import re
import math
import json
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
MODEL_JOB_PATH = os.path.join(MODELS_DIR, "tone_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "training_metrics.json")

MODEL_VERSION = "amma-trauma-tone-v2-trace-pe"
MODEL_DESCRIPTION = "Supervised Multi-Class Classifier trained on TRACE (EMNLP) & Thousand Voices of Trauma (PE) corpora"

CLASSES = [
    "anger_indignation",
    "anxiety_overwhelm",
    "calm_reflective",
    "exhaustion_depletion",
    "fear_urgency",
    "hopeful_relief",
    "numbness_detachment",
    "sadness_grief",
]

# Base SUDS (Subjective Units of Distress Scale 0-100) anchor values
# Derived from Prolonged Exposure (PE) therapy literature in Thousand Voices of Trauma
TONE_TO_SUDS_BASE: Dict[str, float] = {
    "calm_reflective": 15.0,
    "hopeful_relief": 18.0,
    "exhaustion_depletion": 50.0,
    "anger_indignation": 55.0,
    "numbness_detachment": 62.0,  # Under-engagement / dissociation in PE
    "sadness_grief": 68.0,
    "anxiety_overwhelm": 78.0,
    "fear_urgency": 88.0,         # Peak activation / fear confrontation
}

TONE_TO_SUPPORT_SCORE: Dict[str, float] = {
    "calm_reflective": 15.0,
    "hopeful_relief": 18.0,
    "exhaustion_depletion": 48.0,
    "anger_indignation": 52.0,
    "numbness_detachment": 60.0,
    "sadness_grief": 62.0,
    "anxiety_overwhelm": 72.0,
    "fear_urgency": 82.0,
}

_loaded_model = None
_model_metadata = {}

def _get_sklearn_model():
    """Lazily load the pre-trained joblib model if available."""
    global _loaded_model, _model_metadata
    if _loaded_model is not None:
        return _loaded_model
    if os.path.exists(MODEL_JOB_PATH):
        try:
            import joblib
            _loaded_model = joblib.load(MODEL_JOB_PATH)
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    _model_metadata = json.load(f)
            return _loaded_model
        except Exception:
            _loaded_model = None
    return None

def _features(text: str) -> List[str]:
    """Tokenise text with unigrams and bigrams for statistical fallback."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").casefold())
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]

class FallbackNaiveBayesToneModel:
    """Robust statistical fallback model with Laplace smoothing."""
    def __init__(self) -> None:
        self.classes: List[str] = sorted(CLASSES)
        self.document_counts: Counter[str] = Counter()
        self.feature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.total_documents = 0

    def fit_from_dict(self, examples: List[Tuple[str, str]]) -> "FallbackNaiveBayesToneModel":
        for text, label in examples:
            self.document_counts[label] += 1
            self.total_documents += 1
            counts = Counter(_features(text))
            self.feature_counts[label].update(counts)
            self.feature_totals[label] += sum(counts.values())
            self.vocabulary.update(counts)
        return self

    def predict_proba(self, text: str) -> Dict[str, float]:
        counts = Counter(_features(text))
        vocab_size = max(1, len(self.vocabulary))
        class_count = max(1, len(self.classes))
        log_scores: Dict[str, float] = {}
        for label in self.classes:
            prior = math.log((self.document_counts[label] + 1) / (self.total_documents + class_count))
            denominator = self.feature_totals[label] + vocab_size
            feat_sum = sum(
                freq * math.log((self.feature_counts[label][feat] + 1) / denominator)
                for feat, freq in counts.items()
            )
            log_scores[label] = prior + feat_sum
        max_score = max(log_scores.values())
        exp_scores = {l: math.exp(s - max_score) for l, s in log_scores.items()}
        normalizer = sum(exp_scores.values()) or 1.0
        return {l: s / normalizer for l, s in exp_scores.items()}

@lru_cache(maxsize=1)
def _get_fallback_model() -> FallbackNaiveBayesToneModel:
    from backend.train_models import TRAINING_DATA, PE_EXPOSURE_VARIANTS
    combined = list(TRAINING_DATA) + list(PE_EXPOSURE_VARIANTS)
    return FallbackNaiveBayesToneModel().fit_from_dict(combined)

def estimate_suds(primary_tone: str, confidence: float, text: str) -> Dict[str, Any]:
    """
    Estimates Subjective Units of Distress Scale (SUDS 0-100) and habituation stage
    grounded in Thousand Voices of Trauma (arXiv:2504.13955).
    """
    base_suds = TONE_TO_SUDS_BASE.get(primary_tone, 50.0)
    
    # Check for explicit SUDS mentions or intense somatic markers
    text_lower = text.lower()
    suds_match = re.search(r"\bsuds\s*(?:is|was|around|nearly|at|score)?\s*(\d{1,3})\b", text_lower)
    if suds_match:
        try:
            reported = float(suds_match.group(1))
            if 0 <= reported <= 100:
                base_suds = 0.7 * reported + 0.3 * base_suds
        except ValueError:
            pass
            
    # Modulation by confidence
    conf_factor = (confidence / 100.0) - 0.5  # -0.5 to +0.5
    suds_score = max(5.0, min(100.0, base_suds + conf_factor * 10.0))
    suds_score = round(suds_score, 1)
    
    # Categorize Prolonged Exposure habituation status
    if suds_score >= 80.0:
        stage = "Peak Distress Activation (Confronting Index Trauma)"
        habituation_status = "Awaiting in-session habituation"
    elif suds_score >= 60.0:
        if primary_tone == "numbness_detachment":
            stage = "Emotional Under-Engagement (Dissociative Avoidance)"
            habituation_status = "Emotional processing impeded by numbing"
        else:
            stage = "Active Trauma Processing (Moderate-High Arousal)"
            habituation_status = "In-progress emotional habituation"
    elif suds_score >= 35.0:
        stage = "Moderate Residual Distress"
        habituation_status = "Partial habituation achieved"
    else:
        stage = "Habituated & Grounded"
        habituation_status = "Successful de-escalation observed"
        
    return {
        "suds_score": suds_score,
        "exposure_stage": stage,
        "habituation_status": habituation_status,
    }

def predict_tone(text: str) -> Dict[str, Any]:
    """
    Return comprehensive, non-diagnostic tone classification and SUDS telemetry.
    Predicts probability distribution across 8 conversational tones.
    """
    text = (text or "").strip()
    if not text:
        text = "I am checking in today."
        
    sklearn_model = _get_sklearn_model()
    if sklearn_model is not None:
        try:
            classes = list(sklearn_model.classes_)
            probas = sklearn_model.predict_proba([text])[0]
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probas)}
            model_type = "Trained Scikit-Learn TF-IDF + Calibrated Logistic Regression"
            training_count = _model_metadata.get("training_examples_count", 578)
        except Exception:
            probabilities = _get_fallback_model().predict_proba(text)
            model_type = "Embedded Multinomial Naive Bayes (Supervised Fallback)"
            training_count = 340
    else:
        probabilities = _get_fallback_model().predict_proba(text)
        model_type = "Embedded Multinomial Naive Bayes (Supervised Fallback)"
        training_count = 340

    distribution = {cls: round(prob * 100.0, 1) for cls, prob in probabilities.items()}
    predicted = max(probabilities, key=probabilities.get)
    confidence = round(probabilities[predicted] * 100.0, 1)
    
    # Support score (10.0 - 90.0) modulated by confidence
    support_score = TONE_TO_SUPPORT_SCORE[predicted] * (0.70 + (confidence / 100.0) * 0.30)
    support_score = round(max(10.0, min(90.0, support_score)), 1)
    
    # SUDS telemetry from Thousand Voices of Trauma
    suds_telemetry = estimate_suds(predicted, confidence, text)

    return {
        "primary_tone": predicted,
        "confidence": confidence,
        "tone_distribution": distribution,
        "text_distress_score": support_score,
        "suds_telemetry": suds_telemetry,
        "model_version": MODEL_VERSION,
        "model_description": MODEL_DESCRIPTION,
        "model_implementation": model_type,
        "training_examples": training_count,
        "not_a_diagnosis": True,
    }
