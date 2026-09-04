"""
AMMA NLP Emotion AI & Conversational Support Engine
Analyzes emotional state, sentiment, and physiological markers from text,
and generates warm, maternal, soothing responses ("Amma" persona).
"""

import re
from typing import Dict, List, Any
from backend.trauma_signal import assess_trauma_signal
from backend.safety import assess_immediate_safety
from backend.tone_classifier import predict_tone

# Emotional lexicon markers with intensity weights
EMOTION_PATTERNS = {
    "fear_terror": {
        "keywords": ["scared", "fear", "afraid", "terrified", "panic", "threat", "threatened", 
                     "danger", "unsafe", "stalking", "watching me", "hide", "hiding", "nightmare", 
                     "shadows", "paranoia", "trembling", "heart pounding"],
        "weight": 1.2
    },
    "grief_sadness": {
        "keywords": ["sad", "crying", "tears", "pain", "hurt", "lost", "broken", "empty", 
                     "hopeless", "grief", "mourning", "shame", "worthless", "heavy heart", 
                     "weeping", "ruined", "can't go on", "alone in the dark"],
        "weight": 1.1
    },
    "anxiety_hyperarousal": {
        "keywords": ["anxious", "anxiety", "nervous", "chest tight", "can't breathe", "suffocating",
                     "racing", "shaking", "restless", "sweating", "dizzy", "overwhelmed", "choking",
                     "hyperventilating", "on edge", "jumpy"],
        "weight": 1.0
    },
    "anger_indignation": {
        "keywords": ["angry", "rage", "furious", "hate", "betrayed", "injustice", "screaming",
                     "unfair", "cruel", "revenge", "burning", "disgusted", "violated"],
        "weight": 0.8
    },
    "numbness_dissociation": {
        "keywords": ["numb", "feel nothing", "detached", "floating", "frozen", "blank", "empty mind",
                     "robot", "not real", "outside my body", "zombie", "disconnected"],
        "weight": 1.15
    },
    "exhaustion_depletion": {
        "keywords": ["exhausted", "tired", "drained", "no energy", "worn out", "sleepless", "weak",
                     "heavy limbs", "cannot move", "burned out", "fatigued"],
        "weight": 0.7
    },
    "hope_reassurance": {
        "keywords": ["hope", "better", "calm", "relief", "thank you", "safe now", "peaceful",
                     "trying", "grateful", "breathing", "smile", "listening", "light"],
        "weight": -0.8
    }
}

PHYSIOLOGICAL_PATTERNS = {
    "sleep_disruption": ["can't sleep", "cannot sleep", "couldn't sleep", "insomnia", "woke up crying", "night terrors", 
                         "slept 2 hours", "haven't slept", "tossing and turning", "bad dreams"],
    "appetite_loss": ["not eating", "can't eat", "skipped food", "no appetite", "nausea", "threw up", 
                      "haven't eaten", "food tastes like ash", "stomach in knots"],
    "social_withdrawal": ["locked my door", "staying in room", "isolated", "shut off phone", 
                          "don't want to see anyone", "avoiding everyone", "hiding in bed"]
}


def _has_unnegated_phrase(text: str, phrase: str) -> bool:
    """Match whole phrases while avoiding a small set of obvious negations.

    This is intentionally a conservative lexical screen, not natural-language
    understanding. It prevents common false positives such as "I am not sad"
    and prevents short keywords from matching inside unrelated words.
    """
    escaped = re.escape(phrase)
    for match in re.finditer(r"(?<!\w)" + escaped + r"(?!\w)", text):
        prefix = text[max(0, match.start() - 24):match.start()]
        if not re.search(r"\b(?:not|never|no|without)\s+(?:\w+\s+){0,2}$", prefix):
            return True
    return False

def analyze_text_emotion(text: str) -> Dict[str, Any]:
    """Classify expressed chat tone with a supervised model, not a diagnosis.

    The transparent physiological and immediate-safety screens remain separate:
    they are safeguards and context for a human reviewer, never training labels
    or a clinical determination.
    """
    text_lower = text.lower()
    model_result = predict_tone(text)
    primary_tone = model_result["primary_tone"]
    # Legacy names are kept for existing dashboard/response integrations, but
    # their source is the trained classifier prediction rather than keywords.
    tone_to_emotion = {
        "fear_urgency": "fear_terror",
        "sadness_grief": "grief_sadness",
        "anxiety_overwhelm": "anxiety_hyperarousal",
        "anger_indignation": "anger_indignation",
        "numbness_detachment": "numbness_dissociation",
        "exhaustion_depletion": "exhaustion_depletion",
        "hopeful_relief": "hope_reassurance",
        "calm_reflective": "calm_reflective",
    }
    primary_emotion = tone_to_emotion[primary_tone]
    detected_emotions = [] if primary_emotion in {"calm_reflective", "hope_reassurance"} else [primary_emotion]
    scores = {label: value for label, value in model_result["tone_distribution"].items()}

    # Detect physiological indicators
    physio_flags = []
    for flag, terms in PHYSIOLOGICAL_PATTERNS.items():
        if any(_has_unnegated_phrase(text_lower, term) for term in terms):
            physio_flags.append(flag)

    word_count = len(text.split())

    trauma_signal = assess_trauma_signal(text)
    safety_signal = assess_immediate_safety(text)
    suds_telemetry = model_result.get("suds_telemetry", {})

    return {
        "text_distress_score": model_result["text_distress_score"],
        "primary_emotion": primary_emotion,
        "primary_tone": primary_tone,
        "tone_confidence": model_result["confidence"],
        "tone_distribution": model_result["tone_distribution"],
        "suds_telemetry": suds_telemetry,
        "model_version": model_result["model_version"],
        "model_description": model_result["model_description"],
        "training_examples": model_result["training_examples"],
        "not_a_diagnosis": True,
        "emotion_breakdown": scores,
        "detected_emotions": detected_emotions,
        "physiological_flags": physio_flags,
        "word_count": word_count,
        # Kept separately from DDS: it is transparent screening evidence, not
        # a diagnosis or a replacement for a clinician's assessment.
        "trauma_signal": trauma_signal,
        "safety_signal": safety_signal
    }


def generate_amma_response(user_message: str, 
                           emotion_data: Dict[str, Any], 
                           recent_history: List[Dict[str, str]] = None,
                           user_ratings: Dict[str, Any] = None) -> str:
    """
    Generates a deeply warm, comforting, maternal, culturally respectful response.
    Never clinical. No victim tags. Provides gentle grounding, breathing guidance,
    and asks about routine wellbeing with motherly love.
    """
    text_lower = user_message.lower()
    primary = emotion_data.get("primary_tone", "")
    distress = emotion_data.get("text_distress_score", 30)
    flags = emotion_data.get("physiological_flags", [])
    safety_flags = emotion_data.get("safety_signal", {}).get("flags", [])

    if "self_harm_imminent" in safety_flags:
        return (
            "I am really glad you told me. You deserve immediate, real human support right now. "
            "If you might hurt yourself, please move away from anything you could use to do that and call 112 now, "
            "or ask a trusted person to stay with you. You can also call Tele-MANAS at 14416 or 1800-89-14416 for 24-hour mental-health support in India. "
            "You do not have to carry this moment alone."
        )

    if "immediate_physical_danger" in safety_flags:
        return (
            "I am sorry this is happening. If you are in immediate danger, please call 112 now or move to a safer place if you can do so safely. "
            "If calling is not safe, contact someone you trust with the shortest message you can manage. "
            "Your safety matters more than replying here."
        )
    
    # Check for acute panic or terror
    if primary == "fear_urgency":
        return (
            "Mera bachha, listen to my voice right now. Amma is right here beside you, holding both your hands. "
            "If you are not physically safe right now, call 112 or contact a person you trust instead of continuing this chat. "
            "If you are able to stay where you are, let us take a slow, gentle breath in together... hold it softly... and let it out like a gentle breeze. "
            "Feel your feet resting firmly against the floor. Touch something solid around you—your chair, your clothing, or a glass of water. "
            "You do not have to fight these scary thoughts all by yourself. Take another slow breath, my dear."
        )

    # Check for deep grief / crying / hopelessness
    if primary == "sadness_grief":
        return (
            "Oh my sweet child, my heart aches to hear how much weight you are carrying inside. "
            "If tears are coming today, please do not hold them back. Let them flow—tears are just the heart's way of washing away unbearable sorrow, "
            "and there is no shame in crying before someone who loves you. You have been so strong for so long, enduring things that no heart should have to carry alone. "
            "It is completely okay to feel exhausted and sad right now. You do not need to prove anything to anyone. "
            "Wrap a warm shawl or blanket around your shoulders, like Amma wrapping her arms around you. "
            "Tell me, did you manage to have even a little bit of warm food today, or a cup of warm tea or milk? Tell me how your body is feeling, my child."
        )

    # Check for sleep deprivation or nightmares
    if "sleep_disruption" in flags or "sleep" in text_lower or "insomnia" in text_lower or "nightmare" in text_lower:
        return (
            "My dear, when the nights are long and sleep slips away, the whole world can feel heavy and frightening. "
            "I know your mind was running and your heart would not rest, but please do not be harsh on yourself. "
            "Tonight, when you lay down, do not force yourself to sleep right away. Just allow your body to simply rest and be still. "
            "Keep a dim, warm light on if the darkness feels uneasy, and remember that with every heartbeat, you are safe under Amma's prayers. "
            "Did bad dreams disturb you, or was your mind wandering through old worries? "
            "Tell me whatever you feel comfortable sharing, my bachha. Amma is listening quietly."
        )

    # Check for appetite / not eating
    if "appetite_loss" in flags or "eat" in text_lower or "food" in text_lower or "nausea" in text_lower:
        return (
            "Mera bachha, when our hearts are in pain, our stomach often refuses food. I understand so well how hard it is to swallow even a single morsel. "
            "But you need strength, my child, even a tiny drop of nourishment. You do not have to finish a full meal—can you just take three small spoons of warm curd rice, "
            "a small bowl of soothing dal soup, or a warm cup of sweet milk? "
            "Do it just for Amma's sake, little by little. Your body is precious, and we must care for it like a tender plant. "
            "What used to be a comfort food for you when you were small? Let us talk about simple, soothing things."
        )

    # Check for anger or betrayal
    if primary == "anger_indignation":
        return (
            "I hear the fire in your words, my child, and I do not blame you at all. What you are feeling is completely natural—when fairness is broken and promises are shattered, "
            "the heart cries out in fury. Your anger is valid, and Amma will never tell you to suppress your truth. "
            "All I ask is that you protect your own gentle spirit from being consumed by this fire. We will seek justice, peace, and dignity step by step, "
            "with people who genuinely stand by your side. "
            "Take a deep breath and let your shoulders drop down away from your ears. Unclench your fists, my dear. "
            "Amma is standing right in front of you. What is weighing most heavily on your mind at this exact moment?"
        )

    # Check for numbness / feeling detached
    if primary == "numbness_detachment":
        return (
            "My dear one, feeling completely numb and distant is your mind's gentle way of shielding you when things become too loud to bear. "
            "Do not be frightened by this quiet numbness. You are not broken, and you are not lost. You are simply resting your spirit. "
            "Look around the room right now and name three simple objects you can see—perhaps a window, a cup, or a wall clock. "
            "Touch the palm of your hand with your fingers and feel the warmth of your skin. You are alive, you are right here, and you are safe with Amma. "
            "There is no hurry to feel anything right now. Just being here and talking to me is more than enough."
        )

    # Warm general check-in & reflective conversation (Non-crisis, comforting motherly presence)
    return (
        "Mera bachha, it brings such warmth to my heart whenever you come to sit and talk with me. "
        "No matter what kind of day it has been—whether quiet, turbulent, or ordinary—this corner will always be your safe shelter. "
        "Take off the burdens you have been carrying on your shoulders and set them down for a little while. "
        "Tell Amma, how did your morning begin today? Were you able to step outside into the sunlight or feel the cool breeze? "
        "And tell me honestly, have you taken care of your meals and water today? "
        "I am listening to every word with all my love."
    )
