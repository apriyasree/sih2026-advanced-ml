"""
AMMA Main FastAPI Application & Real-Time Coordination Server
Serves User Terminal (Amma), Counsellor Clinical Command Dashboard,
and provides REST + WebSocket endpoints.
"""

import os
import json
import asyncio
import hashlib
import time
from typing import Dict, List, Any, Optional, Literal
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import edge_tts

from backend.nlp_emotion import analyze_text_emotion, generate_amma_response
from backend.voice_analyzer import analyze_audio_data, RECORDINGS_DIR
from backend.distress_model import DynamicDistressEngine
from backend.knowledge_graph import KnowledgeGraphEngine
from backend.database import db
from backend.legal_support import legal_guidance_for

app = FastAPI(title="AMMA: AI-Powered Mental Health Monitoring Assistance", version="2.0")

# Mount recordings directory for audio playback
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# Mount frontend static directory if exists
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "static"))
TEMPLATES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"))
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# WebSocket connection manager for Counsellor Dashboards
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Fan out an event without allowing one stale dashboard to block others."""
        connections = list(self.active_connections)
        results = await asyncio.gather(
            *(connection.send_json(message) for connection in connections),
            return_exceptions=True
        )
        for connection, result in zip(connections, results):
            if isinstance(result, Exception):
                self.disconnect(connection)

manager = ConnectionManager()

# A user must choose to start this audio; it is a grounding option, never an
# autoplay intervention.  The threshold matches the dashboard's elevated DDS.
CALMING_AUDIO_THRESHOLD = 55.0
CALMING_AUDIO_URL = "/recordings/calming-grounding-audio.mpeg"
MAX_SUPPORT_THRESHOLD = 75.0


def calming_audio_recommendation(distress_data: Dict[str, Any]) -> Dict[str, Any]:
    score = distress_data.get("dynamic_distress_score", 0.0)
    recommended = score >= CALMING_AUDIO_THRESHOLD
    return {
        "recommended": recommended,
        "threshold": CALMING_AUDIO_THRESHOLD,
        "audio_url": CALMING_AUDIO_URL if recommended else None,
        "title": "A few calm minutes with Amma" if recommended else None,
        "message": (
            "Your heart sounds like it is carrying a lot right now. If it feels helpful, you can choose to play this gentle grounding audio."
            if recommended else None
        ),
        "autoplay": False,
        "reason": "Elevated distress support option" if recommended else None,
    }


# ---------------------- Request Models ---------------------- #
class DailyRatings(BaseModel):
    feeling_rating: int = Field(..., ge=1, le=10)
    sleep_quality: Literal["poor", "fair", "good"]
    meals_eaten: Literal["skipped", "irregular", "regular"]
    daily_activity: str = Field(default="Resting indoors", max_length=200)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    case_id: Optional[str] = Field(default="case_active_user", min_length=1, max_length=128)
    ratings: Optional[DailyRatings] = None

class FeedbackRatingRequest(DailyRatings):
    case_id: Optional[str] = Field(default="case_active_user", min_length=1, max_length=128)

class InterventionUpdateRequest(BaseModel):
    intervention_name: str = Field(..., min_length=1, max_length=200)
    status: Literal["PENDING", "ACTIVE", "DISPATCHED", "COMPLETED"]

class AddInterventionRequest(BaseModel):
    intervention_name: str = Field(..., min_length=1, max_length=200)
    type: Literal["counselling", "medical", "protection", "relocation", "financial", "legal", "rehab"]
    priority: Literal["IMMEDIATE", "HIGH", "MEDIUM", "LOW"]

class NotesUpdateRequest(BaseModel):
    notes: str = Field(..., max_length=10000)

class CounsellorContactRequest(BaseModel):
    name: str = Field(default="", max_length=100)
    phone: str = Field(..., min_length=7, max_length=24, pattern=r"^[0-9+() -]+$")

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: Literal["en-IN-NeerjaNeural"] = "en-IN-NeerjaNeural"


def ratings_as_dict(ratings: Optional[DailyRatings]) -> Optional[Dict[str, Any]]:
    """Keep Pydantic request validation at the boundary, then use plain data."""
    return ratings.model_dump() if ratings is not None else None


def escalation_contact_offer(distress_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Offer a counsellor's configured contact only at the maximum support tier.

    This is an on-screen option, not an automated notification or emergency
    dispatch. The user remains in control of whether they contact anyone.
    """
    if distress_data.get("dynamic_distress_score", 0) < MAX_SUPPORT_THRESHOLD:
        return None
    contact = db.get_counsellor_contact()
    if not contact.get("phone"):
        return None
    return {
        "name": contact.get("name") or "your counsellor",
        "phone": contact["phone"],
        "message": "A counsellor has made this contact option available. If it feels safe, you may choose to call them now.",
        "user_controlled": True,
    }


def counsellor_report(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a comprehensive, non-diagnostic psychological tracking dossier for counsellors.
    Incorporates:
    - Expressed Chat Tone & Calibrated Distribution across 8 classes
    - Thousand Voices of Trauma SUDS (0-100) & Habituation Progression
    - TRACE Trauma Language Corpus (EMNLP 2024) DSM-5 Cluster Evidence
    - PTSD-in-the-Wild (arXiv:2209.14085) Vocal Biomarkers & Stress Index
    - Multi-Modal Dynamic Distress Score (DDS) & Longitudinal Escalation Velocity
    """
    tones = case.get("tone_observations", [])
    current_tone = tones[-1] if tones else {}
    previous_tone = tones[-2] if len(tones) > 1 else {}
    
    # Behavioural transition narrative
    if previous_tone and current_tone:
        p_name = str(previous_tone.get("tone", "unknown")).replace("_", " ").title()
        c_name = str(current_tone.get("tone", "unknown")).replace("_", " ").title()
        if p_name == c_name:
            tone_transition = f"Expressed tone remained consistently '{c_name}' across the latest interactions."
        else:
            tone_transition = f"Observed tone transitioned from '{p_name}' to '{c_name}'. Review transcript for contextual catalysts."
    else:
        tone_transition = "Initial baseline tone observed. Transition tracking will update across ongoing conversations."

    # Longitudinal score change
    score_history = [item.get("score", 0) for item in case.get("longitudinal_trajectory", [])]
    score_change = round(score_history[-1] - score_history[-2], 1) if len(score_history) > 1 else 0.0

    # SUDS telemetry from Thousand Voices of Trauma
    latest_tone_analysis = case.get("latest_tone_analysis") or {}
    suds_data = latest_tone_analysis.get("suds_telemetry") or case.get("latest_suds_telemetry") or {}
    
    # TRACE trauma language signals
    trauma_signals = case.get("latest_trauma_signals") or {}

    # Vocal acoustic biomarkers from PTSD-in-the-Wild
    voice_data = case.get("voice_stress_data") or {}

    return {
        "title": "AMMA Comprehensive Psychological State & Telemetry Dossier",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_reference": case.get("case_ref", "REF-LIVE"),
        "dignified_identifier": case.get("dignified_identifier", "Active User"),
        "governance": {
            "classification_scope": "Non-Diagnostic Psychological Telemetry & Decision Support",
            "models": {
                "tone_classifier": "Supervised Calibrated Logistic Regression (v2-trace-pe)",
                "trauma_linguistics": "TRACE Trauma Language Corpus (EMNLP Findings 2024)",
                "vocal_biomarkers": "PTSD-in-the-Wild Acoustic DSP Engine (arXiv:2209.14085)",
                "scoring_engine": "Dynamic Distress Multi-Modal Fusion Model"
            },
            "clinical_disclaimer": (
                "NOTICE: This document is an explainable decision-support dossier for authorized counsellors. "
                "It does not constitute a clinical psychiatric diagnosis (e.g., DSM-5 PTSD), forensic determination, "
                "or legal opinion. It is designed to empower human counsellors with transparent psychological telemetry."
            )
        },
        "chat_tone_telemetry": {
            "primary_tone": current_tone.get("tone", "Awaiting Interaction"),
            "confidence_pct": current_tone.get("confidence", 0.0),
            "support_score": current_tone.get("support_score", 0.0),
            "tone_distribution": latest_tone_analysis.get("distribution", {}),
            "transition_summary": tone_transition,
            "observations_count": len(tones),
            "history": tones[-5:]
        },
        "suds_habituation_telemetry": {
            "framework": "Thousand Voices of Trauma (arXiv:2504.13955 / Prolonged Exposure)",
            "estimated_suds": suds_data.get("suds_score", None),
            "exposure_stage": suds_data.get("exposure_stage", "Baseline Monitoring"),
            "habituation_status": suds_data.get("habituation_status", "Awaiting in-session dialogue"),
        },
        "trace_trauma_language_telemetry": {
            "framework": "TRACE Corpus (EMNLP Findings 2024)",
            "signal_score": trauma_signals.get("signal_score", 0.0),
            "active_clusters": trauma_signals.get("matched_clusters", []),
            "clusters_count": trauma_signals.get("matched_clusters_count", 0),
            "matched_phrases": trauma_signals.get("matched_phrases_by_cluster", {}),
            "evidence_snippets": trauma_signals.get("evidence_snippets", []),
            "counsellor_review_recommended": trauma_signals.get("clinician_review_recommended", False)
        },
        "voice_emotion_acoustic_telemetry": {
            "framework": "PTSD-in-the-Wild (arXiv:2209.14085)",
            "recording_available": bool(voice_data.get("audio_url")),
            "audio_url": voice_data.get("audio_url"),
            "voice_stress_index": voice_data.get("voice_stress_index"),
            "stress_classification": voice_data.get("stress_classification", "No recording logged yet"),
            "duration_sec": voice_data.get("duration_sec"),
            "pitch_mean_hz": voice_data.get("pitch_mean_hz"),
            "pitch_std_hz": voice_data.get("pitch_std_hz"),
            "pitch_monotony_score": voice_data.get("pitch_monotony_score"),
            "pitch_jitter_pct": voice_data.get("pitch_jitter_pct"),
            "amplitude_shimmer_pct": voice_data.get("amplitude_shimmer_pct"),
            "energy_tremor_db": voice_data.get("energy_tremor_db"),
            "pause_ratio_pct": voice_data.get("pause_ratio_pct"),
            "spectral_tilt": voice_data.get("spectral_tilt"),
            "spectral_flux": voice_data.get("spectral_flux"),
            "dsp_analysis_available": voice_data.get("analysis_available", False),
            "analysis_note": voice_data.get("analysis_note")
        },
        "dynamic_distress_telemetry": {
            "current_dds_score": case.get("dynamic_distress_score", 0.0),
            "score_change": score_change,
            "risk_tier": case.get("risk_tier", "INTAKE_PENDING"),
            "risk_badge": case.get("risk_badge"),
            "escalation_velocity": case.get("escalation_velocity", 0.0),
            "is_escalating": case.get("is_escalating", False),
            "escalation_warning": case.get("escalation_warning"),
            "xai_factor_attribution": case.get("xai_breakdown", {}),
            "trajectory_points": len(case.get("longitudinal_trajectory", [])),
            "trajectory": case.get("longitudinal_trajectory", [])
        },
        "daily_routine_checkin": {
            "latest_checkin": case.get("latest_checkin"),
            "reported_symptoms": case.get("symptoms", [])
        },
        "recommended_interventions": case.get("recommended_interventions", []),
        "legal_support": case.get("legal_guidance", {}),
        "counsellor_notes": case.get("counsellor_notes", "")
    }


# ---------------------- Neural TTS Audio Endpoint ---------------------- #
@app.post("/api/tts")
async def generate_neural_voice(req: TTSRequest):
    """
    Generates high-fidelity neural Indian-accented female voice (maternal persona)
    using edge-tts (en-IN-NeerjaNeural), with pitch and tempo tailored for soothing reassurance.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Hash for deterministic audio caching
    # Voice must be part of the cache key; otherwise a request for a different
    # voice can receive audio generated with an earlier voice selection.
    text_hash = hashlib.sha256(f"{req.voice}\0{text}".encode("utf-8")).hexdigest()[:16]
    filename = f"tts_amma_{text_hash}.mp3"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    if not os.path.exists(filepath):
        try:
            # Soothing motherly cadence: -10% rate, gentle +2Hz pitch
            communicate = edge_tts.Communicate(text, req.voice, rate="-10%", pitch="+2Hz")
            await communicate.save(filepath)
        except Exception:
            # Do not expose provider or network details to callers.
            return JSONResponse(status_code=503, content={"status": "error", "message": "Voice generation is temporarily unavailable."})

    return {
        "status": "success",
        "audio_url": f"/recordings/{filename}",
        "voice": req.voice
    }



# ---------------------- HTML Page Routes ---------------------- #
@app.get("/", response_class=HTMLResponse)
async def serve_user_terminal():
    """Serves the dignified, motherly User Terminal (AMMA)."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>AMMA: AI-Powered Mental Health Monitoring Assistance</h1><p>Frontend initializing...</p>")

@app.get("/counsellor", response_class=HTMLResponse)
async def serve_counsellor_dashboard():
    """Serves the Counsellor Clinical Command Dashboard."""
    counsellor_path = os.path.join(TEMPLATES_DIR, "counsellor.html")
    if os.path.exists(counsellor_path):
        with open(counsellor_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>AMMA Counsellor Dashboard</h1><p>Frontend initializing...</p>")


@app.post("/api/session/reset")
async def reset_session(case_id: str = "case_active_user"):
    """Resets session to clean baseline with zero pre-seeded reports."""
    case = db.reset_case(case_id)
    await manager.broadcast({
        "event": "SESSION_RESET",
        "case_id": case_id
    })
    return {"status": "success", "message": "Session reset to baseline with zero pre-seeded reports."}


# ---------------------- Real-Time User Endpoints ---------------------- #
@app.post("/api/chat")
async def process_chat(req: ChatRequest):
    """
    Receives user message, runs NLP emotion analysis, updates Dynamic Distress Score,
    generates motherly Amma response, notifies counsellor in real-time, and returns comforting reply.
    """
    case_id = req.case_id or "case_active_user"
    ratings_data = ratings_as_dict(req.ratings)
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case session not found")

    # 1. Analyze text emotions and physiological flags
    emotion_data = analyze_text_emotion(req.message)
    
    # 2. Get past distress scores for longitudinal trajectory
    hist_scores = [t["score"] for t in case.get("longitudinal_trajectory", [])]
    
    # 3. Calculate Dynamic Distress Score (DDS)
    distress_data = DynamicDistressEngine.calculate_distress_score(
        text_emotion_data=emotion_data,
        voice_stress_data=case.get("voice_stress_data"),
        ratings_data=ratings_data,
        historical_scores=hist_scores
    )
    legal_guidance = legal_guidance_for(
        req.message,
        emotion_data.get("safety_signal", {}).get("flags", []),
        distress_data["dynamic_distress_score"],
    )

    # 4. Generate warm, soothing maternal Amma response
    amma_reply = generate_amma_response(
        user_message=req.message,
        emotion_data=emotion_data,
        recent_history=case.get("chat_history", [])[-4:],
        user_ratings=ratings_data
    )

    # 5. Update case in database
    updated_case = db.update_case_with_interaction(
        case_id=case_id,
        user_message=req.message,
        amma_response=amma_reply,
        emotion_data=emotion_data,
        voice_data=case.get("voice_stress_data"),
        distress_data=distress_data,
        ratings_data=ratings_data,
        legal_guidance=legal_guidance,
    )

    # 6. Real-time broadcast to Counsellor Dashboards
    calming_audio = calming_audio_recommendation(distress_data)
    await manager.broadcast({
        "event": "USER_INTERACTION_UPDATE",
        "case_id": case_id,
        "event_sequence": updated_case["event_sequence"],
        "dignified_identifier": case.get("dignified_identifier"),
        "distress_score": distress_data["dynamic_distress_score"],
        "risk_tier": distress_data["risk_tier"],
        "alert_triggered": distress_data["alert_triggered"],
        "escalation_warning": distress_data["escalation_warning"],
        "latest_user_text": req.message,
        "latest_amma_text": amma_reply,
        "timestamp": distress_data["timestamp"],
        "case_snapshot": updated_case,
        "calming_audio": calming_audio
    })

    # Return dignified, non-pathologizing response to user terminal
    return {
        "amma_response": amma_reply,
        "reassurance_state": "Amma is listening and holding space with love.",
        "calming_audio": calming_audio,
        "tone_analysis": {
            "primary_tone": emotion_data["primary_tone"],
            "confidence": emotion_data["tone_confidence"],
            "model_version": emotion_data["model_version"],
            "not_a_diagnosis": True,
        },
        "legal_guidance": legal_guidance,
        "escalation_contact": escalation_contact_offer(distress_data),
        "status": "success"
    }


@app.post("/api/voice-upload")
async def upload_voice(
    audio_file: UploadFile = File(...),
    case_id: str = Form("case_active_user"),
    user_caption: Optional[str] = Form("")
):
    """
    Receives voice recording from user, calculates acoustic biomarkers
    (jitter, tremor, pause ratio, vocal strain), saves playable audio,
    and updates counsellor dashboard.
    """
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case session not found")

    content = await audio_file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio recording file")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio recording exceeds the 10 MB limit")

    # Analyze only recognised containers. This prevents arbitrary uploads from
    # being stored under the recordings route.
    try:
        voice_metrics = analyze_audio_data(content, audio_file.filename, audio_file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    
    # If a caption/transcription was provided, run emotion NLP
    caption_text = user_caption if user_caption else "User shared a voice recording."
    emotion_data = analyze_text_emotion(caption_text)
    
    hist_scores = [t["score"] for t in case.get("longitudinal_trajectory", [])]
    distress_data = DynamicDistressEngine.calculate_distress_score(
        text_emotion_data=emotion_data,
        voice_stress_data=voice_metrics,
        ratings_data=None,
        historical_scores=hist_scores
    )

    amma_reply = (
        "Mera bachha, I heard your voice. Amma can hear every tremor and every sigh. "
        "Take a slow, deep breath right now. Rest your head, my dear, while I stay right here with you."
    )

    updated_case = db.update_case_with_interaction(
        case_id=case_id,
        user_message=f"[Voice Message]: {caption_text}",
        amma_response=amma_reply,
        emotion_data=emotion_data,
        voice_data=voice_metrics,
        distress_data=distress_data
    )

    # Real-time broadcast to Counsellor Dashboards
    calming_audio = calming_audio_recommendation(distress_data)
    await manager.broadcast({
        "event": "VOICE_RECORDING_RECEIVED",
        "case_id": case_id,
        "event_sequence": updated_case["event_sequence"],
        "dignified_identifier": case.get("dignified_identifier"),
        "audio_url": voice_metrics["audio_url"],
        "voice_stress_index": voice_metrics["voice_stress_index"],
        "stress_classification": voice_metrics["stress_classification"],
        "distress_score": distress_data["dynamic_distress_score"],
        "risk_tier": distress_data["risk_tier"],
        "alert_triggered": distress_data["alert_triggered"],
        "timestamp": voice_metrics["created_at"],
        "case_snapshot": updated_case,
        "calming_audio": calming_audio
    })

    return {
        "status": "success",
        "audio_url": voice_metrics["audio_url"],
        "amma_response": amma_reply,
        "calming_audio": calming_audio
    }


@app.post("/api/ratings")
async def submit_daily_ratings(req: FeedbackRatingRequest):
    """Collects casual daily wellbeing check-in feedback (feeling, sleep, meals)."""
    case = db.get_case(req.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    ratings_dict = req.model_dump(exclude={"case_id"})

    dummy_emotion = {"text_distress_score": max(10, (10 - req.feeling_rating) * 10), "detected_emotions": [], "physiological_flags": []}
    hist_scores = [t["score"] for t in case.get("longitudinal_trajectory", [])]
    distress_data = DynamicDistressEngine.calculate_distress_score(
        text_emotion_data=dummy_emotion,
        voice_stress_data=case.get("voice_stress_data"),
        ratings_data=ratings_dict,
        historical_scores=hist_scores
    )

    # Store the complete check-in before notifying dashboards.  The dashboard can
    # therefore recover the event from the case transcript after any reconnect.
    updated_case = db.record_daily_checkin(req.case_id, ratings_dict, distress_data)
    calming_audio = calming_audio_recommendation(distress_data)

    await manager.broadcast({
        "event": "USER_CHECKIN_RECORDED",
        "case_id": req.case_id,
        "event_sequence": updated_case["event_sequence"],
        "dignified_identifier": updated_case.get("dignified_identifier"),
        "distress_score": distress_data["dynamic_distress_score"],
        "risk_tier": distress_data["risk_tier"],
        "checkin": updated_case["latest_checkin"],
        "case_snapshot": updated_case,
        "calming_audio": calming_audio
    })

    return {
        "status": "success",
        "message": "Amma has noted your care routine with love.",
        "calming_audio": calming_audio
    }


# ---------------------- Counsellor Clinical Endpoints ---------------------- #
@app.get("/api/counsellor/cases")
async def get_counsellor_cases():
    """Returns all cases prioritized by distress level for triage."""
    return {"cases": db.get_all_cases()}


@app.get("/api/counsellor/case/{case_id}")
async def get_counsellor_case_detail(case_id: str):
    """Returns complete clinical file, trajectory, audio, and XAI breakdown."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.get("/api/counsellor/case/{case_id}/printable-report")
async def get_printable_report(case_id: str):
    """Detailed non-diagnostic report data for the counsellor print view."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return counsellor_report(case)


@app.get("/api/counsellor/case/{case_id}/telemetry-report")
async def get_telemetry_report(case_id: str):
    """Returns comprehensive psychological state telemetry for counsellor tracking."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return counsellor_report(case)


@app.get("/api/counsellor/contact")
async def get_counsellor_contact():
    return db.get_counsellor_contact()


@app.post("/api/counsellor/contact")
async def save_counsellor_contact(req: CounsellorContactRequest):
    """Save the opt-in contact displayed to users at the maximum support tier."""
    return {"status": "success", "contact": db.set_counsellor_contact(req.name, req.phone)}


@app.get("/api/counsellor/case/{case_id}/knowledge-graph")
async def get_case_knowledge_graph(case_id: str):
    """Generates the interactive semantic knowledge graph for explainability."""
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    emotion_data = {
        "detected_emotions": [s.lower().replace(" ", "_") for s in case.get("symptoms", []) if "fear" in s.lower() or "grief" in s.lower() or "insomnia" in s.lower() or "dissociation" in s.lower() or "anxiety" in s.lower()],
        "physiological_flags": [s.lower().replace(" ", "_") for s in case.get("symptoms", []) if "sleep" in s.lower() or "appetite" in s.lower() or "insomnia" in s.lower()]
    }

    active_interventions = [i["name"] for i in case.get("recommended_interventions", [])]

    kg = KnowledgeGraphEngine.build_user_graph(
        user_id=case["id"],
        user_name=case.get("dignified_identifier", case["id"]),
        emotion_data=emotion_data,
        voice_data=case.get("voice_stress_data"),
        distress_data={"dynamic_distress_score": case.get("dynamic_distress_score")},
        active_interventions=active_interventions
    )
    return kg


@app.post("/api/counsellor/case/{case_id}/intervention/status")
async def update_intervention(case_id: str, req: InterventionUpdateRequest):
    """Updates status of a clinical/social/legal intervention."""
    success = db.update_intervention_status(case_id, req.intervention_name, req.status)
    if not success:
        raise HTTPException(status_code=404, detail="Intervention or Case not found")
    
    await manager.broadcast({
        "event": "INTERVENTION_UPDATED",
        "case_id": case_id,
        "intervention_name": req.intervention_name,
        "status": req.status
    })
    return {"status": "success", "message": f"Updated {req.intervention_name} to {req.status}"}


@app.post("/api/counsellor/case/{case_id}/intervention/add")
async def add_intervention(case_id: str, req: AddInterventionRequest):
    """Dispatches a new intervention."""
    success = db.add_intervention(case_id, req.intervention_name, req.type, req.priority)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    
    await manager.broadcast({
        "event": "INTERVENTION_ADDED",
        "case_id": case_id,
        "intervention_name": req.intervention_name
    })
    return {"status": "success", "message": f"Dispatched {req.intervention_name}"}


@app.post("/api/counsellor/case/{case_id}/notes")
async def update_notes(case_id: str, req: NotesUpdateRequest):
    """Updates formal clinical notes and assessment."""
    success = db.update_counsellor_notes(case_id, req.notes)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"status": "success", "message": "Clinical notes recorded"}


@app.websocket("/ws/counsellor")
async def websocket_counsellor_endpoint(websocket: WebSocket):
    """Real-time live WebSocket stream for Counsellor Dashboard telemetry."""
    await manager.connect(websocket)
    # A just-opened or reconnected dashboard receives the current authoritative
    # state immediately, rather than waiting for the next user action.
    await websocket.send_json({"event": "DASHBOARD_SNAPSHOT", "cases": db.get_all_cases()})
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or heartbeat
            await websocket.send_json({"event": "HEARTBEAT_ACK"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
