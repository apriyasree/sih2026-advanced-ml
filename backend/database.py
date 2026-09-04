"""
AMMA Data Store & Live Case Repository
Maintains zero inflated or pre-seeded fake data.
All reports, distress scores, trajectories, and interventions are generated
strictly and dynamically from genuine interactions between the user and Amma.
"""

import time
from typing import Dict, List, Any, Optional

def create_fresh_case(case_id: str = "case_active_user", identifier: str = "Live Terminal (Active User)") -> Dict[str, Any]:
    """Initializes a clean, uninflated user session with no fake reports."""
    curr_time = time.strftime("%I:%M %p")
    return {
        "id": case_id,
        "case_ref": f"REF-LIVE-{case_id[-4:].upper()}",
        "dignified_identifier": identifier,
        "priority_category": "Real-Time User Interaction",
        "current_status": "AWAITING_INTERACTION",
        "has_interacted": False,
        "dynamic_distress_score": 0.0,
        "risk_tier": "INTAKE_PENDING",
        "risk_badge": "Awaiting Initial Interaction (Baseline Intake)",
        "alert_triggered": False,
        "escalation_velocity": 0.0,
        "is_escalating": False,
        "escalation_warning": "Session ready. Telemetry and reports will generate dynamically once the user speaks with Amma.",
        "longitudinal_trajectory": [],  # Completely empty until user interacts
        "voice_stress_data": None,      # Strictly None until user uploads real audio
        "xai_breakdown": {},            # Strictly empty until calculated from real chat
        "latest_tone_analysis": None,   # Trained model output; not a diagnosis
        "latest_trauma_signals": None,  # TRACE DSM-5 cluster evidence
        "latest_suds_telemetry": None,  # Thousand Voices of Trauma SUDS & habituation
        "tone_observations": [],
        "legal_guidance": {},
        "chat_history": [
            {
                "sender": "amma",
                "text": "Mera bachha, come sit with Amma. Take a long, gentle breath and let your shoulders drop. Whatever burden you are carrying today, you do not have to carry it alone. Tell me, how is your heart feeling today? Have you eaten something warm, and did you sleep peacefully last night?",
                "time": curr_time
            }
        ],
        # A check-in is kept as a first-class event as well as a transcript item.
        # This makes the clinical log useful after a dashboard refresh/reconnect.
        "latest_checkin": None,
        "event_sequence": 0,
        "symptoms": [],                 # Strictly empty until user mentions symptoms
        "recommended_interventions": [], # Strictly empty until distress warrants it
        "counsellor_notes": ""
    }

class CaseDatabase:
    def __init__(self):
        self.cases: Dict[str, Dict[str, Any]] = {}
        # Demonstration-only local configuration. A real deployment requires
        # authenticated counsellor accounts and encrypted persistent storage.
        self.counsellor_contact: Dict[str, str] = {"name": "", "phone": ""}
        # Start with a single clean, real live session
        self.active_user_case_id = "case_active_user"
        self.cases[self.active_user_case_id] = create_fresh_case(self.active_user_case_id)

    def get_all_cases(self) -> List[Dict[str, Any]]:
        """Returns all cases sorted by Dynamic Distress Score descending."""
        all_cases = list(self.cases.values())
        all_cases.sort(key=lambda c: c.get("dynamic_distress_score", 0), reverse=True)
        return all_cases

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        if case_id not in self.cases:
            self.cases[case_id] = create_fresh_case(case_id, f"Live Terminal ({case_id})")
        return self.cases.get(case_id)

    def reset_case(self, case_id: str = "case_active_user") -> Dict[str, Any]:
        """Resets the case to a clean baseline with zero pre-filled reports."""
        self.cases[case_id] = create_fresh_case(case_id)
        return self.cases[case_id]

    def update_case_with_interaction(
        self,
        case_id: str,
        user_message: str,
        amma_response: str,
        emotion_data: Dict[str, Any],
        voice_data: Optional[Dict[str, Any]],
        distress_data: Dict[str, Any],
        ratings_data: Optional[Dict[str, Any]] = None,
        legal_guidance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        case = self.get_case(case_id)
        curr_time = time.strftime("%I:%M %p")
        
        case["has_interacted"] = True
        case["current_status"] = "ACTIVE_MONITORING"
        case["chat_history"].append({"sender": "user", "text": user_message, "time": curr_time})
        case["chat_history"].append({"sender": "amma", "text": amma_response, "time": curr_time})

        self._apply_distress_data(case, distress_data)
        tone_observation = {
            "time": curr_time,
            "tone": emotion_data.get("primary_tone", "unavailable"),
            "confidence": emotion_data.get("tone_confidence"),
            "support_score": emotion_data.get("text_distress_score"),
            "suds_score": emotion_data.get("suds_telemetry", {}).get("suds_score"),
            "exposure_stage": emotion_data.get("suds_telemetry", {}).get("exposure_stage"),
            "model_version": emotion_data.get("model_version"),
        }
        case["latest_tone_analysis"] = {
            **tone_observation,
            "distribution": emotion_data.get("tone_distribution", {}),
            "suds_telemetry": emotion_data.get("suds_telemetry", {}),
            "not_a_diagnosis": True,
        }
        case["latest_trauma_signals"] = emotion_data.get("trauma_signal", {})
        case["latest_suds_telemetry"] = emotion_data.get("suds_telemetry", {})
        case["tone_observations"].append(tone_observation)
        if legal_guidance is not None:
            case["legal_guidance"] = legal_guidance

        # Append genuine real point to longitudinal trajectory
        traj_index = len(case["longitudinal_trajectory"]) + 1
        primary_emo = emotion_data.get("primary_emotion", "reflective").replace("_", " ").title()
        case["longitudinal_trajectory"].append({
            "day": f"Interaction {traj_index}",
            "score": distress_data["dynamic_distress_score"],
            "notes": f"Primary affect: {primary_emo}"
        })

        case["event_sequence"] += 1

        # Update voice if real audio was recorded
        if voice_data:
            case["voice_stress_data"] = voice_data

        # Detect and append genuine symptoms from user's words
        for emo in emotion_data.get("detected_emotions", []):
            emo_name = emo.replace("_", " ").title()
            if emo_name not in case["symptoms"]:
                case["symptoms"].append(emo_name)

        for flag in emotion_data.get("physiological_flags", []):
            flag_name = flag.replace("_", " ").title()
            if flag_name not in case["symptoms"]:
                case["symptoms"].append(flag_name)

        self._recommend_interventions(case, distress_data["dynamic_distress_score"])
        return case

    def record_daily_checkin(
        self,
        case_id: str,
        ratings_data: Dict[str, Any],
        distress_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persist a user check-in as a clinical transcript and telemetry event."""
        case = self.get_case(case_id)
        curr_time = time.strftime("%I:%M %p")
        rating = ratings_data["feeling_rating"]
        sleep = ratings_data["sleep_quality"]
        meals = ratings_data["meals_eaten"]

        case["has_interacted"] = True
        case["current_status"] = "ACTIVE_MONITORING"
        self._apply_distress_data(case, distress_data)
        case["latest_checkin"] = {**ratings_data, "time": curr_time}
        case["chat_history"].append({
            "sender": "checkin",
            "text": f"Daily check-in — heart rating: {rating}/10; sleep: {sleep}; meals: {meals}.",
            "time": curr_time,
            "ratings": ratings_data.copy()
        })
        case["longitudinal_trajectory"].append({
            "day": f"Check-in {len(case['longitudinal_trajectory']) + 1}",
            "score": distress_data["dynamic_distress_score"],
            "notes": f"Rating {rating}/10, Sleep: {sleep}, Meals: {meals}"
        })
        case["event_sequence"] += 1
        self._recommend_interventions(case, distress_data["dynamic_distress_score"])
        return case

    @staticmethod
    def _apply_distress_data(case: Dict[str, Any], distress_data: Dict[str, Any]) -> None:
        """Keep every counsellor-visible risk field synchronized with the DDS result."""
        case["dynamic_distress_score"] = distress_data["dynamic_distress_score"]
        case["risk_tier"] = distress_data["risk_tier"]
        case["risk_badge"] = distress_data["risk_badge"]
        case["alert_triggered"] = distress_data["alert_triggered"]
        case["escalation_velocity"] = distress_data["escalation_velocity"]
        case["is_escalating"] = distress_data["is_escalating"]
        case["escalation_warning"] = distress_data["escalation_warning"]
        case["xai_breakdown"] = distress_data["xai_breakdown"]

    def _recommend_interventions(self, case: Dict[str, Any], score: float) -> None:
        """Recommend care only when the freshly calculated score warrants it."""
        if score >= 75.0:
            self._ensure_intervention(case, "Emergency Crisis Tele-Counselling", "IMMEDIATE", "counselling")
            self._ensure_intervention(case, "Immediate Safety & Protection Review", "HIGH", "protection")
        elif score >= 55.0:
            self._ensure_intervention(case, "Trauma-Informed Clinical Psychotherapy", "HIGH", "counselling")
            self._ensure_intervention(case, "Medical & Sleep Health Evaluation", "MEDIUM", "medical")
        elif score >= 35.0:
            self._ensure_intervention(case, "Routine Amma Check-in & Emotional Grounding", "LOW", "counselling")

    def _ensure_intervention(self, case: Dict[str, Any], name: str, priority: str, itype: str):
        existing = [i["name"] for i in case["recommended_interventions"]]
        if name not in existing:
            case["recommended_interventions"].insert(0, {
                "name": name,
                "status": "RECOMMENDED",
                "type": itype,
                "priority": priority
            })

    def update_intervention_status(self, case_id: str, intervention_name: str, new_status: str) -> bool:
        case = self.get_case(case_id)
        if not case:
            return False
        for interv in case["recommended_interventions"]:
            if interv["name"] == intervention_name:
                interv["status"] = new_status
                return True
        return False

    def add_intervention(self, case_id: str, intervention_name: str, itype: str, priority: str) -> bool:
        case = self.get_case(case_id)
        if not case:
            return False
        self._ensure_intervention(case, intervention_name, priority, itype)
        return True

    def update_counsellor_notes(self, case_id: str, notes: str) -> bool:
        case = self.get_case(case_id)
        if not case:
            return False
        case["counsellor_notes"] = notes
        return True

    def set_counsellor_contact(self, name: str, phone: str) -> Dict[str, str]:
        self.counsellor_contact = {"name": name.strip(), "phone": phone.strip()}
        return self.counsellor_contact.copy()

    def get_counsellor_contact(self) -> Dict[str, str]:
        return self.counsellor_contact.copy()

# Global database instance initialized with zero fake/inflated reports
db = CaseDatabase()
