"""
AMMA Dynamic Distress Score (DDS) & Longitudinal Trend Engine
Predicts escalation velocity, generates Explainable AI (XAI) factor breakdowns,
and triggers counsellor crisis alerts based on risk thresholds.
"""

from typing import Dict, List, Any, Tuple
import time

class DynamicDistressEngine:
    # Product thresholds, not clinical diagnoses or validated risk cut-offs.
    THRESHOLD_STABLE = 30.0
    THRESHOLD_MODERATE = 55.0
    THRESHOLD_ELEVATED = 75.0
    
    @staticmethod
    def calculate_distress_score(
        text_emotion_data: Dict[str, Any],
        voice_stress_data: Dict[str, Any] = None,
        ratings_data: Dict[str, Any] = None,
        historical_scores: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates the multi-modal Dynamic Distress Score (0 - 100),
        generates Explainable AI (XAI) factor attributions,
        and computes longitudinal escalation velocity.
        """
        # 1. Text Emotion Risk Component (0 - 100)
        text_score = text_emotion_data.get("text_distress_score", 25.0)
        
        # 2. Voice component. It is used only if a future, explicitly validated
        # model supplies a numeric score. The bundled waveform heuristics do not.
        supplied_voice_score = (voice_stress_data or {}).get("voice_stress_index")
        if isinstance(supplied_voice_score, (int, float)):
            voice_score = max(0.0, min(100.0, float(supplied_voice_score)))
            voice_available = True
        else:
            voice_score = 0.0
            voice_available = False
            
        # 3. Daily Ratings & Physiological Deficit Component
        # User feeling rating: 1 (very bad) to 10 (very good)
        # Convert to distress: rating 1 = 90 distress, rating 10 = 10 distress
        feel_rating = 5
        sleep_deficit = 25.0
        meal_deficit = 25.0
        
        if ratings_data:
            user_rating = ratings_data.get("feeling_rating", 5)
            user_rating = max(1, min(10, float(user_rating)))
            # Invert 1-10 to distress scale
            feel_score = max(5.0, min(95.0, (10 - user_rating) * 10.0))
            
            # Sleep hours & quality
            sleep_quality = ratings_data.get("sleep_quality", "fair")  # poor, fair, good
            if sleep_quality == "poor" or "sleep_disruption" in text_emotion_data.get("physiological_flags", []):
                sleep_deficit = 85.0
            elif sleep_quality == "fair":
                sleep_deficit = 45.0
            else:
                sleep_deficit = 15.0
                
            # Meals eaten
            meals = ratings_data.get("meals_eaten", "regular")  # skipped, irregular, regular
            if meals == "skipped" or "appetite_loss" in text_emotion_data.get("physiological_flags", []):
                meal_deficit = 80.0
            elif meals == "irregular":
                meal_deficit = 50.0
            else:
                meal_deficit = 15.0
        else:
            feel_score = text_score
            if "sleep_disruption" in text_emotion_data.get("physiological_flags", []):
                sleep_deficit = 80.0
            if "appetite_loss" in text_emotion_data.get("physiological_flags", []):
                meal_deficit = 75.0

        # Weighted fusion calculation
        # The original fallback reused the text score as a faux voice score,
        # silently double-counting one source. Redistribute weight only across
        # observed inputs instead. These weights are product heuristics, not
        # learnt parameters or clinical calibration.
        if voice_available:
            w_text, w_voice, w_feel, w_sleep, w_meal = 0.30, 0.25, 0.20, 0.15, 0.10
        else:
            w_text, w_voice, w_feel, w_sleep, w_meal = 0.40, 0.00, 0.30, 0.20, 0.10

        raw_dds = (
            text_score * w_text +
            voice_score * w_voice +
            feel_score * w_feel +
            sleep_deficit * w_sleep +
            meal_deficit * w_meal
        )
        
        dynamic_distress_score = round(max(5.0, min(99.0, raw_dds)), 1)
        
        # Determine Clinical Risk Tier
        if dynamic_distress_score >= DynamicDistressEngine.THRESHOLD_ELEVATED:
            risk_tier = "CRITICAL_RISK"
            risk_badge = "High support signal (prompt human review recommended)"
            alert_triggered = True
        elif dynamic_distress_score >= DynamicDistressEngine.THRESHOLD_MODERATE:
            risk_tier = "ELEVATED_RISK"
            risk_badge = "Elevated support signal (human review recommended)"
            alert_triggered = True
        elif dynamic_distress_score >= DynamicDistressEngine.THRESHOLD_STABLE:
            risk_tier = "MODERATE_STABLE"
            risk_badge = "Moderate support signal (follow-up may help)"
            alert_triggered = False
        else:
            risk_tier = "LOW_STABLE"
            risk_badge = "Low support signal in this interaction"
            alert_triggered = False

        # Explainable AI (XAI) Attribution Breakdown
        # Calculates relative percentage contribution to the distress score
        contributions = {
            "Linguistic & Emotional Expression": round((text_score * w_text / max(1.0, raw_dds)) * 100.0, 1),
            "Acoustic Vocal Biomarkers (PTSD-in-the-Wild DSP)": round((voice_score * w_voice / max(1.0, raw_dds)) * 100.0, 1),
            "Self-Reported Emotional State": round((feel_score * w_feel / max(1.0, raw_dds)) * 100.0, 1),
            "Sleep Architecture Disruption": round((sleep_deficit * w_sleep / max(1.0, raw_dds)) * 100.0, 1),
            "Nutritional & Care Neglect": round((meal_deficit * w_meal / max(1.0, raw_dds)) * 100.0, 1)
        }
        # Normalize sum to 100%
        c_sum = sum(contributions.values())
        if c_sum > 0:
            for k in contributions:
                contributions[k] = round((contributions[k] / c_sum) * 100.0, 1)
            # Keep the displayed explanation total exact after rounding.
            last_key = next(reversed(contributions))
            contributions[last_key] = round(100.0 - sum(
                value for key, value in contributions.items() if key != last_key
            ), 1)

        # Longitudinal Trend & Escalation Velocity (Strictly from real interactions)
        if historical_scores and len(historical_scores) > 0:
            full_trajectory = historical_scores + [dynamic_distress_score]
            recent_avg = sum(full_trajectory[-3:-1]) / max(1, len(full_trajectory[-3:-1]))
            escalation_velocity = round(dynamic_distress_score - recent_avg, 2)
            is_escalating = escalation_velocity >= 8.0 or dynamic_distress_score >= 70.0

            if is_escalating and dynamic_distress_score >= 60.0:
                escalation_warning = "CRISIS ESCALATION WARNING: Rapid distress surge detected (+{:.1f} pts). Pre-crisis threshold exceeded.".format(escalation_velocity)
            elif is_escalating:
                escalation_warning = "Upward distress trajectory detected (+{:.1f} pts). Monitoring recommended.".format(escalation_velocity)
            elif escalation_velocity <= -5.0:
                escalation_warning = "De-escalation trajectory observed (-{:.1f} pts). Coping techniques proving effective.".format(abs(escalation_velocity))
            else:
                escalation_warning = "Distress trajectory stable within baseline parameters."
        else:
            escalation_velocity = 0.0
            is_escalating = False
            escalation_warning = "Initial interaction recorded. Trajectory will track escalation velocity across subsequent chats."


        return {
            "dynamic_distress_score": dynamic_distress_score,
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "alert_triggered": alert_triggered,
            "escalation_velocity": escalation_velocity,
            "is_escalating": is_escalating,
            "escalation_warning": escalation_warning,
            "xai_breakdown": contributions,
            "voice_analyzed": voice_available,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
