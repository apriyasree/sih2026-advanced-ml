"""
AMMA Semantic Knowledge Graph Engine
Models semantic relations between User Session, Emotional States,
Distress Triggers, Somatic/Physiological Markers, Protective Factors,
and Clinical/Social Interventions for explainable AI in the Counsellor Dashboard.
"""

from typing import Dict, List, Any

class KnowledgeGraphEngine:
    @staticmethod
    def build_user_graph(
        user_id: str,
        user_name: str,
        emotion_data: Dict[str, Any],
        voice_data: Dict[str, Any] = None,
        distress_data: Dict[str, Any] = None,
        active_interventions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Builds an explainable semantic graph with nodes and relationships
        for clinical review on the Counsellor Dashboard.
        """
        nodes = []
        links = []
        
        # 1. Root User Node (Dignified terminal representation)
        nodes.append({
            "id": "node_user",
            "label": f"Session: {user_name}",
            "category": "user",
            "group": 1,
            "val": 25,
            "color": "#4f46e5",
            "details": f"Monitoring Active. Current Distress: {distress_data.get('dynamic_distress_score', 'N/A') if distress_data else 'Evaluating'}"
        })
        
        # 2. Emotional State Nodes (Strictly from real chat)
        detected_emotions = emotion_data.get("detected_emotions", [])
            
        for idx, emo in enumerate(detected_emotions):
            node_id = f"node_emo_{idx}"
            label = emo.replace("_", " ").title()
            color = "#ef4444" if emo in ["fear_terror", "grief_sadness"] else ("#f59e0b" if emo in ["anxiety_hyperarousal", "numbness_dissociation"] else "#10b981")
            nodes.append({
                "id": node_id,
                "label": label,
                "category": "emotion",
                "group": 2,
                "val": 16,
                "color": color,
                "details": f"Affective Signal: {label}. Weight: {emotion_data.get('emotion_breakdown', {}).get(emo, 1.0)}"
            })
            links.append({
                "source": "node_user",
                "target": node_id,
                "relation": "EXHIBITS_AFFECT",
                "weight": 2.5
            })

        # 3. Somatic & Physiological Biomarkers
        flags = emotion_data.get("physiological_flags", [])
        for idx, flag in enumerate(flags):
            node_id = f"node_physio_{idx}"
            label = flag.replace("_", " ").title()
            nodes.append({
                "id": node_id,
                "label": label,
                "category": "somatic",
                "group": 3,
                "val": 14,
                "color": "#f97316",
                "details": f"Somatic Manifestation: {label}. Requires physiological regulation."
            })
            links.append({
                "source": "node_user",
                "target": node_id,
                "relation": "REPORTS_SOMATIC",
                "weight": 2.0
            })

        # A recording may be available for human review, but this prototype has
        # no validated acoustic stress score.  Represent its availability only;
        # do not fabricate a symptom or an acoustic-risk edge.
        if voice_data:
            nodes.append({
                "id": "node_voice_recording",
                "label": "Voice recording submitted (unscored)",
                "category": "acoustic",
                "group": 3,
                "val": 15,
                "color": "#64748b",
                "details": voice_data.get("analysis_note", "Recording available for authorised human review.")
            })
            links.append({
                "source": "node_user",
                "target": "node_voice_recording",
                "relation": "SUBMITTED_RECORDING",
                "weight": 1.0
            })

        # 4. Distress Triggers & External Stressors
        triggers = []
        if "fear_terror" in detected_emotions:
            triggers.append(("Hostility & Intimidation Stress", "Environmental threat exposure"))
        if "sleep_disruption" in flags:
            triggers.append(("Nocturnal Panic & Intrusive Memories", "Hyper-arousal cycle"))
        if "grief_sadness" in detected_emotions:
            triggers.append(("Traumatic Loss & Alienation", "Social disconnect"))
            
        for idx, (trig_name, trig_desc) in enumerate(triggers):
            node_id = f"node_trig_{idx}"
            nodes.append({
                "id": node_id,
                "label": trig_name,
                "category": "trigger",
                "group": 4,
                "val": 15,
                "color": "#dc2626",
                "details": trig_desc
            })
            if detected_emotions:
                links.append({
                    "source": node_id,
                    "target": "node_emo_0",
                    "relation": "TRIGGERS",
                    "weight": 2.0
                })
            links.append({
                "source": "node_user",
                "target": node_id,
                "relation": "VULNERABLE_TO",
                "weight": 1.5
            })

        # 5. Protective Factors & Calming Anchors (AMMA)
        protective_factors = [
            ("Maternal Presence (AMMA)", "De-escalation & Safe Grounding Anchor", "#059669")
        ]
        for idx, (pf_name, pf_desc, pf_color) in enumerate(protective_factors):
            node_id = f"node_pf_{idx}"
            nodes.append({
                "id": node_id,
                "label": pf_name,
                "category": "protective",
                "group": 5,
                "val": 16,
                "color": pf_color,
                "details": pf_desc
            })
            links.append({
                "source": "node_user",
                "target": node_id,
                "relation": "ANCHORED_BY",
                "weight": 2.8
            })

        # 6. Active Clinical / Social Interventions (Strictly from real database)
        interventions = active_interventions if active_interventions else []
        for idx, interv in enumerate(interventions):
            node_id = f"node_interv_{idx}"
            nodes.append({
                "id": node_id,
                "label": interv,
                "category": "intervention",
                "group": 6,
                "val": 17,
                "color": "#2563eb",
                "details": f"Multi-Agency Intervention: {interv}"
            })
            links.append({
                "source": node_id,
                "target": "node_user",
                "relation": "SUPPORTS_REHABILITATION",
                "weight": 3.0
            })
            # Connect to trigger mitigation
            if triggers:
                links.append({
                    "source": node_id,
                    "target": "node_trig_0",
                    "relation": "MITIGATES_RISK",
                    "weight": 2.0
                })

        return {
            "nodes": nodes,
            "links": links,
            "node_count": len(nodes),
            "link_count": len(links)
        }
