# AMMA: AI-Powered Mental Health Monitoring Assistance
### Support-signal monitoring prototype for local demonstration only

---

## 🌟 Executive Overview
**AMMA** (*AI-Powered Mental health Monitoring Assistance*) is a dual-space supportive-chat and wellbeing-monitoring prototype. It includes a trained supervised machine-learning classifier for the *expressed tone of chat messages*. It is not a diagnostic system, emergency service, clinical risk-prediction model, or real multi-agency dispatcher.

The system resolves the critical ethical and clinical dilemmas in monitoring vulnerable populations:
1. **User Terminal ("AMMA")**: Provides a warm, maternal, deeply empathetic AI companion speaking with unconditional kindness, soothing emotional support, and guided grounding. It avoids clinical pathologization, never uses stigmatizing terms (such as *"victim"* or *"rape survivor"*), and presents zero frightening clinical alarms or triage labels. Everyday citizens and high-risk individuals alike can comfortably speak with Amma.
2. **Counsellor Support Dashboard**: A real-time human-review workspace with a model-based tone observation, longitudinal support-signal tracking, listenable voice recordings with descriptive waveform summaries, legal-information routing, printable non-diagnostic reports, and locally tracked support actions.

---

## 🚀 Key Innovations & Components

| Component | Technical Implementation | Purpose & Outcome |
| :--- | :--- | :--- |
| **Supervised chat-tone model** | Multinomial Naive Bayes trained at application startup on 48 independently labelled, auditable support-tone examples using word and bigram features. | Classifies expressed tone and confidence for human review; it does not diagnose a condition, determine intent, or predict risk. |
| **Safety safeguard** | Conservative immediate-safety phrase screen kept separate from the ML model. | Makes urgent routing deterministic and avoids relying on model confidence for an emergency message. |
| **Audio handling** | Stores WAV/WebM; PCM WAV can show descriptive waveform summaries. | No voice-stress or clinical biomarker inference is made. |
| **Dynamic Distress Score (DDS)** | Hand-authored fusion of text, self-rating, sleep and meals. | A non-clinical product support signal, not an objective risk index. |
| **Observed trend** | Difference from up to two preceding interaction/check-in scores. | Displays recorded change only; it does not forecast crises. |
| **Support-signal breakdown** | Displays the observed text, self-rating, sleep and meal inputs to the product support signal. | Arithmetic transparency, not clinical calibration or a diagnosis. |
| **Legal-support routing** | Routes general India-specific options such as 112, Tele-MANAS, NALSA 15100, DLSA, and topic-specific next steps. | General information only; it does not give legal advice or initiate a complaint. |
| **Escalation contact** | Counsellor saves an optional phone number in the dashboard. | Offered to the user only at the maximum support tier and never auto-called or shared externally. |
| **Indian-Accented Maternal TTS** | Client-side speech synthesis configured for Indian English cadence, gentle motherly pitch (1.05), and unhurried pacing (0.88). | Empowers users with soothing auditory reassurance, mimicking the comforting presence of a mother. |

---

## 🛡️ Priority Use Cases Supported

1. **Witnesses Facing Intimidation or Threats**: Continuous monitoring of sudden fear surges, nocturnal panic, and vocal tremors. Integrates direct dispatch for Witness Protection and Police Liaison.
2. **Victims of Severe Physical Harm & Trauma**: Long-term recovery tracking with trauma-informed psychotherapy, somatic stabilization, and DLSA financial relief claims.
3. **Families Affected by Caste-Based Hostility**: Social boycott and community intimidation monitoring, coordinated with legal aid under the SC/ST (Prevention of Atrocities) Act.
4. **General Citizens & Routine Wellness**: Normal individuals seeking daily comfort, sleep hygiene check-ins, and maternal care without any pathologizing labels.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, WebSockets, NumPy, SciPy (Signal DSP), Pydantic.
- **Frontend**: Responsive Single-Page Applications (Vanilla JS + CSS3 Modern Design System + HTML5 Web Audio API + Web Speech API + Canvas Physics Graph).
- **Visualization**: Chart.js 4.4+ (Longitudinal trend curves, XAI attribution), Canvas Force-Directed Knowledge Graph.

---

## 🚦 Quick Start & Execution

### 1. Install, activate and run
```bash
# In project root:
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\activate

# Launch system server:
python run.py
```

### 2. Access Dashboards
- **User Sanctuary Terminal (AMMA)**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Counsellor Clinical Command Dashboard**: [http://127.0.0.1:8000/counsellor](http://127.0.0.1:8000/counsellor)

### 3. Run Automated Verification Tests
```bash
python tests/test_system.py
```

---

## ML model and demonstration statement

You may accurately describe AMMA as: **"a prototype with a trained supervised ML chat-tone classifier (Multinomial Naive Bayes) that produces non-diagnostic, human-review support reports."**

The model learns class priors and word/bigram likelihoods from the auditable corpus in `backend/tone_classifier.py` at startup. The supplied research PDFs informed the supported themes and safeguards; they were not used as labelled chat data and therefore are not represented as a clinical training dataset. Before a real-world deployment, replace the seed corpus with consented, de-identified, independently labelled data; evaluate representative Indian languages; obtain clinical/legal governance; and add authentication, encryption and persistent storage.
