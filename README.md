# 🧠 TherapyLens - Clinical NLP Pipeline for Therapy Transcript Analysis

> An end-to-end NLP pipeline that analyzes de-identified therapy transcripts using emotion detection, topic modeling, engagement signal extraction, LLM-powered clinical summarization, and demographic bias auditing - deployed as an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070b-orange?style=flat-square)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Overview

TherapyLens is a research-grade NLP system designed to surface clinically meaningful signals from therapy session transcripts. It processes transcripts through a modular 6-stage pipeline and visualizes results in an interactive dashboard - with a live inference page where users can input session data and receive real-time LLM-generated clinical summaries.

**Built as a portfolio project demonstrating applied NLP in mental health - synthetic data only, with a clear architecture for production deployment under HIPAA-compliant conditions.**

---

## 🎯 What It Does

| Stage | Module | Output |
|---|---|---|
| Data Generation | `data/synthetic_transcripts.py` | 140–165 sessions across 40 patients, 4 cohorts |
| Preprocessing | `pipeline/preprocess.py` | Structured `Session` + `Turn` dataclasses |
| Sentiment & Emotion | `pipeline/sentiment_emotion.py` | Sentiment score (-1 to +1), 5-class emotion distribution |
| Topic Modeling | `pipeline/topic_modeling.py` | 6-category topic distribution + primary topic |
| Engagement Signals | `pipeline/engagement.py` | TTR, avg words/turn, therapist question rate, turn balance |
| LLM Synthesis | `pipeline/llm_synthesis.py` | Clinical summary, risk flag, engagement level, recommended focus |
| Bias Audit | `pipeline/bias_audit.py` | Cohort disparity report, flagged sentiment gaps |

---

## 🖥️ Live Dashboard

The Streamlit dashboard includes 7 pages:

- **📊 Overview** - KPIs, session distribution, risk breakdown, engagement by cohort
- **😔 Sentiment & Emotion** - Cohort sentiment comparisons, emotion radar charts, distribution plots
- **⚠️ Bias Audit** - Disparity flags, cohort summary table, emotion heatmap
- **📝 Session Summaries** - Filterable LLM-generated clinical summaries per session
- **📈 Patient Timeline** - Per-patient longitudinal view of sentiment, severity, and engagement
- **🗂️ Topic Distribution** - Topic frequency, cohort breakdown, topic × risk heatmap
- **🧪 Live Session Analysis** - Real-time form input → NLP analysis + Groq LLM inference

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/VaibhaVarshney/TherapyLens.git
cd TherapyLens
```

### 2. Create and activate virtual environment
```bash
python -m venv nlp_venv
# Windows
nlp_venv\Scripts\activate
# macOS/Linux
source nlp_venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install python-dotenv
```

### 4. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the pipeline
```bash
python run_pipeline.py
```

Outputs are saved to `outputs/`:
- `results.csv` - full dataset (12 features × ~158 rows)
- `bias_audit.json` - cohort disparity report
- `synthesis.json` - LLM-generated session summaries

### 6. Launch the dashboard
```bash
pip install streamlit plotly
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
TherapyLens/
├── data/
│   ├── __init__.py
│   └── synthetic_transcripts.py    # Generates synthetic therapy sessions
├── pipeline/
│   ├── __init__.py
│   ├── preprocess.py               # Text cleaning, turn segmentation, dataclasses
│   ├── sentiment_emotion.py        # Lexicon-based emotion + sentiment scoring
│   ├── engagement.py               # TTR, words/turn, question rate, turn balance
│   ├── topic_modeling.py           # TF-IDF keyword topic extraction
│   ├── llm_synthesis.py            # Groq LLaMA-3.3-70b session summarization
│   └── bias_audit.py               # Cohort disparity detection
├── dashboard/
│   └── app.py                      # Streamlit dashboard (7 pages)
├── outputs/
│   ├── results.csv
│   ├── bias_audit.json
│   └── synthesis.json
├── run_pipeline.py                  # End-to-end pipeline runner
├── requirements.txt
├── .env                            # (not committed) API keys
├── .gitignore
└── README.md
```

---

## 🔬 NLP Pipeline - Technical Details

### Sentiment & Emotion Detection
Lexicon-based scoring over patient dialogue using keyword matching across 5 emotion categories. Sentiment scored as `(positive_hits - negative_hits) / total_hits` yielding a [-1, +1] range.

**Production upgrade:** Replace with [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) for 28-class transformer-level emotion classification:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)
```

### Topic Modeling
Seed-word frequency scoring across 6 clinical topic domains normalized to a topic distribution per session.

**Production upgrade:** Replace with BERTopic for fully unsupervised semantic topic discovery:
```python
from bertopic import BERTopic
model = BERTopic(language="english", calculate_probabilities=True)
topics, probs = model.fit_transform(transcripts)
```

### Engagement Signals
| Signal | Formula | Clinical Meaning |
|---|---|---|
| Avg words/turn | total patient words / patient turns | Verbosity, openness |
| Type-Token Ratio | unique tokens / total tokens | Lexical diversity, cognitive flexibility |
| Therapist question rate | questions / therapist turns | Therapist driving vs patient-led |
| Turn balance | patient turns / total turns | Dialogue equity |

### LLM Synthesis
Structured prompt engineering with Groq LLaMA-3.3-70b. Each session prompt includes cohort metadata, PHQ-9 severity score, NLP signals, and an 800-character transcript excerpt. Response is forced to a strict JSON schema with 5 clinical fields. Dual-layer parsing handles markdown code fences and JSON decode errors.

### Bias Audit
Computes mean sentiment per cohort and flags pairs exceeding a 0.15 disparity threshold. 

**Finding:** Veteran vs adult general sentiment gap = **0.172** - flagged for clinical review.

**Production upgrade:** Add Welch's t-test with Bonferroni correction for statistical significance, effect size reporting, and intersectional cohort analysis.

---

## 🔒 Ethics & Clinical Considerations

- **Synthetic data only** - no real PHI at any stage. All transcripts are programmatically generated.
- **HIPAA pathway documented** - production deployment requires PII anonymization via [Microsoft Presidio](https://microsoft.github.io/presidio/) before any text reaches an NLP model
- **IRB requirement acknowledged** - any use of real therapy transcripts requires Institutional Review Board approval
- **LLM outputs are decision-support only** - clinical summaries generated by LLaMA-3.3-70b must be reviewed by a licensed clinician before informing any patient care decision
- **Bias auditing built-in** - cohort disparity flags are a first-class output of the pipeline, not an afterthought

---

## 🔧 Swapping to Production Models

Every lightweight component has a documented production upgrade path:

```python
# sentiment_emotion.py → transformer model
from transformers import pipeline
classifier = pipeline("text-classification",
                      model="SamLowe/roberta-base-go_emotions", top_k=5)

# topic_modeling.py → BERTopic
from bertopic import BERTopic
model = BERTopic(language="english", calculate_probabilities=True)

# llm_synthesis.py → Anthropic Claude (if budget allows)
import anthropic
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
result = synthesize_session(session, sentiment, topics, engagement, client=client)

# preprocess.py → Presidio PII anonymization
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
```

---

## 📊 Sample Output

**Bias Audit finding:**
```json
{
  "disparity_flags": [
    "Sentiment gap between veteran and adult_general: 0.172 (threshold=0.15) - warrants review"
  ],
  "cohort_summary": {
    "veteran":      { "n_sessions": 46, "mean_sentiment": -0.401, "dominant_emotion": "neutral" },
    "teen":         { "n_sessions": 29, "mean_sentiment": -0.287, "dominant_emotion": "neutral" },
    "postpartum":   { "n_sessions": 59, "mean_sentiment": -0.272, "dominant_emotion": "neutral" },
    "adult_general":{ "n_sessions": 24, "mean_sentiment": -0.229, "dominant_emotion": "neutral" }
  }
}
```

**LLM Session Summary (sample):**
```json
{
  "session_summary": "Patient presents with moderate anxiety and sleep disturbance, showing limited engagement with coping strategies introduced in prior sessions. Affect is flat with restricted emotional range.",
  "engagement_level": "medium",
  "outcome_risk_flag": "moderate",
  "recommended_focus": "Sleep hygiene and behavioral activation",
  "notable_linguistic_patterns": [
    "Repetitive use of uncertainty markers ('I don't know', 'maybe')",
    "Low lexical diversity suggesting cognitive rumination"
  ]
}
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| LLM | Groq LLaMA-3.3-70b-versatile |
| NLP | HuggingFace Transformers, BERTopic, scikit-learn |
| Data | pandas, numpy |
| Visualization | Plotly, Streamlit |
| Privacy | Microsoft Presidio |
| DevOps | Git, GitHub, Streamlit Community Cloud |

---

## 👤 Author

**Vaibha Varshney**
Masters Student | NLP & Clinical AI
[GitHub](https://github.com/VaibhaVarshney) · [LinkedIn](#)

---

## 📄 License

MIT License - free to use, modify, and distribute with attribution.

---

*Built with a focus on responsible AI in mental health - because the people in these transcripts deserve systems that are accurate, fair, and transparent.*
