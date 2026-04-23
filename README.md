# TherapyNLP: Therapeutic Conversation Analysis Pipeline

An end-to-end NLP framework for analyzing de-identified therapy transcripts.
Identifies linguistic patterns associated with patient engagement, emotional state,
and clinical outcomes across demographic cohorts.

## What it does

- **Preprocessing** — speaker diarization parsing, text cleaning, anonymization hooks
- **Sentiment & emotion** — per-session scoring (production: RoBERTa / GoEmotions)
- **Topic modeling** — keyword extraction and thematic clustering (production: BERTopic)
- **Engagement signals** — turn length, lexical diversity (TTR), therapist question rate
- **LLM synthesis** — Claude-powered session summaries, engagement level, outcome risk flags
- **Bias audit** — cohort-level disparity detection across veteran, teen, postpartum, adult populations

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Outputs land in `outputs/`:
- `results.csv` — flat dataset for statistical analysis
- `bias_audit.json` — cohort disparity report
- `synthesis.json` — LLM-generated session summaries

## Swapping in production models

In `pipeline/sentiment_emotion.py`, replace `score_emotions()` with:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)
```

In `pipeline/topic_modeling.py`, replace `extract_topics()` with BERTopic:
```python
from bertopic import BERTopic
model = BERTopic(language="english", calculate_probabilities=True)
```

In `pipeline/llm_synthesis.py`, pass a live client:
```python
import anthropic
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
result = synthesize_session(session, sentiment, topics, engagement, client=client)
```

## Ethics & bias mitigation

- All transcripts are synthetic — no real PHI
- Bias audit flags cohort-level sentiment disparities > 0.15 SD
- Production system should use Presidio for PII anonymization before any LLM call
- Cohort disparity flags should trigger manual clinical review

## Research outputs

Designed to produce manuscript-ready tables (CSV export) compatible with
JMIR or Nature Digital Medicine submission standards.

## Project structure

```
therapynlp/
├── data/synthetic_transcripts.py   # Synthetic data generator
├── pipeline/
│   ├── preprocess.py               # Text cleaning, session parsing
│   ├── sentiment_emotion.py        # Emotion & sentiment scoring
│   ├── engagement.py               # Engagement signal extraction
│   ├── topic_modeling.py           # Topic extraction
│   ├── llm_synthesis.py            # Claude-powered session summaries
│   └── bias_audit.py               # Cohort disparity analysis
├── run_pipeline.py                  # End-to-end runner
├── requirements.txt
└── README.md
```
