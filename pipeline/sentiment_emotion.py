"""
Sentiment & emotion analysis.
Uses a lightweight rule-based fallback so the project runs without GPU.
In production: swap in transformers pipeline with roberta-base-go_emotions.
"""
import re
from typing import Dict, List
from pipeline.preprocess import Session, patient_transcript

# Lightweight keyword lexicon (production: replace with HuggingFace model)
EMOTION_LEXICON = {
    "anxiety":    ["overwhelmed", "worried", "racing", "anxious", "nervous", "uncertain", "heart", "scared", "panic"],
    "sadness":    ["don't see the point", "disconnected", "joy", "crying", "sad", "hopeless", "lonely", "isolated"],
    "depression": ["bed", "canceling", "no energy", "empty", "numb", "worthless"],
    "progress":   ["helped", "walk", "good week", "lighter", "practiced", "noticed", "talked"],
    "neutral":    ["day by day", "busy", "reading", "family", "work"],
}

SENTIMENT_POSITIVE_WORDS = {"helped", "good", "better", "lighter", "walked", "talked", "noticed", "courage", "effort"}
SENTIMENT_NEGATIVE_WORDS = {"overwhelmed", "worried", "difficult", "hard", "sad", "anxious", "disconnected", "hopeless", "alone"}

def score_emotions(text: str) -> Dict[str, float]:
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = round(hits / max(len(keywords), 1), 4)
    total = sum(scores.values()) or 1
    return {k: round(v / total, 4) for k, v in scores.items()}

def score_sentiment(text: str) -> float:
    """Returns -1 (negative) to +1 (positive)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(words & SENTIMENT_POSITIVE_WORDS)
    neg = len(words & SENTIMENT_NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 4)

def analyze_session(session: Session) -> Dict:
    pt = patient_transcript(session)
    emotions = score_emotions(pt)
    dominant = max(emotions, key=emotions.get)
    return {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "cohort": session.cohort,
        "session_number": session.session_number,
        "severity_score": session.severity_score,
        "sentiment_score": score_sentiment(pt),
        "emotions": emotions,
        "dominant_emotion": dominant,
    }

def analyze_all(sessions: List[Session]) -> List[Dict]:
    return [analyze_session(s) for s in sessions]
