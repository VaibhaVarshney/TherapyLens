"""
Topic modeling via TF-IDF + keyword clustering (lightweight demo).
Production: replace with BERTopic or LDA via sklearn/gensim.
"""
import re
from collections import Counter
from typing import Dict, List
from pipeline.preprocess import Session, patient_transcript

TOPIC_SEEDS = {
    "sleep_fatigue":  ["sleep", "tired", "exhausted", "rest", "bed", "night", "awake"],
    "social_support": ["family", "sister", "friend", "talked", "connected", "relationship", "alone"],
    "work_stress":    ["work", "job", "meeting", "boss", "deadline", "office", "career"],
    "coping_skills":  ["breathing", "exercise", "walk", "practice", "noticed", "pause", "journal"],
    "mood_affect":    ["joy", "happy", "sad", "feel", "emotion", "mood", "dark", "light"],
    "self_worth":     ["point", "worthless", "valuable", "should", "failure", "proud", "accomplish"],
}

def extract_topics(text: str) -> Dict[str, float]:
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    scores = {}
    for topic, seeds in TOPIC_SEEDS.items():
        score = sum(word_freq.get(w, 0) for w in seeds)
        scores[topic] = score
    total = sum(scores.values()) or 1
    return {k: round(v / total, 4) for k, v in scores.items()}

def analyze_topics(session: Session) -> Dict:
    pt = patient_transcript(session)
    topics = extract_topics(pt)
    top_topic = max(topics, key=topics.get)
    return {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "cohort": session.cohort,
        "session_number": session.session_number,
        "topics": topics,
        "primary_topic": top_topic,
    }

def topics_all(sessions: List[Session]) -> List[Dict]:
    return [analyze_topics(s) for s in sessions]
