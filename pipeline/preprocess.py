"""
Preprocessing: speaker diarization parsing, text cleaning, turn segmentation.
In a real system this would connect to AWS Transcribe or Whisper diarization output.
"""
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Turn:
    turn_id: int
    speaker: str
    text: str
    word_count: int
    tokens: List[str] = field(default_factory=list)

@dataclass  
class Session:
    session_id: int
    patient_id: str
    cohort: str
    session_number: int
    severity_score: int
    turns: List[Turn] = field(default_factory=list)

def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # In production: remove PII with a presidio or AWS Comprehend Medical call
    return text

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def load_and_preprocess(path: str) -> List[Session]:
    with open(path) as f:
        raw = json.load(f)
    
    sessions = []
    for s in raw:
        turns = []
        for t in s["turns"]:
            cleaned = clean_text(t["text"])
            turns.append(Turn(
                turn_id=t["turn_id"],
                speaker=t["speaker"],
                text=cleaned,
                word_count=t["word_count"],
                tokens=tokenize(cleaned),
            ))
        sessions.append(Session(
            session_id=s["session_id"],
            patient_id=s["patient_id"],
            cohort=s["cohort"],
            session_number=s["session_number"],
            severity_score=s["severity_score"],
            turns=turns,
        ))
    return sessions

def patient_turns_only(session: Session) -> List[Turn]:
    return [t for t in session.turns if t.speaker == "patient"]

def full_transcript(session: Session) -> str:
    return " ".join(t.text for t in session.turns)

def patient_transcript(session: Session) -> str:
    return " ".join(t.text for t in patient_turns_only(session))
