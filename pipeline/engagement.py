"""
Engagement signal extraction.
Measures patient participation depth, therapist question rate, turn balance.
"""
from typing import Dict, List
from pipeline.preprocess import Session

def compute_engagement(session: Session) -> Dict:
    patient_turns = [t for t in session.turns if t.speaker == "patient"]
    therapist_turns = [t for t in session.turns if t.speaker == "therapist"]

    avg_patient_words = (
        sum(t.word_count for t in patient_turns) / len(patient_turns)
        if patient_turns else 0
    )
    therapist_question_rate = (
        sum(1 for t in therapist_turns if "?" in t.text) / len(therapist_turns)
        if therapist_turns else 0
    )
    # Lexical diversity: unique tokens / total tokens (Type-Token Ratio)
    all_patient_tokens = [tok for t in patient_turns for tok in t.tokens]
    ttr = len(set(all_patient_tokens)) / max(len(all_patient_tokens), 1)

    turn_balance = len(patient_turns) / max(len(session.turns), 1)

    return {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "cohort": session.cohort,
        "session_number": session.session_number,
        "avg_patient_words_per_turn": round(avg_patient_words, 2),
        "therapist_question_rate": round(therapist_question_rate, 4),
        "patient_lexical_diversity": round(ttr, 4),
        "turn_balance": round(turn_balance, 4),
        "total_turns": len(session.turns),
    }

def engagement_all(sessions: List[Session]) -> List[Dict]:
    return [compute_engagement(s) for s in sessions]
