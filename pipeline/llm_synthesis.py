"""
LLM synthesis layer: generates clinical session summaries using an LLM.
Uses the Groq API. In production this would run on de-identified data only.
Requires: pip install groq
"""
from groq import Groq
import json
from typing import Dict
from pipeline.preprocess import Session, full_transcript


def build_prompt(session: Session, sentiment: Dict, topics: Dict, engagement: Dict) -> str:
    transcript = full_transcript(session)
    return f"""You are a clinical NLP assistant. Analyze this de-identified therapy session.

SESSION METADATA:
- Cohort: {session.cohort}
- Session number: {session.session_number}
- Severity score (PHQ-9 proxy): {session.severity_score}/27

NLP SIGNALS:
- Dominant emotion: {sentiment['dominant_emotion']}
- Sentiment score: {sentiment['sentiment_score']} (range -1 to +1)
- Primary topic: {topics['primary_topic']}
- Patient avg words/turn: {engagement['avg_patient_words_per_turn']}
- Patient lexical diversity (TTR): {engagement['patient_lexical_diversity']}

TRANSCRIPT EXCERPT:
{transcript[:800]}...

Respond ONLY with valid JSON matching this schema:
{{
  "session_summary": "<2-3 sentence clinical summary>",
  "engagement_level": "low|medium|high",
  "outcome_risk_flag": "low|moderate|high",
  "recommended_focus": "<one clinical focus area for next session>",
  "notable_linguistic_patterns": ["<pattern1>", "<pattern2>"]
}}"""


def synthesize_session(
    session: Session,
    sentiment: Dict,
    topics: Dict,
    engagement: Dict,
    client=None,
) -> Dict:
    if client is None:
        # Return a mock response for demo without API key
        return {
            "session_summary": f"Session {session.session_number} for {session.cohort} patient. "
            f"Dominant emotion: {sentiment['dominant_emotion']}. "
            f"Engagement appears {'high' if engagement['avg_patient_words_per_turn'] > 15 else 'moderate'}.",
            "engagement_level": "high" if engagement["avg_patient_words_per_turn"] > 15 else "medium",
            "outcome_risk_flag": "moderate" if session.severity_score > 14 else "low",
            "recommended_focus": topics["primary_topic"].replace("_", " ").title(),
            "notable_linguistic_patterns": [
                f"Emotional tone: {sentiment['dominant_emotion']}",
                f"Lexical diversity TTR: {engagement['patient_lexical_diversity']:.2f}",
            ],
        }

    prompt = build_prompt(session, sentiment, topics,
                          engagement)  # ← fix: was missing

    message = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    raw = message.choices[0].message.content.strip()

    # Strip markdown code fences if model wraps response in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "session_summary": "Could not parse LLM response.",
            "engagement_level": "medium",
            "outcome_risk_flag": "moderate",
            "recommended_focus": "General support",
            "notable_linguistic_patterns": [],
        }
