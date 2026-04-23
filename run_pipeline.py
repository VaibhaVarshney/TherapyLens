from pipeline.llm_synthesis import synthesize_session
from pipeline.bias_audit import run_audit
from pipeline.topic_modeling import topics_all
from pipeline.engagement import engagement_all
from pipeline.sentiment_emotion import analyze_all
from pipeline.preprocess import load_and_preprocess
from data.synthetic_transcripts import generate_dataset
import sys
import os
import json
import csv
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=== TherapyNLP Pipeline ===\n")

    # 1. Generate synthetic data
    print("[1/5] Generating synthetic transcripts...")
    os.makedirs("data", exist_ok=True)
    data = generate_dataset(n_patients=40)
    with open("data/sessions.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"    {len(data)} sessions generated.\n")

    # 2. Preprocess
    print("[2/5] Preprocessing...")
    sessions = load_and_preprocess("data/sessions.json")
    print(f"    {len(sessions)} sessions loaded.\n")

    # 3. NLP analysis
    print("[3/5] Running NLP analysis (sentiment, emotion, engagement, topics)...")
    sentiment_results = analyze_all(sessions)
    engagement_results = engagement_all(sessions)
    topic_results = topics_all(sessions)

    # Build lookup dicts for LLM synthesis
    sent_by_id = {r["session_id"]: r for r in sentiment_results}
    eng_by_id = {r["session_id"]: r for r in engagement_results}
    top_by_id = {r["session_id"]: r for r in topic_results}

    # 4. LLM synthesis via Groq
    print("[4/5] Running LLM synthesis...")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    synthesis_results = []
    for session in sessions:
        sid = session.session_id
        result = synthesize_session(
            session,
            sent_by_id[sid],
            top_by_id[sid],
            eng_by_id[sid],
            client=groq_client,
        )
        result["session_id"] = sid
        result["patient_id"] = session.patient_id
        result["cohort"] = session.cohort
        synthesis_results.append(result)
    print(f"    {len(synthesis_results)} sessions synthesized.\n")

    # 5. Bias audit
    print("[5/5] Running bias audit...")
    audit = run_audit(sentiment_results)
    print("    Cohort summary:")
    for cohort, info in audit["cohort_summary"].items():
        print(
            f"      {cohort:20s} n={info['n_sessions']:3d}  mean_sentiment={info['mean_sentiment']:+.3f}  dominant={info['dominant_emotion']}")
    print("\n    Disparity flags:")
    for flag in audit["disparity_flags"]:
        print(f"      {flag}")

    # Export outputs
    os.makedirs("outputs", exist_ok=True)

    fieldnames = [
        "session_id", "patient_id", "cohort", "session_number",
        "severity_score", "sentiment_score", "dominant_emotion",
        "primary_topic", "avg_patient_words_per_turn",
        "patient_lexical_diversity", "engagement_level",
        "outcome_risk_flag", "recommended_focus",
    ]
    rows = []
    for i, session in enumerate(sessions):
        sid = session.session_id
        s = sent_by_id[sid]
        e = eng_by_id[sid]
        t = top_by_id[sid]
        syn = synthesis_results[i]
        rows.append({
            "session_id": sid,
            "patient_id": session.patient_id,
            "cohort": session.cohort,
            "session_number": session.session_number,
            "severity_score": session.severity_score,
            "sentiment_score": s["sentiment_score"],
            "dominant_emotion": s["dominant_emotion"],
            "primary_topic": t["primary_topic"],
            "avg_patient_words_per_turn": e["avg_patient_words_per_turn"],
            "patient_lexical_diversity": e["patient_lexical_diversity"],
            "engagement_level": syn["engagement_level"],
            "outcome_risk_flag": syn["outcome_risk_flag"],
            "recommended_focus": syn["recommended_focus"],
        })

    with open("outputs/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open("outputs/bias_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    with open("outputs/synthesis.json", "w") as f:
        json.dump(synthesis_results, f, indent=2)

    print(f"\nOutputs saved to outputs/")
    print("  results.csv     — full dataset for analysis")
    print("  bias_audit.json — cohort disparity report")
    print("  synthesis.json  — LLM session summaries")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
