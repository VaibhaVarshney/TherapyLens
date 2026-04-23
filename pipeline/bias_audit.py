"""
Bias audit: checks whether NLP model outputs show systematic differences
across demographic cohorts. Essential for HIPAA-adjacent AI ethics work.
"""
import json
from typing import Dict, List
from collections import defaultdict

def cohort_summary(sentiment_results: List[Dict]) -> Dict:
    """Compute mean sentiment and dominant emotions per cohort."""
    cohort_data = defaultdict(lambda: {"sentiments": [], "emotions": defaultdict(list)})
    
    for r in sentiment_results:
        c = r["cohort"]
        cohort_data[c]["sentiments"].append(r["sentiment_score"])
        for emo, score in r["emotions"].items():
            cohort_data[c]["emotions"][emo].append(score)

    summary = {}
    for cohort, data in cohort_data.items():
        sents = data["sentiments"]
        mean_sent = round(sum(sents) / len(sents), 4) if sents else 0
        mean_emos = {
            emo: round(sum(scores) / len(scores), 4)
            for emo, scores in data["emotions"].items()
        }
        dominant = max(mean_emos, key=mean_emos.get)
        summary[cohort] = {
            "n_sessions": len(sents),
            "mean_sentiment": mean_sent,
            "mean_emotions": mean_emos,
            "dominant_emotion": dominant,
        }
    return summary

def flag_disparities(summary: Dict, threshold: float = 0.15) -> List[str]:
    """
    Flag cohort pairs where sentiment diverges beyond threshold.
    In production: use formal statistical tests (Mann-Whitney, permutation test).
    """
    cohorts = list(summary.keys())
    flags = []
    for i in range(len(cohorts)):
        for j in range(i + 1, len(cohorts)):
            c1, c2 = cohorts[i], cohorts[j]
            diff = abs(summary[c1]["mean_sentiment"] - summary[c2]["mean_sentiment"])
            if diff > threshold:
                flags.append(
                    f"Sentiment gap between {c1} and {c2}: {diff:.3f} "
                    f"(threshold={threshold}) — warrants review"
                )
    return flags if flags else ["No major disparities detected above threshold."]

def run_audit(sentiment_results: List[Dict]) -> Dict:
    summary = cohort_summary(sentiment_results)
    flags = flag_disparities(summary)
    return {"cohort_summary": summary, "disparity_flags": flags}
