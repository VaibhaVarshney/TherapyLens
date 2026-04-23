"""
Generates synthetic therapy conversation transcripts.
No real PHI — safe for demo, portfolio, and research.
"""
import random
import json
from datetime import datetime, timedelta

PATIENT_UTTERANCES = {
    "anxiety": [
        "I've been feeling really overwhelmed lately, like I can't catch a breath.",
        "My heart races when I think about the meeting tomorrow.",
        "I keep worrying about things that haven't happened yet.",
        "I couldn't sleep again last night, just lying there with my thoughts.",
        "It's hard to concentrate when everything feels so uncertain.",
    ],
    "depression": [
        "I just don't see the point in a lot of things anymore.",
        "Getting out of bed has been really hard this week.",
        "I feel disconnected from people I used to be close to.",
        "Nothing really brings me joy the way it used to.",
        "I've been canceling plans because I just don't have the energy.",
    ],
    "progress": [
        "Actually, I tried that breathing exercise you suggested and it helped.",
        "I had a good week — I went for a walk three days in a row.",
        "I talked to my sister about how I was feeling and it went well.",
        "I noticed when I was getting anxious and I paused, like we practiced.",
        "Things feel a little lighter this week.",
    ],
    "neutral": [
        "I've just been taking things day by day.",
        "Work has been busy but nothing unusual.",
        "I spent some time with family this weekend.",
        "I've been reading more, which has been nice.",
    ],
}

THERAPIST_UTTERANCES = [
    "That sounds really difficult. Can you tell me more about when that feeling comes up?",
    "I hear you. What do you think is driving that for you?",
    "It takes courage to try something new. How did it feel in the moment?",
    "Let's slow down there — what was going through your mind?",
    "It sounds like you've made a real effort this week.",
    "What would it look like if things were even slightly better?",
    "I notice you said 'should' — where does that pressure come from?",
    "That's an important insight. How does your body feel when that happens?",
]

COHORTS = ["veteran", "teen", "adult_general", "postpartum"]

def generate_session(session_id, patient_id, cohort, session_num):
    n_turns = random.randint(8, 16)
    turns = []
    theme = random.choices(
        ["anxiety", "depression", "progress", "neutral"],
        weights=[0.3, 0.3, 0.25, 0.15]
    )[0]

    for i in range(n_turns):
        if i % 2 == 0:
            pool = PATIENT_UTTERANCES[theme] + PATIENT_UTTERANCES["neutral"]
            text = random.choice(pool)
            speaker = "patient"
        else:
            text = random.choice(THERAPIST_UTTERANCES)
            speaker = "therapist"

        turns.append({
            "turn_id": i + 1,
            "speaker": speaker,
            "text": text,
            "word_count": len(text.split()),
        })

    baseline_severity = random.randint(12, 24)
    improvement = max(0, session_num * random.uniform(0.3, 0.8))
    severity = round(max(5, baseline_severity - improvement))

    return {
        "session_id": session_id,
        "patient_id": patient_id,
        "cohort": cohort,
        "session_number": session_num,
        "dominant_theme": theme,
        "severity_score": severity,
        "turns": turns,
        "date": (datetime.today() - timedelta(days=(10 - session_num) * 7)).strftime("%Y-%m-%d"),
    }

def generate_dataset(n_patients=40, max_sessions=6):
    dataset = []
    session_id = 1
    for p in range(n_patients):
        patient_id = f"P{p+1:03d}"
        cohort = random.choice(COHORTS)
        n_sessions = random.randint(2, max_sessions)
        for s in range(1, n_sessions + 1):
            session = generate_session(session_id, patient_id, cohort, s)
            dataset.append(session)
            session_id += 1
    return dataset

if __name__ == "__main__":
    data = generate_dataset()
    with open("data/sessions.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} sessions across {len(set(s['patient_id'] for s in data))} patients.")
