import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

STATES = {
    0: "fatigued",
    1: "normal",
    2: "zone"   # peak performance state NeuralPort aims to recreate
}

def generate_athlete_session(athlete_id: int, session_id: int,
                              state: int, sport: str) -> dict:
    """
    Generate one 5-minute athlete performance session.
    state: 0=fatigued, 1=normal, 2=zone (peak focus)
    """
    n = 30  # 30 readings per session (every 10 seconds)

    # Heart rate — zone has controlled HR, fatigue has elevated HR
    hr_base  = [88, 75, 68][state]
    hr       = np.clip(np.random.normal(hr_base, [8, 5, 3][state], n), 50, 130)

    # HRV (Heart Rate Variability) — higher HRV = better recovery
    hrv_base = [28, 45, 65][state]
    hrv      = np.clip(np.random.normal(hrv_base, [6, 5, 4][state], n), 10, 100)

    # Reaction time (ms) — faster in zone
    rt_base  = [380, 280, 195][state]
    reaction = np.clip(np.random.normal(rt_base, [35, 20, 12][state], n), 100, 600)

    # Decision accuracy (%) — higher in zone
    acc_base = [62, 78, 94][state]
    accuracy = np.clip(np.random.normal(acc_base, [8, 5, 3][state], n), 30, 100)

    # Pupil diameter (normalized) — moderate in zone, small when fatigued
    pupil_base = [0.48, 0.64, 0.74][state]
    pupil      = np.clip(np.random.normal(pupil_base, 0.04, n), 0.2, 1.0)

    # Blink rate (blinks/min) — lower in zone (inhibited during focus)
    blink_base = [24, 16, 9][state]
    blink      = np.clip(np.random.normal(blink_base, [4, 3, 2][state], n), 2, 40)

    # Saccade velocity (deg/s)
    sacc_base  = [210, 300, 390][state]
    saccade    = np.clip(np.random.normal(sacc_base, 30, n), 100, 600)

    # Cortisol proxy (0-1, higher = more stress/fatigue)
    cortisol_base = [0.72, 0.45, 0.28][state]
    cortisol      = np.clip(np.random.normal(cortisol_base, 0.07, n), 0.0, 1.0)

    # Focus score (subjective 1-10)
    focus_base = [3.5, 6.0, 9.2][state]
    focus      = np.clip(np.random.normal(focus_base, [0.8, 0.6, 0.4][state], n),
                          1.0, 10.0)

    # Movement efficiency (higher = smoother, less wasted motion)
    move_base  = [0.55, 0.72, 0.91][state]
    movement   = np.clip(np.random.normal(move_base, 0.05, n), 0.2, 1.0)

    return {
        "athlete_id":        [athlete_id] * n,
        "session_id":        [session_id] * n,
        "reading":           list(range(n)),
        "sport":             [sport] * n,
        "state":             [state] * n,
        "state_label":       [STATES[state]] * n,
        "heart_rate":        hr.tolist(),
        "hrv":               hrv.tolist(),
        "reaction_time_ms":  reaction.tolist(),
        "decision_accuracy": accuracy.tolist(),
        "pupil_diameter":    pupil.tolist(),
        "blink_rate":        blink.tolist(),
        "saccade_velocity":  saccade.tolist(),
        "cortisol_proxy":    cortisol.tolist(),
        "focus_score":       focus.tolist(),
        "movement_efficiency": movement.tolist(),
    }


def generate_dataset(n_athletes: int = 100,
                     sessions_per_state: int = 7) -> pd.DataFrame:
    sports = ["football", "basketball", "tennis",
              "swimming", "athletics", "cycling"]
    all_rows = []
    session_id = 0

    for athlete_id in range(n_athletes):
        sport = np.random.choice(sports)
        for state in [0, 1, 2]:
            for _ in range(sessions_per_state):
                session = generate_athlete_session(
                    athlete_id, session_id, state, sport)
                all_rows.append(pd.DataFrame(session))
                session_id += 1

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    print("Generating athlete performance dataset...")
    df = generate_dataset(n_athletes=100, sessions_per_state=7)
    out = Path("data/raw/athlete_sessions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows — "
          f"{df['session_id'].nunique()} sessions, "
          f"{df['athlete_id'].nunique()} athletes")
    print(df.groupby("state_label")["session_id"].nunique())