import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.generate import generate_dataset, generate_athlete_session

def test_session_shape():
    s = generate_athlete_session(0, 0, state=0, sport="football")
    df = pd.DataFrame(s)
    assert len(df) == 30
    assert "heart_rate" in df.columns

def test_three_states():
    for state in [0, 1, 2]:
        s = generate_athlete_session(0, 0, state=state, sport="tennis")
        df = pd.DataFrame(s)
        assert df["state"].iloc[0] == state

def test_zone_has_lower_reaction_time():
    df = generate_dataset(n_athletes=20, sessions_per_state=5)
    agg = df.groupby(["session_id","state_label"])["reaction_time_ms"].mean()
    sess = agg.reset_index()
    zone_rt     = sess[sess["state_label"]=="zone"]["reaction_time_ms"].mean()
    fatigued_rt = sess[sess["state_label"]=="fatigued"]["reaction_time_ms"].mean()
    assert zone_rt < fatigued_rt

def test_zone_has_higher_focus():
    df = generate_dataset(n_athletes=20, sessions_per_state=5)
    agg = df.groupby(["session_id","state_label"])["focus_score"].mean()
    sess = agg.reset_index()
    zone_f     = sess[sess["state_label"]=="zone"]["focus_score"].mean()
    fatigued_f = sess[sess["state_label"]=="fatigued"]["focus_score"].mean()
    assert zone_f > fatigued_f

def test_dataset_balance():
    df = generate_dataset(n_athletes=10, sessions_per_state=3)
    counts = df.groupby("state_label")["session_id"].nunique()
    assert len(counts) == 3
    assert counts.min() == 30  # 10 athletes × 3 sessions

def test_no_nulls():
    df = generate_dataset(n_athletes=5, sessions_per_state=2)
    assert df.isnull().sum().sum() == 0