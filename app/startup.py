import subprocess, sys
from pathlib import Path

def run_pipeline():
    if not Path("data/raw/athlete_sessions.csv").exists():
        print("Generating data...")
        subprocess.run([sys.executable, "data/generate.py"], check=True)
    if not Path("models/zone_classifier.pkl").exists():
        print("Note: Run notebook 04 to generate the classifier model")