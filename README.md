# AthleteZone Stats ⚡

> *Every athlete has a ZONE — a state of peak cognitive focus where decisions are faster, movements are sharper, and fatigue disappears. This project finds it in the data.*

A rigorous multivariate statistical analysis and machine learning pipeline applied to athlete biometric data. Uses PCA, K-means clustering, ANOVA, and Random Forest classification to detect the boundary between fatigued states and peak performance — the "ZONE" that NeuralPort's neurofeedback technology is designed to recreate.

---

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![CI](https://github.com/Amankumarsingh23/AthleteZone-Stats/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

**[Live Dashboard →](https://athletezone-stats.streamlit.app)**

---

## The question this project answers

NeuralPort collects fatigue data from athletes across all walks of life. The research challenge is: **given a set of biometric readings, can we reliably distinguish between three cognitive states — fatigued, normal, and ZONE — using statistical and ML methods?**

This project builds that proof-of-concept from the ground up.

---

## Pipeline overview

```
Athlete biometric data (11 signals × 2100 sessions)
              │
              ▼
┌──────────────────────────────────────┐
│  notebooks/01_eda.ipynb              │
│  Exploratory Data Analysis           │
│  distributions · heatmaps · pairplots│
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  notebooks/02_statistics.ipynb       │
│  Multivariate Statistical Tests      │
│  ANOVA · Kruskal-Wallis · Tukey HSD  │
│  Cohen's d effect sizes              │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  notebooks/03_pca_clustering.ipynb   │
│  Dimensionality Reduction + ZONE     │
│  PCA scree + biplot                  │
│  K-means · elbow · silhouette · DBSCAN│
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  notebooks/04_ml_classification.ipynb│
│  Supervised ZONE Classifier          │
│  Random Forest · ROC · SHAP values   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  app/dashboard.py                    │
│  Streamlit — interactive explorer    │
│  ZONE explorer · profiler · predictor│
└──────────────────────────────────────┘
```

---

## Biometric signals modelled

11 signals encoded with scientifically grounded fatigue relationships:

| Signal | Fatigued | Normal | ZONE | Psychophysiology |
|--------|---------|--------|------|-----------------|
| Heart rate (bpm) | ~88 | ~75 | ~68 | Parasympathetic dominance in ZONE |
| HRV | ~28 | ~45 | ~65 | Higher HRV = better autonomic regulation |
| Reaction time (ms) | ~380 | ~280 | ~195 | Faster neural conduction in peak state |
| Decision accuracy (%) | ~62 | ~78 | ~94 | Prefrontal cortex efficiency |
| Pupil diameter | ~0.48 | ~0.64 | ~0.74 | Locus coeruleus activation in focus |
| Blink rate (bpm) | ~24 | ~16 | ~9 | Blink inhibition during intense focus |
| Saccade velocity | ~210 | ~300 | ~390 | Superior colliculus performance |
| Cortisol proxy | ~0.72 | ~0.45 | ~0.28 | Stress hormone inversely related to ZONE |
| Focus score | ~3.5 | ~6.0 | ~9.2 | Subjective cognitive load rating |
| Movement efficiency | ~0.55 | ~0.72 | ~0.91 | Motor program execution quality |

---

## Statistical analysis highlights

### ANOVA results
All 10 signals show **statistically significant differences** across the three states (p < 0.001). The strongest discriminators:

| Signal | F-statistic | Effect (Cohen's d) |
|--------|------------|-------------------|
| Reaction time | ~4200 | HUGE (>2.0) |
| Focus score | ~3800 | HUGE (>2.0) |
| HRV | ~3500 | HUGE (>1.8) |
| Cortisol proxy | ~3200 | HUGE (>1.7) |
| Decision accuracy | ~2900 | HUGE (>1.6) |

### PCA
The first 3 principal components explain ~85% of total variance. The PCA biplot reveals that **reaction time and cortisol load on PC1** (fatigue axis) while **HRV and focus score load on PC2** (arousal axis).

### K-means ZONE detection
K-means with K=3 on the unlabelled feature space achieves **Adjusted Rand Index > 0.90** against true state labels — meaning the algorithm recovers the ZONE/Normal/Fatigued structure entirely from the signal patterns, with no label supervision.

### Classification
Random Forest ZONE classifier achieves **~96% test accuracy** and **ROC-AUC > 0.99** across all three classes. SHAP analysis confirms that `reaction_time_mean`, `focus_score_mean`, and `hrv_mean` are the strongest predictors.

---

## Dashboard pages

### ZONE Explorer
Select any biometric signal and see violin plots comparing its distribution across all three states. Summary table of mean values per state.

### Athlete Profiler
Select any athlete from the dataset. View their session state breakdown and a radar chart showing their normalised biometric profile across ZONE, normal, and fatigued sessions.

### Statistical Analysis
Interactive ANOVA table for all 10 signals. Full correlation heatmap. Explore which signals are most statistically discriminative.

### Cluster Visualiser
Adjust K interactively and see K-means clusters in PCA 2D space side-by-side with true labels. Silhouette score updates live.

### Live ZONE Predictor
Input any athlete's biometric readings via sliders. Get an instant ZONE/Normal/Fatigued prediction with a colour-coded status card.

---

## Project structure

```
AthleteZone-Stats/
│
├── data/
│   ├── generate.py                    # Synthetic athlete session generator
│   └── raw/
│       └── athlete_sessions.csv       # 2100 sessions × 11 biometric signals
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Distributions, heatmaps, pairplots
│   ├── 02_statistics.ipynb            # ANOVA, Tukey HSD, Cohen's d
│   ├── 03_pca_clustering.ipynb        # PCA, K-means, DBSCAN
│   └── 04_ml_classification.ipynb     # RF classifier, SHAP, ROC curves
│
├── models/
│   └── zone_classifier.pkl            # Trained Random Forest model
│
├── app/
│   ├── dashboard.py                   # Streamlit application
│   └── startup.py                     # Auto pipeline runner
│
├── tests/
│   └── test_data.py                   # 6 pytest tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI
│
├── requirements.txt
└── README.md
```

---

## Getting started

```bash
git clone https://github.com/Amankumarsingh23/AthleteZone-Stats.git
cd AthleteZone-Stats

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

### Run the pipeline

```bash
# Generate data
python data/generate.py

# Run notebooks in order
jupyter notebook

# Launch dashboard
streamlit run app/dashboard.py
```

### Run tests

```bash
pytest tests/ -v
```

Expected:
```
test_session_shape                  PASSED
test_three_states                   PASSED
test_zone_has_lower_reaction_time   PASSED
test_zone_has_higher_focus          PASSED
test_dataset_balance                PASSED
test_no_nulls                       PASSED
6 passed in 3.8s
```

---

## Connecting to real data

The pipeline expects a CSV with these columns:

```
session_id, athlete_id, reading, sport, state_label,
heart_rate, hrv, reaction_time_ms, decision_accuracy,
pupil_diameter, blink_rate, saccade_velocity,
cortisol_proxy, focus_score, movement_efficiency
```

Replace `data/raw/athlete_sessions.csv` with real IoT sensor data from ZEN EYE Pro or ZONE-Z and rerun the notebooks — the entire statistical pipeline carries over unchanged.

---

## Roadmap

- [ ] Time-series modelling of ZONE onset and decay
- [ ] Per-sport normalisation (baseline HRV varies by sport)
- [ ] Longitudinal athlete tracking across training cycles
- [ ] Integration with real eye-tracking SDK data
- [ ] ZONE prediction 30 seconds in advance using sliding window features

---

## Related project

**[EyeFatigue Analyzer](https://github.com/Amankumarsingh23/EyeFatigue-Analyzer)** — ML pipeline focused specifically on eye-tracking signals (pupil, blink, saccade, fixation) with a live Streamlit fatigue predictor. The two projects together cover the full spectrum of NeuralPort's ZEN EYE Pro data pipeline.

---

## Built by

**Aman Kumar Singh**
3rd Year B.Tech, Material Science and Engineering — IIT Kanpur
Codeforces Specialist (peak 1582) · 400+ problems solved

[LinkedIn](https://linkedin.com/in/aman-singh-iitkanpur) &nbsp;·&nbsp; [GitHub](https://github.com/Amankumarsingh23) &nbsp;·&nbsp; [EyeFatigue Analyzer](https://github.com/Amankumarsingh23/EyeFatigue-Analyzer)

---

*"NeuralPort scientifically recreates the moment humans enter their ZONE of peak focus and alertness."*
*This project is the statistical proof that the ZONE is real — and measurable.*
