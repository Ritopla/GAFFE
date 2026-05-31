# GAFFE — Geometric Analysis of Facial Features for Emotion/Stress

Deterministic, explainable stress detection pipeline based on facial landmark geometry.  
Uses MediaPipe Face Mesh (478 landmarks) to compute psychophysiological stress metrics frame-by-frame from video.  
An alternative FER (Facial Expression Recognition) pipeline is also available.

## Project Structure

```
GAFFE/
├── detection/          ← Video → JSON stress extraction
│   ├── detect.py       ← CLI: process a video (landmark or FER)
│   ├── config.py       ← All tunable parameters
│   ├── landmarks.py    ← MediaPipe landmark indices
│   ├── stress_scorer.py← Main scoring pipeline (landmark)
│   └── metrics/        ← Individual stress metrics
│       ├── brow_furrow.py      (BFI — AU4 proxy)
│       ├── eye_aspect.py       (EAR + blink dynamics)
│       └── brow_asymmetry.py   (BAD — asymmetry + variance)
│
├── analysis/           ← JSON → Statistics & Charts
│   ├── cli.py          ← CLI: single / batch / compare subcommands
│   ├── single_video.py ← Core single-video analysis engine
│   ├── batch.py        ← Core batch processing engine
│   └── charts.py       ← Matplotlib visualization suite
│
├── CLEANING/           ← Dataset pre-processing utilities
│   ├── reference_extractor.py  ← Identify the main subject in a video
│   ├── video_filter.py         ← Filter frames by face identity
│   └── clean_dataset.py        ← End-to-end cleaning script
│
├── models/             ← ML model assets (not tracked, see below)
├── results/            ← All outputs land here (not tracked by git)
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the FaceLandmarker model (landmark pipeline only)

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 3. FER pipeline extra dependency (optional)

```bash
pip install fer tensorflow
```

---

## Usage

### Step 1 — Detection: extract stress from video

```bash
# Landmark pipeline (default) — with live display
python -m detection.detect path/to/video.mp4

# Landmark pipeline — headless, custom output path
python -m detection.detect path/to/video.mp4 --no-display -o results/my_video/out.json

# FER pipeline
python -m detection.detect path/to/video.mp4 --method fer

# FER pipeline — MTCNN detector, analyse every 2nd frame
python -m detection.detect path/to/video.mp4 --method fer --use-mtcnn --sample-every 2
```

Output is written to **`results/<video_stem>/<video_stem>_gaffe.json`** by default  
(the directory is created automatically and overwritten on each run).

### Step 2 — Analysis: analyze JSON results

All analysis commands share the same entry point with subcommands:

```bash
# Single-video analysis — print summary to stdout
python -m analysis.cli single results/my_video/my_video_gaffe.json

# Single-video analysis — also generate the summary dashboard chart
python -m analysis.cli single results/my_video/my_video_gaffe.json --charts

# Single-video analysis — generate all individual charts as well
python -m analysis.cli single results/my_video/my_video_gaffe.json --charts --all-charts

# Batch analysis of a dataset directory (reads existing _gaffe.json files)
python -m analysis.cli batch datasets/MyDataset/

# Batch analysis — process raw videos first, then generate summary charts
python -m analysis.cli batch datasets/MyDataset/ --process-videos --charts

# Batch analysis — generate all per-video charts too
python -m analysis.cli batch datasets/MyDataset/ --charts --all-charts

# Cross-dataset comparison plot
python -m analysis.cli compare results/Deceptive/batch_analysis.json \
                                results/Truthful/batch_analysis.json
```

Default output locations (always overwritten):

| Subcommand | Default output |
|------------|----------------|
| `single`   | `results/<video_stem>/` |
| `batch`    | `results/<dataset_name>/` |
| `compare`  | `results/comparison/` |

Use `-o` / `--output-dir` to override.

### Charts

By default only the **summary dashboard** is generated (one PNG per video).  
Add `--all-charts` to also produce the individual per-metric / per-emotion charts:

| Chart | Dashboard only | With `--all-charts` |
|-------|:--------------:|:-------------------:|
| Summary dashboard | ✓ | ✓ |
| Stress timeline | — | ✓ |
| Level distribution | — | ✓ |
| Metric contributions (landmark) | — | ✓ |
| Blink rate overlay (landmark) | — | ✓ |
| Metric bars (landmark) | — | ✓ |
| Emotion timeline (FER) | — | ✓ |
| Emotion contributions (FER) | — | ✓ |
| Emotion bars (FER) | — | ✓ |

---

## Stress Metrics (Landmark Pipeline)

| Metric | Full Name | What it measures | Weight |
|--------|-----------|-----------------|--------|
| **BFI** | Brow Furrow Index | Eyebrow convergence + lowering (AU4) | 45% |
| **EAR** | Eye Aspect Ratio | Lid tightening + blink rate + blink speed | 35% |
| **BAD** | Brow Asymmetry + Dynamics | Left/right asymmetry + temporal variance | 20% |

The composite stress score is a weighted sum of these three metrics, producing a value in [0, 1] classified as:  
**LOW** (≤ 0.25) · **MODERATE** (≤ 0.50) · **HIGH** (≤ 0.75) · **VERY HIGH** (> 0.75)

## Stress Score (FER Pipeline)

The FER pipeline classifies each frame into 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral).  
The stress score is computed as:

```
stress = clip(angry + fear + disgust, 0, 1)
```

Classification thresholds are identical to the landmark pipeline, so scores are directly comparable.

---

## Requirements

- Python ≥ 3.10
- MediaPipe ≥ 0.10.0
- OpenCV ≥ 4.8.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- SciPy ≥ 1.10.0
- `fer` + `tensorflow` *(FER pipeline only — install separately)*
