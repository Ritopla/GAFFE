# GAFFE — Geometric Analysis of Facial Features for Emotion/Stress

Deterministic, explainable stress detection pipeline based on facial landmark geometry.  
Uses MediaPipe Face Mesh (478 landmarks) to compute psychophysiological stress metrics frame-by-frame from video.

## Project Structure

```
GAFFE/
├── detection/          ← Video → JSON stress extraction
│   ├── demo.py         ← CLI: process a video file
│   ├── config.py       ← All tunable parameters
│   ├── landmarks.py    ← MediaPipe landmark indices
│   ├── stress_scorer.py← Main scoring pipeline
│   └── metrics/        ← Individual stress metrics
│       ├── brow_furrow.py      (BFI — AU4 proxy)
│       ├── eye_aspect.py       (EAR + blink dynamics)
│       └── brow_asymmetry.py   (BAD — asymmetry + variance)
│
├── analysis/           ← JSON → Statistics & Charts
│   ├── analyze.py      ← CLI: single-video analysis
│   ├── batch_analyze.py← CLI: batch dataset analysis
│   ├── plot_results.py ← CLI: cross-dataset comparison plots
│   ├── single_video.py ← Core single-video analysis engine
│   ├── batch.py        ← Core batch processing engine
│   └── charts.py       ← Matplotlib visualization suite
│
├── models/             ← ML model assets (not tracked, see below)
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

### 2. Download the FaceLandmarker model

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Usage

### Detection — Extract stress from video

```bash
# Process a single video (with live display)
python -m detection.demo path/to/video.mp4

# Headless mode with custom output path
python -m detection.demo path/to/video.mp4 --no-display -o results/output_gaffe.json
```

This produces a `_gaffe.json` file containing per-frame stress scores, metric breakdowns, and summary statistics.

### Analysis — Analyze JSON results

```bash
# Single video analysis
python -m analysis.analyze results/video_gaffe.json --charts

# Batch analysis of a dataset directory
python -m analysis.batch_analyze datasets/MyDataset/ --charts

# Cross-dataset comparison plot
python -m analysis.plot_results results/Dataset1/batch_analysis.json results/Dataset2/batch_analysis.json
```

## Stress Metrics

| Metric | Full Name | What it measures | Weight |
|--------|-----------|-----------------|--------|
| **BFI** | Brow Furrow Index | Eyebrow convergence + lowering (AU4) | 45% |
| **EAR** | Eye Aspect Ratio | Lid tightening + blink rate + blink speed | 35% |
| **BAD** | Brow Asymmetry + Dynamics | Left/right asymmetry + temporal variance | 20% |

The composite stress score is a weighted sum of these three metrics, producing a value in [0, 1] classified as: **LOW** (≤0.25), **MODERATE** (≤0.50), **HIGH** (≤0.75), **VERY HIGH** (>0.75).

## Requirements

- Python ≥ 3.10
- MediaPipe ≥ 0.10.0
- OpenCV ≥ 4.8.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- SciPy ≥ 1.10.0
