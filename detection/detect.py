#!/usr/bin/env python3
"""
GAFFE Detect — Extract stress metrics from a video file.

Two detection pipelines are available:

  landmark  (default)
      MediaPipe Face Mesh + geometric metrics (BFI, EAR, BAD).
      Requires the face_landmarker.task model (see README).

  fer
      FER library + emotion-based stress scoring (angry + fear + disgust).
      Requires: pip install fer tensorflow

Subcommands:
    single   Process a single video file (default when no subcommand is given).
    batch    Process all videos in a directory.

Usage:
    python -m detection.detect [single] <video> [--method landmark|fer] [options]
    python -m detection.detect batch <directory> [--method landmark|fer] [options]

Output (written to results/<video_stem>/ by default, overwrites):
    <video_stem>_gaffe.json

Examples:
    python -m detection.detect video.mp4
    python -m detection.detect single video.mp4 --method fer
    python -m detection.detect video.mp4 --method fer --use-mtcnn
    python -m detection.detect video.mp4 --no-display
    python -m detection.detect video.mp4 -o /custom/path/out.json
    python -m detection.detect video.mp4 --method fer --sample-every 2
    python -m detection.detect batch /path/to/videos/ --method fer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────

# Stress-level thresholds (shared across both pipelines).
STRESS_LOW_MAX: float = 0.25
STRESS_MODERATE_MAX: float = 0.50
STRESS_HIGH_MAX: float = 0.75


def classify_stress(score: float) -> str:
    """Map a [0, 1] stress score to a human-readable level."""
    if score <= STRESS_LOW_MAX:
        return "LOW"
    elif score <= STRESS_MODERATE_MAX:
        return "MODERATE"
    elif score <= STRESS_HIGH_MAX:
        return "HIGH"
    else:
        return "VERY_HIGH"


def _default_output_path(video_path: Path) -> Path:
    """
    Build the default output path for a video.

    Convention: results/<video_stem>/<video_stem>_gaffe.json
    The directory is created if needed; the file is overwritten on each run.
    """
    out_dir = Path("results") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{video_path.stem}_gaffe.json"


def _print_progress(frame_id: int, total_frames: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    fps_actual = frame_id / elapsed if elapsed > 0 else 0
    pct = (frame_id / total_frames * 100) if total_frames > 0 else 0
    print(
        f"  Frame {frame_id}/{total_frames} ({pct:.0f}%) — {fps_actual:.1f} fps",
        end="\r",
    )


# ──────────────────────────────────────────────────────────────
# Landmark pipeline (MediaPipe Face Mesh)
# ──────────────────────────────────────────────────────────────


def _run_landmark(
    video_path: Path,
    output_path: Path,
    no_display: bool,
    fps_override: float | None,
) -> None:
    """Process a video with the MediaPipe landmark pipeline."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
    except ImportError:
        raise RuntimeError(
            "mediapipe is required for the landmark pipeline. "
            "Install it with: pip install mediapipe"
        )

    from detection.config import StressConfig
    from detection.stress_scorer import StressScorer

    model_path = Path(__file__).parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        raise RuntimeError(
            f"FaceLandmarker model not found at {model_path}. "
            "Download it with: curl -L -o models/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:    {video_path.name}")
    print(f"Method:   landmark (MediaPipe Face Mesh)")
    print(f"Res/FPS:  {width}x{height} @ {fps:.1f}")
    print(f"Frames:   {total_frames}  ({total_frames / fps:.1f}s)")
    print(f"Output:   {output_path}")
    print("-" * 50)

    config = StressConfig(fps=fps)
    scorer = StressScorer(config)

    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    frame_results: list[dict] = []
    frame_id = 0
    start_time = time.time()

    # Level colors for overlay (BGR)
    level_colors = {
        "LOW": (0, 200, 0),
        "MODERATE": (0, 200, 255),
        "HIGH": (0, 100, 255),
        "VERY_HIGH": (0, 0, 255),
    }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_id / fps) * 1000.0

            # Landmark extraction
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            lm_result = face_landmarker.detect_for_video(mp_image, int(timestamp_ms))

            if lm_result.face_landmarks:
                face = lm_result.face_landmarks[0]
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in face], dtype=np.float64
                )
                result = scorer.process_frame(landmarks, frame_id, timestamp_ms)
            else:
                landmarks = None
                result = StressScorer.no_face_result(frame_id, timestamp_ms)

            frame_results.append(result)

            if not no_display:
                overlay = frame.copy()
                if result["face_detected"] and landmarks is not None:
                    stress = result["stress"]
                    metrics = result["metrics"]
                    color = level_colors.get(stress["level"], (255, 255, 255))
                    bar_width = int(stress["score"] * 200)
                    cv2.rectangle(overlay, (20, 15), (220, 45), (50, 50, 50), -1)
                    cv2.rectangle(overlay, (20, 15), (20 + bar_width, 45), color, -1)
                    cv2.putText(
                        overlay,
                        f"Stress: {stress['score']:.2f} [{stress['level']}]",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    )
                    for i, text in enumerate([
                        f"BFI: {metrics['bfi']['value']:.3f}",
                        f"EAR: {metrics['ear']['value']:.3f}  Blink: {metrics['ear']['blink_rate_per_min']:.0f}/min",
                        f"BAD: {metrics['bad']['value']:.3f}",
                    ]):
                        cv2.putText(
                            overlay, text, (20, 70 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                        )
                    for idx in [55, 285, 70, 300, 159, 386, 33, 133, 263, 362]:
                        if idx < len(landmarks):
                            pt = tuple(landmarks[idx].astype(int))
                            cv2.circle(overlay, pt, 2, (0, 255, 0), -1)
                else:
                    cv2.putText(
                        overlay, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    )
                cv2.imshow("GAFFE — Landmark Detection", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            frame_id += 1
            if frame_id % 100 == 0:
                _print_progress(frame_id, total_frames, start_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        face_landmarker.close()
        if not no_display:
            cv2.destroyAllWindows()

    _save_landmark_output(
        frame_results, video_path, output_path,
        width, height, fps, frame_id, start_time,
    )


def _save_landmark_output(
    frame_results: list[dict],
    video_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_id: int,
    start_time: float,
) -> None:
    """Compute summary and write landmark JSON output."""
    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_id} frames in {elapsed:.1f}s "
          f"({frame_id / elapsed:.1f} fps)" if elapsed > 0 else "")

    detected = [r for r in frame_results if r["face_detected"]]
    if not detected:
        summary: dict[str, Any] = {
            "total_frames": len(frame_results),
            "face_detected_frames": 0,
            "face_detection_rate": 0.0,
        }
    else:
        scores = [r["stress"]["score"] for r in detected]
        levels = [r["stress"]["level"] for r in detected]
        level_counts = {}
        for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]:
            count = levels.count(level)
            level_counts[level] = {
                "count": count,
                "percentage": round(count / len(detected) * 100, 1),
            }
        summary = {
            "total_frames": len(frame_results),
            "face_detected_frames": len(detected),
            "face_detection_rate": round(len(detected) / len(frame_results) * 100, 1),
            "stress_score": {
                "mean":   round(float(np.mean(scores)),   4),
                "std":    round(float(np.std(scores)),    4),
                "min":    round(float(np.min(scores)),    4),
                "max":    round(float(np.max(scores)),    4),
                "median": round(float(np.median(scores)), 4),
            },
            "level_distribution": level_counts,
            "metrics_summary": {
                "bfi_mean":      round(float(np.mean([r["metrics"]["bfi"]["value"] for r in detected])), 4),
                "ear_mean":      round(float(np.mean([r["metrics"]["ear"]["value"] for r in detected])), 4),
                "ear_stress_mean": round(float(np.mean([r["metrics"]["ear"]["ear_stress_score"] for r in detected])), 4),
                "bad_mean":      round(float(np.mean([r["metrics"]["bad"]["value"] for r in detected])), 4),
            },
        }

    output = {
        "metadata": {
            "video_file":             video_path.name,
            "video_resolution":       f"{width}x{height}",
            "video_fps":              fps,
            "total_frames_processed": frame_id,
            "pipeline_version":       "0.2.0",
            "detection_method":       "landmark",
        },
        "summary": summary,
        "frames":  frame_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
    if "stress_score" in summary:
        ss = summary["stress_score"]
        print(f"\n── Summary ──────────────────────────────────────")
        print(f"  Face detection rate: {summary.get('face_detection_rate', 0)}%")
        print(f"  Stress score — mean: {ss['mean']:.3f}, "
              f"median: {ss['median']:.3f}, std: {ss['std']:.3f}")
        for level, info in summary.get("level_distribution", {}).items():
            bar = "█" * int(info["percentage"] / 5)
            print(f"  {level:>10}: {info['percentage']:5.1f}% {bar}")


# ──────────────────────────────────────────────────────────────
# FER pipeline (emotion-based)
# ──────────────────────────────────────────────────────────────

# FER emotion names in canonical order.
_FER_EMOTION_NAMES: tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
)
# Weights for the stress score: stress = angry + fear + disgust (all = 1.0).
_FER_WEIGHTS: dict[str, float] = {
    "angry": 1.0, "fear": 1.0, "disgust": 1.0,
}
_FER_VERSION = "0.2.0-fer"


def _run_fer(
    video_path: Path,
    output_path: Path,
    no_display: bool,
    fps_override: float | None,
    use_mtcnn: bool,
    sample_every: int,
    save_annotated: bool,
    annotated_path: Path | None,
) -> None:
    """Process a video with the FER emotion-recognition pipeline."""
    try:
        from fer.fer import FER
    except ImportError:
        raise RuntimeError(
            "the 'fer' package is required for the FER pipeline. "
            "Install it with: pip install fer tensorflow"
        )

    detector_name = "MTCNN" if use_mtcnn else "Haarcascade"
    print(f"Initialising FER (detector: {detector_name})...")
    fer_detector = FER(mtcnn=use_mtcnn)
    print("FER ready.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video:    {video_path.name}")
    print(f"Method:   FER (emotion-based)")
    print(f"Res/FPS:  {width}x{height} @ {fps:.1f}")
    print(f"Frames:   {total_frames}  ({total_frames / fps:.1f}s)")
    print(f"Output:   {output_path}")
    if sample_every > 1:
        print(f"Sampling: every {sample_every} frames")
    print("-" * 50)

    writer = None
    if save_annotated and annotated_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    frame_results: list[dict] = []
    frame_id = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_id / fps) * 1000.0

            if frame_id % sample_every == 0:
                detections = fer_detector.detect_emotions(frame)
                if detections:
                    em_raw = detections[0]["emotions"]
                    em = {k: float(em_raw.get(k, 0.0)) for k in _FER_EMOTION_NAMES}
                    stress_score = float(np.clip(
                        sum(_FER_WEIGHTS.get(k, 0.0) * em[k] for k in _FER_EMOTION_NAMES),
                        0.0, 1.0,
                    ))
                    result: dict[str, Any] = {
                        "frame_id": frame_id,
                        "face_detected": True,
                        "timestamp_ms": round(timestamp_ms, 1),
                        "emotions": {k: round(em[k], 4) for k in _FER_EMOTION_NAMES},
                        "stress": {
                            "score": round(stress_score, 4),
                            "level": classify_stress(stress_score),
                            "weights_used": dict(_FER_WEIGHTS),
                        },
                    }
                    if not no_display or writer is not None:
                        x, y, bw, bh = detections[0]["box"]
                        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1)
                        _draw_fer_overlay(frame, result)
                else:
                    result = _fer_no_face(frame_id, timestamp_ms)
                    if not no_display or writer is not None:
                        _draw_fer_overlay(frame, result)
            else:
                result = _fer_no_face(frame_id, timestamp_ms)

            frame_results.append(result)

            if writer is not None:
                writer.write(frame)
            if not no_display:
                cv2.imshow("GAFFE — FER Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            frame_id += 1
            if frame_id % 50 == 0:
                _print_progress(frame_id, total_frames, start_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_id} frames in {elapsed:.1f}s "
          f"({frame_id / elapsed:.1f} fps)" if elapsed > 0 else "")

    _save_fer_output(frame_results, video_path, output_path, width, height, fps, frame_id)


def _fer_no_face(frame_id: int, timestamp_ms: float) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "face_detected": False,
        "timestamp_ms": round(timestamp_ms, 1),
        "emotions": None,
        "stress": None,
    }


def _draw_fer_overlay(frame: np.ndarray, result: dict) -> None:
    """Draw FER stress overlay on the frame (in-place)."""
    if not result["face_detected"]:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    score = result["stress"]["score"]
    level = result["stress"]["level"]
    if score <= STRESS_LOW_MAX:
        color = (0, 200, 0)
    elif score <= STRESS_MODERATE_MAX:
        color = (0, 200, 255)
    elif score <= STRESS_HIGH_MAX:
        color = (0, 100, 255)
    else:
        color = (0, 0, 220)
    cv2.putText(frame, f"Stress: {score:.2f} [{level}]", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    bar_len = int(score * 200)
    cv2.rectangle(frame, (20, 60), (220, 75), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 60), (20 + bar_len, 75), color, -1)
    t = result["timestamp_ms"] / 1000.0
    cv2.putText(frame, f"t={t:.1f}s", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    emotions = result.get("emotions") or {}
    top3 = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
    for i, (name, val) in enumerate(top3):
        cv2.putText(frame, f"{name}: {val:.2f}", (20, 130 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def _save_fer_output(
    frame_results: list[dict],
    video_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_id: int,
) -> None:
    """Compute FER summary and write JSON output."""
    detected = [r for r in frame_results if r["face_detected"]]
    if not detected:
        summary: dict[str, Any] = {
            "total_frames": len(frame_results),
            "face_detected_frames": 0,
            "face_detection_rate": 0.0,
        }
    else:
        scores = [r["stress"]["score"] for r in detected]
        levels = [r["stress"]["level"] for r in detected]
        level_counts = {}
        for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]:
            count = levels.count(level)
            level_counts[level] = {
                "count": count,
                "percentage": round(count / len(detected) * 100, 1),
            }
        emotions_summary = {
            f"{em}_mean": round(float(np.mean([r["emotions"][em] for r in detected])), 4)
            for em in _FER_EMOTION_NAMES
        }
        summary = {
            "total_frames": len(frame_results),
            "face_detected_frames": len(detected),
            "face_detection_rate": round(len(detected) / len(frame_results) * 100, 1),
            "stress_score": {
                "mean":   round(float(np.mean(scores)),   4),
                "std":    round(float(np.std(scores)),    4),
                "min":    round(float(np.min(scores)),    4),
                "max":    round(float(np.max(scores)),    4),
                "median": round(float(np.median(scores)), 4),
            },
            "level_distribution": level_counts,
            "emotions_summary": emotions_summary,
        }

    output = {
        "metadata": {
            "video_file":             video_path.name,
            "video_resolution":       f"{width}x{height}",
            "video_fps":              fps,
            "total_frames_processed": frame_id,
            "pipeline_version":       _FER_VERSION,
            "detection_method":       "fer",
        },
        "summary": summary,
        "frames":  frame_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


# ──────────────────────────────────────────────────────────────
# Batch detection
# ──────────────────────────────────────────────────────────────

_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def _cmd_batch_detect(args: argparse.Namespace) -> None:
    """Process all videos in a directory with the chosen detection pipeline."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / directory.name
    output_dir.mkdir(parents=True, exist_ok=True)

    method: str = args.method

    videos: list[Path] = []
    for ext in _VIDEO_EXTENSIONS:
        videos.extend(directory.glob(f"*{ext}"))
    videos = sorted(videos)

    if not videos:
        print(f"No video files found in {directory}")
        return

    n = len(videos)
    processed = 0
    skipped = 0
    failed = 0

    for i, video_path in enumerate(videos, start=1):
        gaffe_json = output_dir / f"{video_path.stem}_gaffe.json"
        prefix = f"[{i}/{n}] {video_path.name}"

        if gaffe_json.exists():
            print(f"{prefix} — skipped")
            skipped += 1
            continue

        print(f"{prefix} — processing...")
        try:
            if method == "landmark":
                _run_landmark(
                    video_path=video_path,
                    output_path=gaffe_json,
                    no_display=True,
                    fps_override=args.fps_override,
                )
            else:
                _run_fer(
                    video_path=video_path,
                    output_path=gaffe_json,
                    no_display=True,
                    fps_override=args.fps_override,
                    use_mtcnn=args.use_mtcnn,
                    sample_every=args.sample_every,
                    save_annotated=False,
                    annotated_path=None,
                )
            processed += 1
        except RuntimeError as exc:
            print(f"{prefix} — ERROR: {exc}")
            failed += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────


def _add_shared_options(p: argparse.ArgumentParser) -> None:
    """Add options common to both single and batch subcommands."""
    p.add_argument(
        "--method",
        choices=["landmark", "fer"],
        default="landmark",
        help="Detection pipeline (default: landmark)",
    )
    p.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="Override the detected video FPS",
    )
    fer_group = p.add_argument_group("FER-specific options (--method fer)")
    fer_group.add_argument(
        "--use-mtcnn",
        action="store_true",
        help="Use MTCNN face detector (slower, more accurate)",
    )
    fer_group.add_argument(
        "--sample-every",
        type=int,
        default=1,
        metavar="N",
        help="Analyse every Nth frame, record the rest as no-face (default: 1)",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Supports subcommands ``single`` and ``batch``.  If the first positional
    argument is neither of those it is treated as the video path and the
    ``single`` subcommand is assumed (backward-compatibility mode).
    """
    import sys as _sys
    raw = argv if argv is not None else _sys.argv[1:]

    # Backward-compat: if first token is not a known subcommand, inject "single"
    known_subcommands = {"single", "batch"}
    if raw and raw[0] not in known_subcommands:
        raw = ["single"] + list(raw)

    parser = argparse.ArgumentParser(
        description="GAFFE — Stress detection from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ── single ────────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="Process a single video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_single.add_argument("video", type=str, help="Path to the input video file")
    p_single.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/<video_stem>/<video_stem>_gaffe.json)",
    )
    p_single.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without showing the video window",
    )
    _add_shared_options(p_single)
    p_single.add_argument(
        "--save-annotated",
        action="store_true",
        help="(FER only) Save an MP4 with the stress overlay drawn on each frame",
    )
    p_single.add_argument(
        "--annotated-output",
        type=str,
        default=None,
        help="(FER only) Path for the annotated video "
             "(default: results/<stem>/<stem>_fer_annotated.mp4)",
    )

    # ── batch ─────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Process all videos in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_batch.add_argument("directory", type=str, help="Directory containing video files")
    p_batch.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for GAFFE JSONs (default: results/<directory_name>/)",
    )
    _add_shared_options(p_batch)

    return parser.parse_args(raw)


def _cmd_single(args: argparse.Namespace) -> None:
    """Handle the ``single`` subcommand."""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else _default_output_path(video_path)

    try:
        if args.method == "landmark":
            _run_landmark(
                video_path=video_path,
                output_path=output_path,
                no_display=args.no_display,
                fps_override=args.fps_override,
            )
        else:  # fer
            if args.save_annotated:
                if args.annotated_output:
                    annotated_path = Path(args.annotated_output)
                else:
                    ann_dir = Path("results") / video_path.stem
                    ann_dir.mkdir(parents=True, exist_ok=True)
                    annotated_path = ann_dir / f"{video_path.stem}_fer_annotated.mp4"
            else:
                annotated_path = None

            _run_fer(
                video_path=video_path,
                output_path=output_path,
                no_display=args.no_display,
                fps_override=args.fps_override,
                use_mtcnn=args.use_mtcnn,
                sample_every=args.sample_every,
                save_annotated=args.save_annotated,
                annotated_path=annotated_path,
            )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.subcommand == "single":
        _cmd_single(args)
    else:
        _cmd_batch_detect(args)


if __name__ == "__main__":
    main()
