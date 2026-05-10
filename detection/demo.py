#!/usr/bin/env python3
"""
GAFFE Demo — Analyze a video file for stress indicators.

Usage:
    python demo.py <video_path> [--output results.json] [--no-display]

Processes the video frame-by-frame with MediaPipe Face Mesh,
computes stress metrics using the GAFFE pipeline, and saves
a JSON report with per-frame results + summary statistics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
import numpy as np

from detection.config import StressConfig
from detection.stress_scorer import StressScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAFFE — Geometric stress detection from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <video_name>_gaffe.json)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without showing the video window (headless mode)",
    )
    parser.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="Override the detected video FPS",
    )
    return parser.parse_args()


def extract_landmarks(
    face_landmarker: FaceLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> np.ndarray | None:
    """
    Run MediaPipe FaceLandmarker on a BGR frame.

    Returns:
        (478, 2) array of (x, y) pixel coordinates, or None if no face.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None

    # Take the first (most prominent) face
    face = result.face_landmarks[0]
    landmarks = np.array(
        [(lm.x * w, lm.y * h) for lm in face],
        dtype=np.float64,
    )
    return landmarks


def draw_overlay(
    frame: np.ndarray,
    result: dict,
    landmarks: np.ndarray | None,
) -> np.ndarray:
    """Draw stress metrics overlay on the frame."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    if not result["face_detected"] or landmarks is None:
        cv2.putText(
            overlay, "No face detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return overlay

    stress = result["stress"]
    metrics = result["metrics"]

    # Color based on stress level
    level_colors = {
        "LOW": (0, 200, 0),         # Green
        "MODERATE": (0, 200, 255),   # Yellow-orange
        "HIGH": (0, 100, 255),       # Orange
        "VERY_HIGH": (0, 0, 255),    # Red
    }
    color = level_colors.get(stress["level"], (255, 255, 255))

    # Stress bar
    bar_width = int(stress["score"] * 200)
    cv2.rectangle(overlay, (20, 15), (220, 45), (50, 50, 50), -1)
    cv2.rectangle(overlay, (20, 15), (20 + bar_width, 45), color, -1)
    cv2.putText(
        overlay,
        f"Stress: {stress['score']:.2f} [{stress['level']}]",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )

    # Metric details
    y_offset = 70
    details = [
        f"BFI: {metrics['bfi']['value']:.3f}",
        f"EAR: {metrics['ear']['value']:.3f}  |  Blink: {metrics['ear']['blink_rate_per_min']:.0f}/min",
        f"EAR Stress: {metrics['ear']['ear_stress_score']:.3f}",
        f"BAD: {metrics['bad']['value']:.3f}  (asym={metrics['bad']['components']['asymmetry']:.3f})",
    ]
    for text in details:
        cv2.putText(
            overlay, text, (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
        )
        y_offset += 22

    # Draw key landmarks (brows + eyes)
    key_indices = [55, 285, 70, 300, 159, 386, 33, 133, 263, 362]
    for idx in key_indices:
        if idx < len(landmarks):
            pt = tuple(landmarks[idx].astype(int))
            cv2.circle(overlay, pt, 2, (0, 255, 0), -1)

    return overlay


def compute_summary(frame_results: list[dict]) -> dict:
    """Compute summary statistics over all analyzed frames."""
    detected = [r for r in frame_results if r["face_detected"]]

    if not detected:
        return {
            "total_frames": len(frame_results),
            "face_detected_frames": 0,
            "face_detection_rate": 0.0,
        }

    scores = [r["stress"]["score"] for r in detected]
    levels = [r["stress"]["level"] for r in detected]
    bfi_values = [r["metrics"]["bfi"]["value"] for r in detected]
    ear_values = [r["metrics"]["ear"]["value"] for r in detected]
    ear_stress_values = [r["metrics"]["ear"]["ear_stress_score"] for r in detected]
    bad_values = [r["metrics"]["bad"]["value"] for r in detected]

    level_counts = {}
    for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]:
        count = levels.count(level)
        level_counts[level] = {
            "count": count,
            "percentage": round(count / len(detected) * 100, 1),
        }

    return {
        "total_frames": len(frame_results),
        "face_detected_frames": len(detected),
        "face_detection_rate": round(len(detected) / len(frame_results) * 100, 1),
        "stress_score": {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "median": round(float(np.median(scores)), 4),
        },
        "level_distribution": level_counts,
        "metrics_summary": {
            "bfi_mean": round(float(np.mean(bfi_values)), 4),
            "ear_mean": round(float(np.mean(ear_values)), 4),
            "ear_stress_mean": round(float(np.mean(ear_stress_values)), 4),
            "bad_mean": round(float(np.mean(bad_values)), 4),
        },
    }


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_name(f"{video_path.stem}_gaffe.json")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = args.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.1f}s")
    print(f"Output: {output_path}")
    print("-" * 50)

    # Initialize pipeline
    config = StressConfig(fps=fps)
    scorer = StressScorer(config)

    model_path = Path(__file__).parent.parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        print(
            f"Error: FaceLandmarker model not found at {model_path}\n"
            "Download it with:\n"
            "  curl -L -o face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task",
            file=sys.stderr,
        )
        sys.exit(1)

    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # Process frames
    frame_results: list[dict] = []
    frame_id = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_id / fps) * 1000.0

            # Extract landmarks
            landmarks = extract_landmarks(
                face_landmarker, frame, int(timestamp_ms)
            )

            if landmarks is not None:
                result = scorer.process_frame(landmarks, frame_id, timestamp_ms)
            else:
                result = StressScorer.no_face_result(frame_id, timestamp_ms)

            frame_results.append(result)

            # Display (if not headless)
            if not args.no_display:
                display = draw_overlay(frame, result, landmarks)
                cv2.imshow("GAFFE - Stress Analysis", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

            # Progress
            frame_id += 1
            if frame_id % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_id / elapsed if elapsed > 0 else 0
                pct = (frame_id / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"  Frame {frame_id}/{total_frames} "
                    f"({pct:.0f}%) — {fps_actual:.1f} fps",
                    end="\r",
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        face_landmarker.close()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_id} frames in {elapsed:.1f}s "
          f"({frame_id / elapsed:.1f} fps)")

    # Compute summary
    summary = compute_summary(frame_results)

    # Save output
    output = {
        "metadata": {
            "video_file": str(video_path.name),
            "video_resolution": f"{width}x{height}",
            "video_fps": fps,
            "total_frames_processed": frame_id,
            "pipeline_version": "0.1.0",
        },
        "summary": summary,
        "frames": frame_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
    print(f"\n── Summary ─────────────────────────────────────")
    print(f"  Face detection rate: {summary.get('face_detection_rate', 0)}%")
    if "stress_score" in summary:
        ss = summary["stress_score"]
        print(f"  Stress score — mean: {ss['mean']:.3f}, "
              f"median: {ss['median']:.3f}, "
              f"std: {ss['std']:.3f}")
        print(f"  Level distribution:")
        for level, info in summary.get("level_distribution", {}).items():
            bar = "█" * int(info["percentage"] / 5)
            print(f"    {level:>10}: {info['percentage']:5.1f}% {bar}")


if __name__ == "__main__":
    main()
