# -*- coding: utf-8 -*-

"""
Lo script fa queste cose in sequenza:
1. Inizializzazione
Carica FER con Haarcascade come face detector interno. Niente dlib, niente landmark — FER gestisce tutto da sola.
2. Analisi video frame per frame
Per ogni frame rileva se c'è un volto. Se lo trova, FER classifica tutte e 7 le emozioni (angry, fear, disgust, happy, sad, surprise, neutral) su scala 0-1. Calcola lo stress come (angry + fear + disgust) * 100. Disegna il bounding box del volto e sovrappone percentuale stress + barra colorata sul frame.
3. Output

CSV con tutti i dati frame per frame — tutte e 7 le emozioni salvate separatamente, stress score, timestamp, flag face_detected
Video annotato con l'overlay dello stress visibile in tempo reale
Grafico con due subplot: andamento temporale dello stress + istogramma distribuzione

4. Riproduzione
Apre il video annotato in una finestra OpenCV con controlli base (pausa, avanti/indietro 5s, esci).

"""


"""
Stress Detection su Video — versione FER only
Analizza il livello di stress frame per frame da un video
usando esclusivamente la libreria FER (Facial Expression Recognition).

REQUISITI:
    pip install fer opencv-python pandas matplotlib tensorflow

NOTE:
    - FER gestisce internamente il rilevamento del volto (Haarcascade di default)
    - Lo stress è calcolato come: angry + fear + disgust (scala 0-100)
    - Tutte e 7 le emozioni vengono salvate nel CSV per analisi successive
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fer.fer import FER


# ==============================================================================
# CONFIGURAZIONE — modifica questi parametri
# ==============================================================================

VIDEO_PATH            = "video.mp4"           # path al video da analizzare
SAMPLE_EVERY_N_FRAMES = 1                     # 1 = tutti i frame; aumenta per velocizzare
SAVE_ANNOTATED_VIDEO  = True                  # salva il video con overlay stress

OUTPUT_CSV   = "stress_results.csv"
OUTPUT_VIDEO = "stress_output.mp4"
OUTPUT_PLOT  = "stress_plot.png"

# Face detector interno a FER: False = Haarcascade (veloce), True = MTCNN (preciso)
USE_MTCNN = False

# ==============================================================================


def stress_color(level: int) -> tuple:
    """Restituisce un colore BGR in base al livello di stress."""
    if level < 33:
        return (0, 200, 0)      # verde  — basso
    elif level < 66:
        return (0, 165, 255)    # arancio — medio
    else:
        return (0, 0, 220)      # rosso  — alto


def draw_overlay(frame: np.ndarray, stress: int, timestamp: float) -> None:
    """Disegna percentuale stress e barra colorata sul frame."""
    color   = stress_color(stress)
    cv2.putText(frame, f"Stress: {stress}%", (20, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
    cv2.putText(frame, f"t={timestamp:.1f}s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    bar_len = int(stress * 2)
    cv2.rectangle(frame, (20, 95),  (220, 110), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 95),  (20 + bar_len, 110), color, -1)


def compute_stress(emotions: dict) -> float:
    """
    Calcola il punteggio di stress da un dizionario FER.
    Proxy: angry + fear + disgust, scalato 0-100.
    """
    raw = (
        emotions.get("angry",   0) +
        emotions.get("fear",    0) +
        emotions.get("disgust", 0)
    )
    return round(raw * 100, 1)


def analyze_video(
    video_path:    str,
    sample_every:  int  = 1,
    use_mtcnn:     bool = False,
    save_video:    bool = True,
    output_csv:    str  = "stress_results.csv",
    output_video:  str  = "stress_output.mp4",
) -> pd.DataFrame:
    """
    Analizza un video frame per frame e restituisce un DataFrame con:
    - frame, time_s
    - angry, disgust, fear, happy, sad, surprise, neutral  (tutte le emozioni FER, 0-1)
    - stress_score  (angry+fear+disgust * 100)
    - face_detected
    """

    if not os.path.exists(video_path):
        sys.exit(f"[ERRORE] Video non trovato: {video_path}")

    # --- Inizializza FER (gestisce internamente face detection + CNN emozioni) ---
    print(f"Inizializzazione FER (MTCNN={'MTCNN' if use_mtcnn else 'Haarcascade'})...")
    fer_detector = FER(mtcnn=use_mtcnn)
    print("FER pronto.")

    # --- Apertura video ---
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {w}x{h} @ {fps:.1f} fps — {total_frames} frame (~{total_frames/fps:.1f}s)")

    # --- VideoWriter opzionale ---
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    records   = []
    frame_idx = 0
    analyzed  = 0

    print("\nAvvio analisi...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp     = frame_idx / fps
        face_detected = False
        emotions_row  = {e: None for e in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
        stress_score  = None

        if frame_idx % sample_every == 0:
            # FER: rileva volti e classifica emozioni in un solo passaggio
            result = fer_detector.detect_emotions(frame)

            if result:
                face_detected = True
                # Prende il primo volto trovato
                emotions      = result[0]["emotions"]
                box           = result[0]["box"]   # [x, y, w, h]

                # Salva tutte le emozioni
                for key in emotions_row:
                    emotions_row[key] = round(emotions.get(key, 0), 4)

                # Calcola stress
                stress_score = compute_stress(emotions)

                # Disegna bounding box del volto
                x, y, bw, bh = box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1)

                # Overlay stress sul frame
                draw_overlay(frame, int(stress_score), timestamp)

            analyzed += 1

        records.append({
            "frame":        frame_idx,
            "time_s":       round(timestamp, 2),
            **emotions_row,
            "stress_score": stress_score,
            "face_detected": face_detected,
        })

        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  Progresso: {pct:.1f}%  ({frame_idx}/{total_frames})", end="\r")

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\n\nAnalisi completata.")
    print(f"  Frame totali  : {frame_idx}")
    print(f"  Frame analizzati : {analyzed}")
    print(f"  CSV salvato   : {output_csv}")
    if save_video:
        print(f"  Video salvato : {output_video}")

    return df


def print_stats(df: pd.DataFrame) -> None:
    """Stampa statistiche riepilogative."""
    df_v = df[df["face_detected"] == True].dropna(subset=["stress_score"])

    print("\n=== STATISTICHE STRESS ===")
    print(f"Frame con volto rilevato : {len(df_v)} / {len(df)} ({int(len(df_v)/len(df)*100)}%)")
    if df_v.empty:
        print("  Nessun volto rilevato nel video.")
        return
    print(f"Stress medio             : {df_v['stress_score'].mean():.1f}%")
    print(f"Stress massimo           : {df_v['stress_score'].max():.1f}%")
    print(f"Stress minimo            : {df_v['stress_score'].min():.1f}%")
    print(f"\nEmozioni medie (frame con volto):")
    for em in ["angry", "fear", "disgust", "happy", "sad", "surprise", "neutral"]:
        print(f"  {em:<10}: {df_v[em].mean():.3f}")


def plot_results(df: pd.DataFrame, save_path: str = "stress_plot.png") -> None:
    """Genera grafico andamento stress + istogramma distribuzione."""
    df_v = df[df["face_detected"] == True].dropna(subset=["stress_score"])

    if df_v.empty:
        print("Nessun dato da plottare.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor("#0f0f0f")

    # --- Andamento temporale ---
    ax = axes[0]
    ax.set_facecolor("#0f0f0f")

    t = df_v["time_s"].values
    s = df_v["stress_score"].values

    ax.fill_between(t, s, alpha=0.15, color="#DC2626")
    ax.plot(t, s, color="#DC2626", linewidth=1.8, label="Stress (angry+fear+disgust)")

    ax.axhline(33, color="#16A34A", linewidth=1.2, linestyle="--", alpha=0.7, label="Soglia bassa (33)")
    ax.axhline(66, color="#DC2626", linewidth=1.2, linestyle="--", alpha=0.7, label="Soglia alta (66)")
    ax.axhspan(0,  33,  alpha=0.04, color="#16A34A")
    ax.axhspan(66, 105, alpha=0.04, color="#DC2626")

    # Annotazione picco
    idx_max = df_v["stress_score"].idxmax()
    t_max   = df_v.loc[idx_max, "time_s"]
    s_max   = df_v.loc[idx_max, "stress_score"]
    ax.annotate(f"picco {s_max:.0f}%", xy=(t_max, s_max),
                xytext=(t_max + max(t) * 0.02, s_max + 6),
                fontsize=8, color="#ffffff",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8))

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Tempo (s)", color="#888888")
    ax.set_ylabel("Stress score (0-100)", color="#888888")
    ax.set_title("Andamento Stress nel Tempo", color="#cccccc")
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#cccccc", fontsize=9)

    # --- Istogramma ---
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f0f")

    mean_val = df_v["stress_score"].mean()
    ax2.hist(df_v["stress_score"].dropna(), bins=20, color="#DC2626", alpha=0.7, edgecolor="#222222")
    ax2.axvline(mean_val, color="#ffffff", linestyle="--", linewidth=1.5,
                label=f"Media: {mean_val:.1f}%")

    ax2.set_xlabel("Stress score", color="#888888")
    ax2.set_ylabel("Frequenza (frame)", color="#888888")
    ax2.set_title("Distribuzione Livelli di Stress", color="#cccccc")
    ax2.tick_params(colors="#666666")
    for spine in ax2.spines.values():
        spine.set_color("#333333")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#cccccc", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nGrafico salvato: {save_path}")
    plt.show()


def play_annotated_video(video_path: str, window_name: str = "Stress Detection") -> None:
    """
    Riproduce il video annotato in una finestra OpenCV.
    SPAZIO = pausa/riprendi | A/← = -5s | D/→ = +5s | Q/ESC = esci
    """
    if not os.path.exists(video_path):
        print(f"[ERRORE] Video non trovato: {video_path}")
        return

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay        = max(1, int(1000 / fps))

    print(f"\nRiproduzione: {video_path}")
    print("  SPAZIO = pausa/riprendi  |  A/← = -5s  |  D/→ = +5s  |  Q/ESC = esci\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    paused    = False
    frame_idx = 0
    frame     = None

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame is not None:
            display  = frame.copy()
            h_f, w_f = display.shape[:2]
            progress = frame_idx / total_frames if total_frames > 0 else 0
            bar_x    = int(progress * w_f)

            cv2.rectangle(display, (0, h_f - 12), (w_f, h_f), (30, 30, 30), -1)
            cv2.rectangle(display, (0, h_f - 12), (bar_x, h_f), (0, 180, 255), -1)

            time_text = f"{frame_idx/fps:.1f}s / {total_frames/fps:.1f}s"
            cv2.putText(display, time_text, (w_f - 130, h_f - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            if paused:
                cv2.putText(display, "|| PAUSA", (10, h_f - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow(window_name, display)

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("d"), 83):
            new_frame = min(frame_idx + int(fps * 5), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            frame_idx = new_frame
        elif key in (ord("a"), 81):
            new_frame = max(frame_idx - int(fps * 5), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            frame_idx = new_frame

    cap.release()
    cv2.destroyAllWindows()
    print("Riproduzione terminata.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    df = analyze_video(
        video_path   = VIDEO_PATH,
        sample_every = SAMPLE_EVERY_N_FRAMES,
        use_mtcnn    = USE_MTCNN,
        save_video   = SAVE_ANNOTATED_VIDEO,
        output_csv   = OUTPUT_CSV,
        output_video = OUTPUT_VIDEO,
    )

    print_stats(df)
    plot_results(df, save_path=OUTPUT_PLOT)
    play_annotated_video(OUTPUT_VIDEO)