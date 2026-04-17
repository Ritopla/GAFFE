"""
pipeline:

1. se possibile, aumentare qualità delle immagini con filtri 
2. individuare frame adeguati in cui i 68 punti sono pienamente visibili
3. trovare metriche adeguate

"""

# ---- PARAMETRI ----
SAMPLE_EVERY_N_FRAMES = 3    # Analizza 1 frame ogni N (velocità vs dettaglio)
USE_FER = True               # Usa anche FER per la classificazione emozioni
SAVE_ANNOTATED_VIDEO = True  # Salva video con stress overlay
OUTPUT_CSV = 'stress_results.csv'
OUTPUT_VIDEO = 'stress_output.mp4'
# -------------------


import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

if USE_FER:
    from fer.fer import FER # Changed import statement
    fer_detector = FER(mtcnn=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')




"""## 5. Analisi del video"""

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))