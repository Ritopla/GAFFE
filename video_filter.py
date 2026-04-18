import cv2
import face_recognition
import numpy as np

class VideoFaceFilter:

    def __init__(self, main_encodings, tolerance=0.6, resize_scale=0.25):
        self.main_encodings = main_encodings
        self.tolerance = tolerance
        self.resize_scale = resize_scale


    def frame_contains_subject(self, frame):
        # Resize per velocità
        small = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect
        face_locations = face_recognition.face_locations(rgb)
        if not face_locations:
            return False

        # Encode
        encodings = face_recognition.face_encodings(rgb, face_locations)

        # Check match
        for enc in encodings:
            distances = face_recognition.face_distance(self.main_encodings, enc)

            # 👇 usa la distanza minima
            if np.min(distances) < self.tolerance:
                return True

        return False


    def filter_video(self, video_path, sample_interval=1):
        cap = cv2.VideoCapture(video_path)
        kept_frames = []
        total = 0
        kept = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if total % sample_interval == 0:
                if self.frame_contains_subject(frame):
                    kept_frames.append(frame)
                    kept += 1

            total += 1

        cap.release()

        print(f"Frame totali analizzati: {total}")
        print(f"Frame mantenuti: {kept}")

        return kept_frames
    


    def frames_to_video(self, frames, output_path, fps=25):
        if len(frames) == 0:
            raise ValueError("No frames")

        h, w, _ = frames[0].shape

        # forza consistenza dimensioni
        frames = [cv2.resize(f, (w, h)) for f in frames]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # prova H264

        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not out.isOpened():
            raise RuntimeError("VideoWriter non aperto: codec non supportato")

        for f in frames:
            out.write(f)

        out.release()