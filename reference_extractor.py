import cv2
import face_recognition
from collections import Counter
import numpy as np

class AutoReferenceExtractor:
    """
    Estrae automaticamente il volto del soggetto principale dal video
    senza bisogno di una foto esterna
    """
    
    def extract_main_face_from_video(self, video_path, sample_interval=5):
        """
        Analizza il video e identifica il volto che appare più spesso
        sample_interval: analizza 1 frame ogni N per velocizzare
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Dizionario: encoding -> contatore apparizioni
        face_counter = {}
        face_images = {}  # salva un'immagine di esempio per ogni volto unico
        
        frame_idx = 0
        print(f"Analisi {total_frames} frame per identificare il soggetto principale...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analisi ogni N frame
            if frame_idx % sample_interval == 0:
                # Riduci risoluzione per velocizzare
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                # Trova volti
                face_locations = face_recognition.face_locations(rgb)
                face_encodings = face_recognition.face_encodings(rgb, face_locations)
                
                for i, encoding in enumerate(face_encodings):
                    # Converti encoding in tuple per usarlo come chiave
                    enc_tuple = tuple(encoding)
                    
                    # Conta occorrenze
                    if enc_tuple not in face_counter:
                        # Salva l'immagine del volto (scalata alla dimensione originale)
                        top, right, bottom, left = face_locations[i]
                        # Scala indietro le coordinate (perché abbiamo ridotto del 50%)
                        top *= 2
                        right *= 2
                        bottom *= 2
                        left *= 2
                        face_img = frame[top:bottom, left:right]
                        face_images[enc_tuple] = face_img
                    
                    face_counter[enc_tuple] = face_counter.get(enc_tuple, 0) + 1
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Analizzati {frame_idx}/{total_frames} frame")
        
        cap.release()
        
        if not face_counter:
            raise ValueError("Nessun volto trovato nel video")
        
        # Trova il volto più frequente (soggetto principale)
        main_face_enc = max(face_counter, key=face_counter.get)
        main_face_image = face_images[main_face_enc]
        
        print(f"\nSoggetto principale identificato!")
        print(f"Apparso in {face_counter[main_face_enc]} frame su {frame_idx // sample_interval} analizzati")
        
        return main_face_image, main_face_enc

# Utilizzo
extractor = AutoReferenceExtractor()
reference_image, reference_encoding = extractor.extract_main_face_from_video("YW_WILTY_EP70_truth15.mp4")

# Salva la reference image per future analisi
#cv2.imwrite("reference_face.jpg", reference_image)
print(reference_image)