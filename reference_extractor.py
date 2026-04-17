import cv2
import face_recognition
from collections import Counter
import numpy as np

class AutoReferenceExtractor:
    
    def extract_main_face_from_video(self, video_path, sample_interval=1):
        # sample_interval si può estendere per avere maggiore efficienza
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # CAP_PROP_FRAME_COUNT retrieves the total number of frames in a video file
        
        # Dizionario: encoding -> contatore apparizioni
        face_counter = []
        face_images = []  # salva un'immagine di esempio per ogni volto unico
        known_face_encodings = []
        
        frame_idx = 0
        print(f"Analisi {total_frames} frame per identificare il soggetto principale...")
        
        for i in range(0, total_frames):
            ret, frame = cap.read()    # cap.read() returns a bool (true if the frame has been read properly) and the frame itself
            if not ret:
                break
            
            
            
            if frame_idx % sample_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #nome_file = f"frame_{i:02d}.jpg"
                #cv2.imwrite(nome_file, rgb)
                #print("Frame saved successfully!")
                
                # find faces
                face_locations = face_recognition.face_locations(rgb)         
                # detects the presence and locations of human faces in a given image; it takes a numpy array representing an image in RGB color format and returns a list of tuples, where each tuple contains the coordinates of a detected face
                face_encodings = face_recognition.face_encodings(rgb, face_locations)
                # converts the facial features of detected faces in an image into a compact numerical representation. It generates a 128-dimensional vector (a list of 128 numbers) for each face, which acts as a unique signature for that person. 
                

                #if (len(face_encodings)>0):
                    #print("Abbiamo ", {len(face_encodings)}, " facce")
                    #print(face_encodings[0])

                

                for encoding in face_encodings:
                    trovato = False

                    # da rivedere 
                    for index, known_encoding in enumerate(known_face_encodings):
                        risultato = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.6)

                        if risultato[0]:
                            # Se la persona esiste già, aggiorna magari l'immagine di riferimento (opzionale)
                            face_images[index] = frame 
                            trovato = True
                            face_counter[index] += 1 
                            break # Esci dal ciclo: abbiamo trovato chi è
                    
                    # 2. Se NON è stato trovato, aggiungilo come nuova persona
                    if not trovato:
                        known_face_encodings.append(encoding) # Salva l'encoding per i prossimi confronti
                        face_images.append(frame)             # Salva il frame corrispondente
                        face_counter.append(1)
                        print(f"Nuova persona rilevata! Totale: {len(known_face_encodings)}")
            
            frame_idx += 1
        

        cap.release()
        
        if not face_counter:
            raise ValueError("Nessun volto trovato nel video")
        
        # Trova il volto più frequente (soggetto principale)
        idx = face_counter.index(max(face_counter))
        
        print(f"\nSoggetto principale identificato!")
        print(f"Apparso in {face_counter[idx]} frame su {frame_idx // sample_interval} analizzati")
        
        return face_images[idx]







# Utilizzo
extractor = AutoReferenceExtractor()
reference_image = extractor.extract_main_face_from_video("YW_WILTY_EP70_truth15.mp4")

# Salva la reference image per future analisi
cv2.imwrite("reference_face.jpg", reference_image)