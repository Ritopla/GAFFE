import cv2
import face_recognition
from collections import Counter
import numpy as np

class AutoReferenceExtractor:

    def __init__(self):
        pass

    
    def extract_main_face_from_video(self, video_path, sample_interval=10):
        # sample_interval si può estendere per avere maggiore efficienza
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # CAP_PROP_FRAME_COUNT retrieves the total number of frames in a video file
        
        face_counter = []
        face_images = []  # salva un'immagine di esempio per ogni volto unico
        known_face_encodings = []
        encodings_per_person = []
        
        frame_idx = 0
        print(f"Analisi {total_frames} frame per identificare il soggetto principale...")
        
        for i in range(0, total_frames):
            ret, frame = cap.read()    # cap.read() returns a bool (true if the frame has been read properly) and the frame itself
            if not ret:
                break
            
            
            # if inserito qualora si volessero analizzare un sottoinsieme dei frame
            if frame_idx % sample_interval == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                #nome_file = f"frame_{i:02d}.jpg"
                #cv2.imwrite(nome_file, rgb)
                #print("Frame saved successfully!")
                
                # find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)         
                # detects the presence and locations of human faces in a given image; it takes a numpy array representing an image in RGB color format and returns a list of tuples, where each tuple contains the coordinates of a detected face


                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                # converts the facial features of detected faces in an image into a compact numerical representation. It generates a 128-dimensional vector (a list of 128 numbers) for each face, which acts as a unique signature for that person. 
                

                #if (len(face_encodings)>0):
                    #print("Abbiamo ", {len(face_encodings)}, " facce")
                    #print(face_encodings[0])

                

                for encoding in face_encodings:
                    trovato = False

                    for index, known_encoding in enumerate(known_face_encodings):
                        result = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.6)
                        #Studi su embedding mostrano che testi casuali e non correlati tendono ad avere cosine similarity mediamente intorno a 0.34-0.35 . Questo significa che 0.6 è già abbastanza sopra la "media del rumore di fondo". Dai testi, il concetto è stato trasportato alle immagini

                        # risultato è una lista di bool anche se ha size 1
                        if result[0]:
                            # Se la persona esiste già, aggiorna magari l'immagine di riferimento (opzionale)
                            face_images[index] = frame 
                            trovato = True
                            encodings_per_person[index].append(encoding)
                            face_counter[index] += 1 
                            break # Esci dal ciclo: abbiamo trovato chi è
                    
                    # 2. Se NON è stato trovato, aggiungilo come nuova persona
                    if not trovato:
                        known_face_encodings.append(encoding) # Salva l'encoding per i prossimi confronti
                        face_images.append(frame)             # Salva il frame corrispondente
                        face_counter.append(1)
                        encodings_per_person.append([encoding])
                        print(f"Nuova persona rilevata! Totale: {len(known_face_encodings)}")
            
            frame_idx += 1
        

        cap.release()
        
        if not face_counter:
            raise ValueError("Nessun volto trovato nel video")
        
        # Trova il volto più frequente (soggetto principale)
        idx = face_counter.index(max(face_counter))
        
        print(f"\nSoggetto principale identificato!")
        print(f"Apparso in {face_counter[idx]} frame su {frame_idx // sample_interval} analizzati")
        
        return face_images[idx], encodings_per_person[idx]
