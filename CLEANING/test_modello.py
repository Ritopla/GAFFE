import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('AN_WILTY_EP16_lie4.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # fine video

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
        break

cap.release()
cv2.destroyAllWindows()