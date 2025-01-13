import cv2
import mediapipe as mp

# Mediapipe initialisieren
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Webcam initialisieren
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Bild in RGB konvertieren (Mediapipe erwartet RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Gesichtserkennung durchführen
    results = face_mesh.process(rgb_frame)

    # Gesichtspunkte zeichnen
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Bild anzeigen
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Mit 'q' beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
