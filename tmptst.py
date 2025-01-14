import cv2
import mediapipe as mp
import pyvirtualcam
from pygrabber.dshow_graph import FilterGraph

# Mediapipe initialisieren
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


def select_camera():
    graph = FilterGraph()
    cameras = graph.get_input_devices()
    if not cameras:
        print("No cameras found.")
        return None
    print("Available cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i}: {cam}")
    cam_index = int(input("Select camera index: "))
    return cam_index, cameras[cam_index]


# Initialize selected webcam
selected_camera_index, selected_camera_name = select_camera()
if selected_camera_index is not None:
    cap = cv2.VideoCapture(selected_camera_index)
    print(f"Camera {selected_camera_name} initialized.")
else:
    print("No camera selected.")

## end cam grab


if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get the camera's resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps,
                         fmt=pyvirtualcam.PixelFormat.BGR) as virtual_cam:


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
        virtual_cam.send(frame)
        virtual_cam.sleep_until_next_frame()
        # Mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
