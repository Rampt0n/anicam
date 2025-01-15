import cv2
import mediapipe as mp
import pyvirtualcam
from overlay_utils import blur_image
from src.general_utils import  select_camera

# Mediapipe initialisieren
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
cover_img = cv2.imread(r"images\eggplant.png", cv2.IMREAD_UNCHANGED)
last_result = None

def overlay_image(background, overlay, position, scale=1.0):
    """
    Overlays a transparent image onto a background at the given position and scale.
    :param background: The background image (BGR).
    :param overlay: The overlay image (BGRA with transparency).
    :param position: Tuple (x, y) for the top-left corner of the overlay.
    :param scale: Scale factor for the overlay image.
    :return: Background image with overlay.
    """
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w, _ = overlay.shape
    x, y = position

    # Ensure overlay is within frame boundaries
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background

    alpha_overlay = overlay[:, :, 3] / 255.0  # Extract alpha channel and normalize
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):  # Blend BGR channels
        background[y:y + h, x:x + w, c] = (
            alpha_overlay * overlay[:, :, c] + alpha_background * background[y:y + h, x:x + w, c]
        )

    return background


# Initialize selected webcam
selected_camera_index, selected_camera_name = 0, "cam"  #select_camera()
if selected_camera_index is not None:
    cap = cv2.VideoCapture(selected_camera_index)
    print(f"Camera {selected_camera_name} initialized.")
else:
    print("No camera selected.")

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

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks and last_result:
            results = last_result
        print("results.multi_face_landmarks ", results.multi_face_landmarks)
        if results.multi_face_landmarks:
            last_result = results

            for face_landmarks in results.multi_face_landmarks:

                # Get the bounding box of the face based on landmarks
                x_coords = [int(landmark.x * frame_width) for landmark in face_landmarks.landmark]
                y_coords = [int(landmark.y * frame_height) for landmark in face_landmarks.landmark]

                # Determine position and size for the aubergine
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                face_width = max_x - min_x
                face_height = max_y - min_y

                # Overlay position: top of the face
                aubergine_position = (min_x, min_y - face_height // 2)
                aubergine_scale = face_width / cover_img.shape[1]

                frame = blur_image(frame)
                # Overlay the aubergine image
                frame = overlay_image(frame, cover_img, aubergine_position, scale=aubergine_scale)






        # Display or stream the frame
        cv2.imshow('Aubergine Overlay', frame)
        virtual_cam.send(frame)
        virtual_cam.sleep_until_next_frame()

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()