import cv2
import pyvirtualcam
from overlay_utils import  blur_image, apply_cover_img_on_face
from pygrabber.dshow_graph import FilterGraph


# Load the cover image with alpha channel (RGBA)
cover_img = cv2.imread("images/aubergine.png", cv2.IMREAD_UNCHANGED)
# Load eye images with alpha channel (RGBA)
eye_img = cv2.imread("images/eye.png", cv2.IMREAD_UNCHANGED)
eye_alpha = eye_img[:, :, 3] / 255.0
eye_color = eye_img[:, :, :3]

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


cover_scaling = 1.5  # Scale aubergine size (1.0 = same size as face, >1 = larger)
eye_scaling = 0.3


# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get the camera's resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

previous_face_location = [[1, 2, 3, 250]]

with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps,
                         fmt=pyvirtualcam.PixelFormat.BGR) as virtual_cam:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        frame = blur_image(frame, kernel_size=(25, 25))
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) < 1:   # if no face is detected, the latest face location will be useds
            face = previous_face_location
        else:
            face = faces
            previous_face_location = faces

        frame = apply_cover_img_on_face(frame, face, cover_scaling, cover_img, eye_scaling, eye_color, eye_alpha)


        virtual_cam.send(frame)
        virtual_cam.sleep_until_next_frame()

        # Show the frame with the cover replacement
        cv2.imshow("Aubergine Face Replacement - quit with  'q'", frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()