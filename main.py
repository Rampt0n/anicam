import cv2
import pyvirtualcam


def overlay_image_alpha(img, overlay, x, y, alpha_mask):
    """
    Overlays `overlay` onto `img` at (x, y) with an alpha mask.
    """
    h, w = overlay.shape[:2]


    # Determine slices for the overlay and background
    slice_y = slice(max(0, y), min(img.shape[0], y + h))
    slice_x = slice(max(0, x), min(img.shape[1], x + w))

    # Adjust overlay and alpha mask to match the available region
    overlay = overlay[: slice_y.stop - slice_y.start, : slice_x.stop - slice_x.start]
    alpha_mask = alpha_mask[: slice_y.stop - slice_y.start, : slice_x.stop - slice_x.start]

    # Extract the region of the background to blend with
    img_part = img[slice_y, slice_x]

    # Blend overlay with the background using the alpha mask
    img[slice_y, slice_x] = (
        alpha_mask[:, :, None] * overlay + (1 - alpha_mask[:, :, None]) * img_part
    )

    return img


# Load the aubergine image with alpha channel (RGBA)
aubergine_img = cv2.imread("aubergine.png", cv2.IMREAD_UNCHANGED)

# Separate the alpha channel and the color channels
aubergine_alpha = aubergine_img[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
aubergine_color = aubergine_img[:, :, :3]

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Scaling factor for the aubergine relative to face size
aubergine_scaling = 1.5  # Scale aubergine size (1.0 = same size as face, >1 = larger)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to quit.")

# Get the camera's resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Adjust aubergine size based on scaling factor
        new_width = int(w * aubergine_scaling)
        new_height = int(h * aubergine_scaling)

        # Adjust coordinates to center the aubergine over the face
        x_center = x + w // 2
        y_center = y + h // 2
        x_start = x_center - new_width // 2
        y_start = y_center - new_height // 2

        # Resize the aubergine image and alpha mask
        aubergine_resized = cv2.resize(aubergine_color, (new_width, new_height), interpolation=cv2.INTER_AREA)
        aubergine_resized_alpha = cv2.resize(aubergine_alpha, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Overlay the aubergine on the face
        frame = overlay_image_alpha(frame, aubergine_resized, x_start, y_start, aubergine_resized_alpha)

    with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps,
                             fmt=pyvirtualcam.PixelFormat.BGR) as virtual_cam:

        virtual_cam.send(frame)
        virtual_cam.sleep_until_next_frame()

    # Show the frame with the aubergine replacement
    cv2.imshow("Aubergine Face Replacement", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
