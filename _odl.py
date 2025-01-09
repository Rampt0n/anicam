import cv2
import pyvirtualcam

# Initialize the camera (default: 0)
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get the camera's resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Camera resolution: {frame_width}x{frame_height}, FPS: {fps}")
print("Press 'q' to quit.")

# Start the virtual camera
with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as virtual_cam:
    print(f"Virtual camera started: {virtual_cam.device}")

    while True:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert single-channel grayscale back to 3-channel image
        processed_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Send frame to the virtual camera
        virtual_cam.send(processed_frame)
        virtual_cam.sleep_until_next_frame()

        # Optionally display the processed frame (for debugging)
        cv2.imshow('Black & White Output', processed_frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
