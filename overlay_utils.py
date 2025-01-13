import cv2


def blur_image(frame, kernel_size=(15, 15)):
    return cv2.GaussianBlur(frame, kernel_size, 0)


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


def apply_cover_img_on_face(frame, face, cover_scaling, cover_img, eye_scaling, eye_color, eye_alpha):
    for (x, y, w, h) in face:

        new_width = int(w * cover_scaling)
        new_height = int(h * cover_scaling)
        # Separate the alpha channel and the color channels
        cover_alpha = cover_img[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
        cover_color = cover_img[:, :, :3]

        # Adjust coordinates to center the cover over the face
        x_center = x + w // 2
        y_center = y + h // 2
        x_start = x_center - new_width // 2
        y_start = y_center - new_height // 2

        eye_width = max(1,int(w * eye_scaling))
        eye_height = max(1, int(h * eye_scaling))

        # Eye positions relative to the face
        left_eye_x = x_start + int(new_width * 0.3)
        left_eye_y = y_start + int(new_height * 0.25)

        right_eye_x = x_start + int(new_width * 0.7) - eye_width
        right_eye_y = y_start + int(new_height * 0.25)

        eye_resized = cv2.resize(eye_color, (eye_width, eye_height), interpolation=cv2.INTER_AREA)
        eye_resized_alpha = cv2.resize(eye_alpha, (eye_width, eye_height), interpolation=cv2.INTER_AREA)

        # Resize the cover image and alpha mask
        aubergine_resized = cv2.resize(cover_color, (new_width, new_height), interpolation=cv2.INTER_AREA)
        aubergine_resized_alpha = cv2.resize(cover_alpha, (new_width, new_height), interpolation=cv2.INTER_AREA)
        frame = overlay_image_alpha(frame, aubergine_resized, x_start, y_start, aubergine_resized_alpha)

        frame = overlay_image_alpha(frame, eye_resized, left_eye_x, left_eye_y, eye_resized_alpha)
        frame = overlay_image_alpha(frame, eye_resized, right_eye_x, right_eye_y, eye_resized_alpha)

    return frame

