from pathlib import Path

import cv2

file_path = Path(__file__).parent.resolve()


def overlay_info(image, **kwargs):
    """Overlay key-value information on an image."""
    output_image = image.copy()

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 255)
    thickness = 1

    # Define initial position for the text
    text_x, text_y = 10, 30

    # Get text height for proper spacing
    (w, h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
    line_height = h + baseline

    # Iterate over key-value pairs and overlay text
    for i, (key, value) in enumerate(kwargs.items()):
        text = f"{key}: {value}"
        cv2.putText(
            output_image,
            text,
            (text_x, text_y + i * line_height),
            font,
            font_scale,
            color,
            thickness,
        )

    return output_image
