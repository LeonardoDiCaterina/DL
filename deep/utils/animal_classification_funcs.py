import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk  # Cross-platform way to get screen resolution


def get_screen_size():
    """Returns the screen width and height using tkinter (cross-platform)."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def show_image(image, title):
    """
    Displays the image resized to fit within the screen size while preserving aspect ratio.
    Waits for the user to press any key to close the image window.
    """

    # Convert TensorFlow tensor to NumPy array (if needed)
    if not isinstance(image, np.ndarray):
        image = image.numpy()

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get screen dimensions
    screen_width, screen_height = get_screen_size()

    # Image dimensions
    img_height, img_width = image_bgr.shape[:2]

    # Determine scaling factor
    width_ratio = screen_width / img_width
    height_ratio = screen_height / img_height
    scale = min(width_ratio, height_ratio, 1.0)  # Don't upscale

    # Resize if necessary
    if scale < 1.0:
        new_size = (int(img_width * scale), int(img_height * scale))
        image_bgr = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)

    # Show image
    cv2.imshow(title, image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()