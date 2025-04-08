"""Useful methods to display images."""

import cv2
import numpy as np
import tkinter as tk
from typing import Tuple, Union, Any

def get_screen_size(
) -> Tuple[int, int]:
    """
    Returns the screen width and height using tkinter (cross-platform).

    Returns:
        Tuple[int, int]: (width, height) of the screen.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

def show_image(image, title):
    '''
    Args:
        image: image file
        title: title of the image
    
    Plots the image, with a title.
    
    '''

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Read the image
    image = mpimg.imread(image)

    # Plot the image
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_resized_await(
    image: Union[np.ndarray, Any],
    title: str
) -> None:
    """
    Displays the image resized to fit within the screen while preserving aspect ratio.
    Waits for the user to press any key to close the image window.

    Args:
        image (Union[np.ndarray, Any]): The image to display.
        title (str): Window title for the image display.
    """
    # Convert TensorFlow tensor to NumPy array (if needed)
    if not isinstance(image, np.ndarray):
        image = image.numpy()

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get screen dimensions
    screen_width, screen_height = get_screen_size()

    # Image dimensions
    img_height, img_width = image_bgr.shape[:2]

    # Determine scaling factor
    width_ratio = screen_width / img_width
    height_ratio = screen_height / img_height
    scale = min(width_ratio, height_ratio, 1.0)

    # Resize if necessary
    if scale < 1.0:
        new_size = (
            int(img_width * scale),
            int(img_height * scale)
        )
        image_bgr = cv2.resize(
            image_bgr,
            new_size,
            interpolation=cv2.INTER_AREA
        )

    # Show image
    cv2.imshow(title, image_bgr)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Useful methods to display images")
