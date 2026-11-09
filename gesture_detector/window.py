"""OpenCV window helpers shared by the application's loops."""

import cv2


def window_is_open(window_name: str) -> bool:
    """Return False after the user closes an OpenCV window with its X button."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        # OpenCV may raise once the native window has already been destroyed.
        return False
