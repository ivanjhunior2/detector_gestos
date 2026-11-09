"""Controlled keyboard output; dry-run is the default."""

from gesture_detector.config import GESTURE_TO_KEY


class KeyboardController:
    """Send a one-shot keypress or report it without sending input."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._pyautogui = None

    def press_for_gesture(self, gesture: str) -> str:
        try:
            key = GESTURE_TO_KEY[gesture]
        except KeyError as error:
            raise ValueError(f"No hay tecla asignada al gesto {gesture!r}.") from error

        if not self.enabled:
            return f"SIMULACIÓN: {gesture} → {key.upper()}"

        self.press_key(key)
        return f"PULSACIÓN: {gesture} → {key.upper()}"

    def press_key(self, key: str) -> None:
        """Send one press and release to the application with focus."""
        if not self.enabled:
            return

        if self._pyautogui is None:
            try:
                import pyautogui
            except ImportError as error:
                raise RuntimeError(
                    "PyAutoGUI no está instalado en el entorno activo. "
                    "Instala las dependencias del proyecto."
                ) from error

            # Keep PyAutoGUI's built-in corner fail-safe enabled.
            pyautogui.FAILSAFE = True
            self._pyautogui = pyautogui

        try:
            self._pyautogui.press(key)
        except self._pyautogui.FailSafeException as error:
            raise RuntimeError(
                "PyAutoGUI activó su fail-safe; mueve el cursor fuera de una "
                "esquina y reinicia la aplicación."
            ) from error
