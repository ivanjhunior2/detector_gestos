"""Launch gesture-to-key typing in dry-run or keyboard-enabled mode."""

import argparse

from gesture_detector import __version__


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconoce sellos y una palma abierta para enviar Q, W, E, R, D o F."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--send-keys",
        action="store_true",
        help="Envía pulsaciones al programa que tenga el foco; por defecto solo simula.",
    )
    arguments = parser.parse_args()
    from gesture_detector.app import run_gesture_typing

    return run_gesture_typing(send_keys=arguments.send_keys)


if __name__ == "__main__":
    raise SystemExit(main())
