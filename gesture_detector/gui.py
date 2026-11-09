"""Desktop test interface with a mirrored camera and gesture text field."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from gesture_detector import __version__
from gesture_detector.camera import Camera, CameraError
from gesture_detector.config import (
    AppConfig, GESTURE_DISPLAY_NAMES, GESTURE_HAND_COUNTS, GESTURE_TO_KEY, PROJECT_ROOT,
)
from gesture_detector.event_gate import GestureEventGate
from gesture_detector.gesture_rules import GestureRuleRecognizer
from gesture_detector.hand_tracker import HandTracker
from gesture_detector.keyboard import KeyboardController
from gesture_detector.sequence import SEQUENCE, GestureSequence
from gesture_detector.visualizer import draw_hand_landmarks


LOGGER = logging.getLogger(__name__)
PREVIEW_SIZE = (640, 480)
FRAME_INTERVAL_MS = 15
SEAL_REFERENCE = PROJECT_ROOT / "assets" / "seal-reference.png"
SEAL_PANELS = {
    "bode": (285, 0, 407, 136),
    "javali": (285, 323, 407, 460),
    "cachorro": (142, 323, 265, 460),
    "cavalo": (285, 160, 407, 297),
    "passaro": (0, 484, 122, 621),
}
GESTURE_GUIDES = {
    "bode": "Levanta el índice de una mano y rodea su base con la otra.",
    "palma_abierta": "Muestra una sola mano, izquierda o derecha, con la palma hacia la cámara y los cinco dedos abiertos.",
    "javali": "Pon las muñecas arriba y apunta ambas manos hacia abajo con los dedos doblados.",
    "cachorro": "Coloca una mano horizontal con los dedos extendidos encima de la otra, cerrada.",
    "cavalo": "Junta los índices en un vértice arriba y dobla los dedos medios.",
    "passaro": "Acerca los índices y medios extendidos de ambas manos formando un vértice arriba.",
}


class GestureTestApp:
    """Show the webcam and insert recognized keys into a test field."""

    def __init__(self, root: tk.Tk, config: AppConfig | None = None) -> None:
        self.root = root
        self.config = config or AppConfig()
        self.recognizer = GestureRuleRecognizer()
        self.gate = self._new_gate()
        self.sequence = GestureSequence()
        self.sequence_keyboard = KeyboardController(enabled=True)
        self.camera: Camera | None = None
        self.tracker: HandTracker | None = None
        self.running = False
        self.frame_job: str | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.seal_images = self._load_seal_images()
        self.last_action = "Esperando gesto"

        self.root.title(f"Detector de gestos · v{__version__}")
        self.root.geometry("1060x660")
        self.root.minsize(1000, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        ttk.Label(
            container,
            text="Detector de sellos de manos",
            font=("Segoe UI", 17, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.preview = ttk.Label(
            container,
            text="Pulsa «Iniciar cámara» para comenzar la prueba.",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
        )
        self.preview.grid(row=1, column=0, sticky="nsew", padx=(0, 14))

        sequence_panel = ttk.LabelFrame(
            container, text=f"Secuencia Q → W → E · {self.sequence.timeout_seconds:g} s desde Q",
            padding=8,
        )
        sequence_panel.grid(row=2, column=0, sticky="ew", padx=(0, 14), pady=(8, 0))
        sequence_panel.columnconfigure(0, weight=1)
        self.sequence_status = tk.StringVar(value="0/3 · Haz Q para empezar")
        self.sequence_count = tk.IntVar(value=0)
        ttk.Label(sequence_panel, textvariable=self.sequence_status).grid(row=0, column=0, sticky="w")
        ttk.Label(sequence_panel, text="Completadas").grid(row=0, column=1, padx=(12, 5))
        ttk.Label(sequence_panel, textvariable=self.sequence_count).grid(row=0, column=2)
        self.sequence_bar = ttk.Progressbar(sequence_panel, maximum=len(SEQUENCE))
        self.sequence_bar.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        ttk.Label(sequence_panel, text="Al completar, envía Espacio a la aplicación con foco.").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(5, 0),
        )

        controls = ttk.Frame(container)
        controls.grid(row=1, column=1, rowspan=2, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Estado", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.status_var = tk.StringVar(value="Cámara detenida")
        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 12))

        button_row = ttk.Frame(controls)
        button_row.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        self.start_button = ttk.Button(
            button_row, text="Iniciar cámara", command=self.start
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = ttk.Button(
            button_row, text="Detener", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)

        guide = ttk.LabelFrame(controls, text="Gestos para probar", padding=7)
        guide.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        guide.columnconfigure(0, weight=1)
        self.gesture_list = ttk.Treeview(
            guide, columns=("key", "gesture"), show="headings", height=6,
            selectmode="browse",
        )
        self.gesture_list.heading("key", text="Tecla")
        self.gesture_list.heading("gesture", text="Sello")
        self.gesture_list.column("key", width=44, stretch=False, anchor=tk.CENTER)
        self.gesture_list.column("gesture", width=103)
        for label, key in GESTURE_TO_KEY.items():
            self.gesture_list.insert("", tk.END, iid=label, values=(key.upper(), GESTURE_DISPLAY_NAMES[label]))
        self.gesture_list.grid(row=0, column=0, sticky="new", padx=(0, 8))
        reference_panel = ttk.Frame(guide, width=166, height=184)
        reference_panel.grid(row=0, column=1, sticky="n")
        reference_panel.pack_propagate(False)
        self.seal_image = ttk.Label(
            reference_panel, text="Selecciona un sello", anchor=tk.CENTER, wraplength=145,
        )
        self.seal_image.pack(fill=tk.BOTH, expand=True)
        self.gesture_list.bind("<<TreeviewSelect>>", self._show_gesture_guide)
        self.guide_var = tk.StringVar(value="Elige un gesto para ver cómo colocar las manos.")
        ttk.Label(guide, textvariable=self.guide_var, wraplength=310, justify=tk.LEFT).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )

        text_header = ttk.Frame(controls)
        text_header.grid(row=4, column=0, sticky="new")
        text_header.columnconfigure(0, weight=1)
        ttk.Label(
            text_header, text="Campo de prueba", font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(text_header, text="Limpiar", command=self.clear_text).grid(
            row=0, column=1, sticky="e"
        )

        self.text_field = tk.Text(
            controls,
            height=4,
            width=25,
            wrap=tk.WORD,
            font=("Consolas", 14),
            undo=True,
        )
        self.text_field.grid(row=5, column=0, sticky="nsew", pady=(7, 12))
        controls.rowconfigure(5, weight=1)

        ttk.Label(
            controls,
            text=(
                "Haz un sello y mantenlo estable. Su letra se agrega aquí una vez; "
                "al soltarlo podrás activarlo de nuevo."
            ),
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=6, column=0, sticky="ew")

    def _load_seal_images(self) -> dict[str, ImageTk.PhotoImage]:
        images = {}
        references = (
            (SEAL_REFERENCE, SEAL_PANELS),
            (PROJECT_ROOT / "assets" / "gesture-keymap.png", {"palma_abierta": (607, 9, 939, 372)}),
        )
        for path, panels in references:
            try:
                with Image.open(path) as sheet:
                    for label, bounds in panels.items():
                        panel = sheet.crop(bounds)
                        panel = panel.resize((160, 180), Image.Resampling.LANCZOS)
                        images[label] = ImageTk.PhotoImage(panel, master=self.root)
            except OSError:
                LOGGER.warning("No se pudo cargar la referencia de gestos: %s", path)
        return images

    def _show_gesture_guide(self, _event) -> None:
        selection = self.gesture_list.selection()
        if selection:
            label = selection[0]
            self.guide_var.set(GESTURE_GUIDES[label])
            image = self.seal_images.get(label)
            self.seal_image.configure(
                image=image if image is not None else "",
                text="" if image is not None else "Imagen no disponible",
            )

    def _new_gate(self) -> GestureEventGate:
        return GestureEventGate(
            stable_frames=self.config.gesture_stable_frames,
            release_frames=self.config.gesture_release_frames,
        )

    def start(self) -> None:
        if self.running:
            return
        try:
            self.tracker = HandTracker(self.config)
            self.camera = Camera(self.config.camera_index)
            self.camera.__enter__()
        except (CameraError, FileNotFoundError, RuntimeError, OSError) as error:
            self._release_resources()
            messagebox.showerror("No se pudo iniciar", str(error), parent=self.root)
            self.status_var.set("No se pudo iniciar la cámara o MediaPipe.")
            return

        self.gate = self._new_gate()
        self.last_action = "Esperando gesto"
        self.running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Cámara activa · modo de prueba de texto")
        self.text_field.focus_set()
        self._update_frame()

    def _update_frame(self) -> None:
        self.frame_job = None
        if not self.running or self.camera is None or self.tracker is None:
            return

        try:
            frame = self.camera.read()
            captured_at = time.monotonic_ns()
            result = self.tracker.detect(frame, captured_at // 1_000_000)
            label = self.recognizer.classify(result, frame.shape)
            event = self.gate.update(label)
            key = GESTURE_TO_KEY.get(event)

            if event is not None and key is not None:
                self.text_field.insert(tk.INSERT, key)
                self.text_field.see(tk.INSERT)
                display_name = GESTURE_DISPLAY_NAMES[event]
                self.last_action = f"{display_name} → {key.upper()} agregado"
                selection = self.gesture_list.selection()
                if selection and selection[0] != event:
                    expected_key = GESTURE_TO_KEY[selection[0]].upper()
                    self.last_action += f" · esperabas {expected_key}"

            self._update_sequence(key, captured_at / 1_000_000_000)

            hands_detected = len(result.hand_landmarks)
            selection = self.gesture_list.selection()
            expected_hands = GESTURE_HAND_COUNTS[selection[0]] if selection else None
            if label is None and expected_hands is not None and hands_detected != expected_hands:
                prediction = f"Manos detectadas: {hands_detected}/{expected_hands}"
            elif hands_detected == 0:
                prediction = "Sin manos detectadas"
            elif label is None:
                prediction = "Sin regla coincidente"
            else:
                prediction = (
                    f"{GESTURE_DISPLAY_NAMES[label]} → "
                    f"{GESTURE_TO_KEY[label].upper()}"
                )
            mode = "Prueba local: escribe en el campo de esta ventana"
            gate_state = "Listo" if self.gate.armed else "Gesto activo"
            status_lines = (
                mode,
                f"Deteccion actual: {prediction}",
                f"Estado: {gate_state}",
                f"Última tecla: {self.last_action}",
            )
            annotated = draw_hand_landmarks(frame, result, status_lines)
            self._show_frame(annotated)
            self.status_var.set(
                f"Cámara activa · {prediction}\nÚltima tecla: {self.last_action}"
            )
        except (CameraError, RuntimeError, OSError, ValueError) as error:
            LOGGER.exception("Falló la prueba de gestos")
            self.stop()
            messagebox.showerror("Error durante la prueba", str(error), parent=self.root)
            return

        self.frame_job = self.root.after(FRAME_INTERVAL_MS, self._update_frame)

    def _update_sequence(self, key: str | None, now: float) -> None:
        previous = self.sequence.progress
        if self.sequence.update(key, now):
            self.sequence_count.set(self.sequence_count.get() + 1)
            self.sequence_bar.configure(value=len(SEQUENCE))
            self.sequence_keyboard.press_key("space")
            self.sequence_status.set("3/3 · Espacio enviado")
        elif self.sequence.progress != previous:
            self._show_sequence_progress()

    def _show_sequence_progress(self) -> None:
        matched = len(self.sequence.progress)
        self.sequence_bar.configure(value=matched)
        if matched:
            next_key = SEQUENCE[matched].upper()
            self.sequence_status.set(f"{matched}/{len(SEQUENCE)} · Ahora haz {next_key}")
        else:
            self.sequence_status.set("0/3 · Haz Q para empezar")

    def _show_frame(self, frame) -> None:
        width, height = PREVIEW_SIZE
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self.photo, text="")

    def clear_text(self) -> None:
        self.text_field.delete("1.0", tk.END)
        self.text_field.focus_set()

    def stop(self) -> None:
        self.running = False
        if self.frame_job is not None:
            self.root.after_cancel(self.frame_job)
            self.frame_job = None
        self.sequence.reset()
        self._show_sequence_progress()
        self._release_resources()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Cámara detenida")
        self.preview.configure(image="", text="Cámara detenida")
        self.photo = None

    def _release_resources(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def run_gesture_gui(config: AppConfig | None = None) -> int:
    """Start the Tkinter test interface."""
    root = tk.Tk()
    GestureTestApp(root, config)
    root.mainloop()
    return 0
