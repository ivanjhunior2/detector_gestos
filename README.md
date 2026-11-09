# Detector de gestos para teclado

**Versión 2.0.0 · Windows · Python 3.10**

Convierte seis gestos de manos captados por la webcam en pulsaciones de teclado. La interfaz muestra la cámara en espejo, los puntos de las manos y una guía ilustrada para practicar cada gesto. Las letras detectadas aparecen en un campo de prueba.

La versión 2.0.0 añade la secuencia **Q → W → E**, que envía una pulsación de **Espacio** a la aplicación activa.

## Gestos y teclas

| Tecla | Gesto | Cómo hacerlo |
| --- | --- | --- |
| Q | Bode | Eleva el índice de una mano y rodea su base con la otra. |
| W | Cachorro | Coloca una mano horizontal con los dedos extendidos sobre la otra, cerrada. |
| E | Javali | Coloca las muñecas arriba y orienta ambas manos hacia abajo con los dedos doblados. |
| R | Palma abierta | Muestra los cinco dedos abiertos de una sola mano, izquierda o derecha, con la palma hacia la cámara. |
| D | Cavalo | Junta los índices en un vértice arriba y dobla los dedos medios. |
| F | Pássaro | Acerca los índices y medios extendidos de ambas manos formando un vértice arriba. |

Los cinco sellos requieren dos manos. La palma abierta requiere una sola.

## Instalación en Windows

Necesitas Python 3.10 y una webcam. Desde PowerShell, en la carpeta del proyecto, crea el entorno virtual e instala las dependencias.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si ya tienes el entorno del proyecto, basta con ejecutar el segundo comando. Descarga el [modelo oficial Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) y guárdalo en `models/hand_landmarker.task`. El archivo se descarga por separado del código. [Documentación de MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## Cómo ejecutar el programa

**Interfaz de práctica**

```powershell
.\venv\Scripts\python.exe run_gesture_gui.py
```

Pulsa **Iniciar cámara** y elige una opción en **Gestos para probar**. Verás el dibujo del gesto y sus instrucciones. Forma el gesto y comprueba la letra en **Campo de prueba**.

**Pulsaciones individuales en otra aplicación**

```powershell
.\venv\Scripts\python.exe run_gesture_typing.py --send-keys
```

Cierra primero la interfaz de práctica para liberar la cámara. Después de iniciar el modo de teclado, enfoca la aplicación que recibirá las pulsaciones, como el Bloc de notas. Puedes omitir `--send-keys` para simular las pulsaciones en la terminal.

## Cómo probar la secuencia Q → W → E

1. Abre `run_gesture_gui.py` y pulsa **Iniciar cámara**. El panel debajo de la cámara muestra el progreso y el contador **Completadas**.
2. Enfoca el Bloc de notas y escribe `A` para tener una referencia visible.
3. Haz **Bode (Q) → Cachorro (W) → Javali (E)**. Completa la secuencia en cuatro segundos desde que se confirma Q. El panel debe indicar **Espacio enviado** y el contador debe subir en uno.
4. Mantén el último gesto, **Javali (E)**, durante unos segundos. Luego baja las manos y escribe `B`. El resultado debe ser `A B`, con exactamente un espacio, y el contador debe conservar el mismo valor.
5. Si quieres comprobar el reinicio, vuelve a hacer la secuencia completa. Un gesto confirmado fuera de orden cancela el intento. Una nueva Q inicia otra cuenta de cuatro segundos.

## Cómo funciona

OpenCV captura la imagen y MediaPipe estima 21 puntos por mano. El clasificador compara la geometría de la postura con las reglas de los seis gestos. La aplicación confirma el candidato después de cuatro fotogramas consecutivos y genera una sola pulsación mientras se mantiene el gesto. PyAutoGUI envía las teclas en el modo de teclado y Tkinter muestra la interfaz de práctica.

## Validación

El autor validó con su webcam en Windows el reconocimiento de los gestos individuales en la versión 1.0 y la pulsación de Espacio al completar la secuencia en la versión 2.0. Las pruebas automatizadas cubren los seis gestos y sus variantes, el orden de la secuencia, el límite de tiempo y el evento único al sostener el último gesto.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para repetir la evaluación con una webcam, ejecuta `evaluate_gestures.py` con el mismo intérprete. El asistente guía los intentos en ambas orientaciones y genera un informe local con los puntos de las manos y las predicciones.
