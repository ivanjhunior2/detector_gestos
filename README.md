# Detector de gestos para teclado

**Versión 1.0.0 · Windows · Python 3.10**

Convierte seis gestos de manos captados por la webcam en pulsaciones de teclado. La interfaz muestra la cámara en espejo, los puntos de las manos y una guía ilustrada para practicar cada gesto. Las letras detectadas aparecen en un campo de prueba.

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

**Pulsaciones en otra aplicación**

```powershell
.\venv\Scripts\python.exe run_gesture_typing.py --send-keys
```

Cierra primero la interfaz de práctica para liberar la cámara. Después de iniciar el modo de teclado, enfoca la aplicación que recibirá las pulsaciones, como el Bloc de notas. Puedes omitir `--send-keys` para simular las pulsaciones en la terminal.

## Cómo funciona

OpenCV captura la imagen y MediaPipe estima 21 puntos por mano. El clasificador compara la geometría de la postura con las reglas de los seis gestos. La aplicación confirma el candidato después de cuatro fotogramas consecutivos y genera una sola pulsación mientras se mantiene el gesto. PyAutoGUI envía las teclas en el modo de teclado y Tkinter muestra la interfaz de práctica.

## Validación de la versión 1.0

El funcionamiento de la versión 1.0 fue validado manualmente por el autor con su webcam en Windows. Las pruebas automatizadas usan datos simulados para cubrir los seis gestos, sus variantes en espejo, la palma abierta con cualquiera de las manos, la distinción entre posturas similares y la emisión de una sola pulsación al sostener un gesto.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para repetir la evaluación con una webcam, ejecuta `evaluate_gestures.py` con el mismo intérprete. El asistente guía los intentos en ambas orientaciones y genera un informe local con los puntos de las manos y las predicciones.

## Alcance

La versión 1.0 reconoce posturas individuales. El reconocimiento de secuencias de sellos queda para una versión posterior.
