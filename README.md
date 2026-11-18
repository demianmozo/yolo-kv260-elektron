# Detectores YOLO INT8 en Kria KV260

Materiales de revisión del artículo de Elektron sobre partición CPU/DPU. [El manuscrito de septiembre](manuscript/main.tex) es una copia histórica anterior a la revisión de noviembre de 2026. Las tablas y reportes conservan los resultados de las campañas originales.

| Plataforma | Fingerprint | Alcance |
|---|---|---|
| B | `0x101000016010407` | YOLOv7-tiny en el overlay `kv260-elektron-b4096-v41` del artículo. |
| A | `0x101000056010407` | Comparación histórica con YOLOv8n y campaña previa de YOLOv7-tiny. |

## Verificar la evidencia sin placa ni datos privados

```bash
python3 reproducibility/verify.py
```

El verificador comprueba los hashes del paquete, recalcula las estadísticas de latencia de las 100 repeticiones normales y las 100 perfiladas de B a partir de sus CSV y verifica la energía total aproximada de VCC_SOM a partir de la telemetría. Los reportes permiten inspeccionar partición y utilización del diseño. Los resúmenes de precisión y paridad documentan los resultados registrados. [Tablas](tables/), [evidencia B](evidence/platform_B/), [evidencia A](evidence/historical_A/) y [reportes de hardware](evidence/hardware/reports/).

## Ejecutar YOLOv7-tiny con una imagen propia

Requiere una Kria KV260 con el overlay B ya cargado, el entorno de la placa con XIR, VART/Graph Runner 3.0, NumPy y OpenCV, y una copia autorizada del XMODEL exacto. Confirmá con `xdputil query` que la placa informa el fingerprint B indicado arriba.

```bash
python3 software/benchmark/infer_y7.py \
  --xmodel ./yolov7tiny_custom.xmodel --image ./mi_ecograma.png
```

El comando verifica el SHA-256 y el fingerprint del grafo, procesa una imagen con el preprocesamiento de 1024 × 1024 y devuelve detecciones JSON con coordenadas en píxeles de la imagen original. Usá una imagen legible por OpenCV. Para interpretar detecciones, la entrada debe tener el recorte y la representación acústica compatibles con el detector. Una imagen sintética solo permite comprobar la ejecución; la paridad y las métricas del artículo requieren las entradas originales. La corrida de una imagen tampoco reproduce el benchmark de latencia o potencia.

El XMODEL de B y el checkpoint FP32 de referencia están identificados por hash, con redistribución pendiente de confirmación. El overlay, bitstream y firmware se obtienen por separado. [Artefactos y procedencia](reproducibility/release_assets.json), [protocolo de la campaña](software/protocol.json).

## Qué requiere acceso a los datos originales

Repetir entrenamiento, calibración o métricas de detección requiere los datos, etiquetas y splits originales. Repetir la prueba de paridad requiere los tres ecogramas, los tensores de entrada ARM y las referencias INT8 originales. Estos materiales quedan fuera del paquete, junto con los datos de entrenamiento. La precisión histórica de YOLOv8n en A y la validación interna de YOLOv7-tiny se obtuvieron sobre conjuntos distintos. [Protocolo experimental](documentation/experimental_protocol.md) y [limitaciones](documentation/limitations.md).

Los scripts históricos y las configuraciones de hardware documentan cómo se obtuvo la evidencia. Conservan dependencias y montajes de sus entornos originales. Los parches de YOLOv7 se refieren al upstream fijado y se acompañan del texto GPLv3. Los avisos de los reportes y de la clase LaTeX se mantienen. [Licencia de YOLOv7](licenses/YOLOv7-GPL-3.0.txt) y [referencias de hardware](hardware/repositories.lock.json).
