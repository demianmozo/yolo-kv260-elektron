# Detectores YOLO INT8 en Kria KV260

Materiales de revisión del artículo de Elektron sobre partición CPU/DPU, etiqueta `elektron-revision-2026-11`. El manuscrito de septiembre es una copia histórica anterior a la revisión de noviembre de 2026. Las tablas y reportes conservan los resultados de las campañas originales. [Manuscrito histórico](manuscript/main.tex).

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

Requiere una Kria KV260 con el overlay B ya cargado y el entorno de la placa con XIR, VART/Graph Runner 3.0, NumPy y OpenCV. Confirmá con `xdputil query` que la placa informa el fingerprint B indicado arriba. Desde el árbol de código extraído, indicá la ubicación del XMODEL descargado por separado y de tu imagen.

```bash
python3 software/benchmark/infer_y7.py \
  --xmodel ./yolov7tiny_custom.xmodel --image ./mi_ecograma.png
```

El comando verifica el SHA-256 y el fingerprint del grafo, procesa una imagen con el preprocesamiento de 1024 × 1024 y devuelve detecciones JSON con coordenadas en píxeles de la imagen original. Usá una imagen legible por OpenCV. Para interpretar detecciones, la entrada debe tener el recorte y la representación acústica compatibles con el detector. Una imagen sintética solo permite comprobar la ejecución; la paridad y las métricas del artículo requieren las entradas originales. La corrida de una imagen tampoco reproduce el benchmark de latencia o potencia.

La release incluye dos adjuntos separados del ZIP de código: `yolov7tiny_custom.xmodel` (INT8, plataforma B) y `best.pt` (checkpoint FP32 de referencia; usar su componente EMA con el código YOLOv7 fijado). `SHA256SUMS.release` verifica los adjuntos y el ZIP. El overlay, bitstream y firmware se obtienen por separado. [Hashes y procedencia](reproducibility/release_assets.json), [protocolo de la campaña](software/protocol.json).

## Qué requiere acceso a los datos originales

La release no incluye las imágenes ni las etiquetas necesarias para recalcular las métricas de validación. Los 18 pares P66–P83 coinciden con el manifiesto de validación interna y están separados del entrenamiento y la calibración de YOLOv7-tiny. Sus imágenes coinciden por SHA-256 con entradas de la calibración histórica de YOLOv8n, por lo que se excluyen bajo la restricción de datos de calibración. Tampoco se incluyen datos de entrenamiento ni tensores de entrada ARM o referencias INT8 para repetir la paridad.

Las métricas de YOLOv7-tiny del artículo se obtuvieron con el checkpoint FP32 y la simulación INT8 sobre validación interna. Para recalcularlas se requieren los datos originales y el procedimiento de evaluación correspondiente. El ejecutor de B procesa una imagen y devuelve detecciones. La precisión histórica de YOLOv8n en A corresponde a otro conjunto de evaluación. [Protocolo experimental](documentation/experimental_protocol.md) y [limitaciones](documentation/limitations.md).

Los scripts históricos y las configuraciones de hardware documentan cómo se obtuvo la evidencia. Conservan dependencias y montajes de sus entornos originales. Los parches de YOLOv7 se refieren al upstream fijado y se acompañan del texto GPLv3. Los avisos de los reportes y de la clase LaTeX se mantienen. [Licencia de YOLOv7](licenses/YOLOv7-GPL-3.0.txt) y [referencias de hardware](hardware/repositories.lock.json).
