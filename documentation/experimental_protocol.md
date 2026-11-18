# Protocolo experimental consolidado

Documento retrospectivo; no ejecutar campañas nuevas. Fuentes: `results/new_platform/protocol.json`, scripts retenidos y datos crudos en `results/new_platform/raw/`.

| Aspecto | Condición |
|---|---|
| Red B | YOLOv7-tiny INT8 recompilada para 0x101000016010407 |
| Resolución / batch | 1024×1024 / 1 |
| Entradas | P66, P74, P83; orden circular |
| Repeticiones | 10 calentamientos excluidos + 100 medidas por pasada |
| Imágenes | BGR uint8 precargadas; lectura/PNG excluidos |
| Preprocesado | AREA en reducción, dimensiones enteras, letterbox gris 114, RGB NCHW/255 |
| Cuantización entrada | NHWC INT8, fix_point=6; floor(x×64+0,5), saturación |
| Posprocesado | decodificador retenido; confianza 0,25, NMS IoU 0,6 |
| CPU | userspace 1333333 kHz según protocolo; un hilo OpenCV/BLAS |
| DPU | 300 MHz nominales; 600 MHz segundo clock en implementación |
| Ejecución | secuencial, sin solapamiento ni optimizaciones nuevas |
| Pasadas | normal con telemetría; perfilada independiente sin telemetría |
| Sensor | INA260 U14, VCC_SOM; no carrier completo |
| Potencia | objetivo 0,5 s; 30 lecturas antes, 92 activas, 30 después |

Latencias: media, mediana, desviación estándar muestral (ddof=1), p95 lineal de las cien muestras. En XAT se emparejan 440 intervalos y se excluyen 40 de calentamiento: quedan 100 grupos de 1 DPU + 3 CPU. El controlador DPU contiene el tiempo hardware; no sumarlos. Los porcentajes usan el denominador de la misma pasada perfilada.

Paridad: comparar NPZ hardware con NPZ `int8_board` usando exactamente los tensores ARM retenidos. No usar salidas `int8_val` del host para sustituirlas: el resize host/ARM difería en P83. Nueve tensores NCHW, tres escalas por imagen; comparación de elementos, máximo absoluto y RMSE. Las detecciones 17/27/25 corresponden al benchmark a 0,25, no al diagnóstico a 0,2147436.

Energía: integrar lecturas activas mediante trapecios, mantener primer/último valor hasta extremos del intervalo cronometrado, dividir por 100. Potencia media temporal = energía/tiempo. Reposo = media aritmética de las 60 lecturas anteriores/posteriores; energía incremental = (energía − reposo×tiempo)/100. La DE de potencia describe las 92 lecturas, no incertidumbre entre campañas.

No hay sesiones repetidas independientes, medición de frecuencia máxima, tráfico DRAM físico, captura acústica end-to-end ni test independiente. La temperatura ambiente no está registrada. XRT 2.14.0/SO 2022.2 constan en inventario previo y protocolo; la traza B confirma VART/XIR y DPU, pero no reemplaza una captura B completa de todos los paquetes.

Recalcular offline: `python3 reproducibility/consolidate.py` y `python3 reproducibility/figures_tables.py` desde este paquete. El primer script consulta originales locales para verificar/copiar; las copias y sus hashes permiten inspección sin ejecutar modelos. Los scripts de placa están conservados exclusivamente como evidencia, no como instrucciones actuales de ejecución.
