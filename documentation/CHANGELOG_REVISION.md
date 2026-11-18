# Revisión editorial y científica — 20/09/2026

## Versión integrada y preservación

Se integró el manuscrito de septiembre en `manuscript/main.tex`. El PDF de siete páginas correspondía a esa versión del manuscrito. El TEX incluido se conserva como copia histórica anterior a la revisión de noviembre de 2026.

Autoría, orden, afiliaciones y correos permanecen **byte por byte** como en el TEX recibido. SHA-256 del bloque comprendido entre `\author{` y `\maketitle`: `0ef0b97cc7cfbb380db80df1296de985a327d7751033cfdbda7fd19c251198a7`.

## Cambios aplicados

- Resumen, abstract y discusión: formulación explícita de que los MACs convolucionales solos no predijeron la latencia extremo a extremo. Se conservan las salvedades sobre ausencia de ablación causal de SiLU y diferencias de arquitectura.
- Resumen y abstract: 2,634 J/imagen se identifica como energía total aproximada sobre VCC_SOM, excluyendo carrier completo y alimentación de red. Vivado vectorless sigue separado de INA260.
- Semántica de clase: `case_yolov7tiny/data/fish.yaml` declara `fish`; CASE, sección II-A, páginas 1–2, establece su asociación con ecos de merluza. Se explicita esa asociación documental con cita a CASE, sin inventar corroboración biológica o capacidad multiespecie. Fuente local: `sources/legacy_documentary/references/CASE2026_Mozo_et_al.pdf`; complemento: `sources/legacy_documentary/evidence/dataset_split_summary.json`.
- Antecedentes y bibliografía: 15 referencias verificadas y citadas, ordenadas según primera aparición. Se añadieron el repositorio YOLOv7, Ultralytics 8.4.46, cuantización entera, LPYOLO, Hybrid YOLO FPGA, Embedded YOLO, clasificación acústica multifrecuencia, KV260, DPUCZDX8G, VART y XRT. No se trasladan resultados de esos trabajos al experimento local ni se comparan cifras incompatibles.
- Embedded YOLO incorpora la autoría corregida por su corrigendum (incluye Jiali Liu). No se atribuye a ese trabajo una implementación DPUCZDX8G.
- Entorno: KV260/K26; inventario previo PetaLinux 2022.2_release_S10071807, kernel 5.15.36-xilinx-v2022.2 y XRT/ZOCL 2.14.0; traza B con VART 3.0.0, XIR 9204ac72 y DPU 300 MHz; benchmark B con OpenCV 4.5.2 y NumPy 1.21.2. Se diferencia inventario previo de consulta retenida en B. ARM: protocolo 1333333 kHz y traza 1333 MHz. Fuentes: `logs/board_current_inventory_confirmed.json`, `results/new_platform/runtime_from_trace.json`, `results/new_platform/raw/y7_{normal,profiled}.json`, `results/new_platform/protocol.json` y `SOURCE_ROOT/experiments/kv260_20260916/results/platform.json`.
- Utilización: se añadieron en texto los porcentajes del diseño completo directamente de `results/hardware/reports/utilization.rpt`: LUT 59,43%; FF 55,40%; BRAM 65,63%; URAM 87,50%; DSP 56,89%. La tabla absoluta se conserva. No se infieren capacidades ni se suman LUTRAM/SRL al total.
- P, R y mAP se muestran a tres decimales; cambios de mAP también redondeados para presentación. Los valores completos de evidencia, tiempos, timing y comparaciones A/B permanecen intactos.
- CASE: se verificó sede/fecha en el sitio oficial, pero no se encontró un registro específico con actas, DOI o ISBN del trabajo. Se mantiene como trabajo presentado.

## Fuentes bibliográficas contrastadas

Consulta del 20/09/2026; las referencias completas están en el TEX, sin un archivo bibliográfico duplicado.

- YOLOv7: [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html) y [repositorio oficial](https://github.com/WongKinYiu/yolov7); commit retenido en `y7_source.json` de la campaña histórica.
- YOLOv8: [Ultralytics v8.4.46](https://github.com/ultralytics/ultralytics/tree/v8.4.46), contrastado con `sources/legacy_documentary/reproducibility/y8_requirements.txt`.
- Cuantización: [Jacob et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html).
- [LPYOLO, original arXiv](https://arxiv.org/abs/2207.10482).
- [Hybrid YOLO FPGA, página editorial](https://irjaeh.com/index.php/journal/article/view/530).
- [Embedded YOLO](https://onlinelibrary.wiley.com/doi/10.1155/2021/6555513) y [corrigendum](https://onlinelibrary.wiley.com/doi/10.1155/2021/9818014).
- [Clasificación acústica, editorial ICES](https://academic.oup.com/icesjms/article/77/4/1391/5712978), contrastada con [registro del instituto autor](https://nr.no/publikasjon/1800050/) y [lista de autores en IMR](https://www.hi.no/en/hi/about-us/employees/espen-johnsen).
- [DPU PG338 v4.1](https://docs.amd.com/r/en-US/pg338-dpu/Features), [KV260 DS986](https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Ordering-Information), [VART 3.0](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Programming-with-VART), [XRT 2022.2](https://docs.amd.com/r/2022.2-English/ug1451-xrt-release-notes/About-Xilinx-Runtime-XRT), documentación y repositorio oficial ya retenidos para Vitis AI/plataforma.
- [CASE 2026](https://case.ar/) y [sección Publicaciones](https://case.ar/?page_id=986): no se confirmó publicación específica del artículo.

## Verificación y maquetación

Se verificaron correspondencia uno a uno entre citas y bibliografía (15/15), orden de referencias, entradas de tablas/figuras, llaves y entornos LaTeX. Se ejecutó `python3 reproducibility/verify_package.py` y la comparación SHA-256 de 117 archivos de resultados, tablas, figuras y despliegue: todos permanecen iguales al manifiesto anterior. Se preservó la clase original y el bloque de autoría recibido.

Se renderizaron y revisaron las siete páginas del **PDF recibido**, sin identificar cortes o solapamientos que justificaran cambiar márgenes, figuras o tablas. No se garantiza conservar sus siete páginas después de ampliar el texto; no se redujo contenido para ese fin.

Por indicación posterior del usuario, la compilación final se realizará en Overleaf. El intento preparatorio con Tectonic portátil se interrumpió antes de generar un PDF; no se instaló ningún paquete del sistema. No hay un PDF revisado ni un conteo final de páginas o warnings de compilación que pueda certificarse aquí. Se mantiene `thebibliography`, por lo que no hace falta BibTeX/Biber: basta la compilación normal de Overleaf y sus pasadas para referencias cruzadas.

Exportación actualizada: `elektron_overleaf.zip`. El manifiesto raíz se actualiza únicamente para reflejar archivos documentales nuevos/cambiados; los hashes experimentales permanecen idénticos y el manifiesto anterior se conserva en el archivo histórico.

## Pendientes que requieren intervención humana

1. Compilar en Overleaf y comprobar paginación final ≤12 páginas, warnings y distribución de floats tras la ampliación.
2. Completar los campos declarativos aún marcados: ORCID/CRediT, financiación, agradecimientos y conflictos de interés, sin cambiar la autoría ya fijada.
3. Confirmar permisos y ubicación pública de materiales; completar procedencia acústica o conciliación 84/85 solo si se dispone de documentación adicional. Las limitaciones actuales permanecen explícitas.
4. Aportar referencia definitiva del trabajo CASE si se recibe un registro de publicación; mientras tanto la cita como trabajo presentado es válida y honesta.
