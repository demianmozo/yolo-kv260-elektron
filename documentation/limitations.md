# Limitaciones y pendientes reales

## Científicas

- 59 imágenes de entrenamiento y 18 de validación interna; selección de checkpoint sobre esa validación, sin test ni campaña independientes.
- 84 archivos auditados, frente a 85 mencionados anteriormente; P39 con dos archivos excluidos. No inventar conciliación ni procedencia acústica faltante.
- YOLOv8n y YOLOv7-tiny no tienen métricas de precisión comparables por split. No se aisló SiLU mediante ablación; la comparación incluye otras diferencias arquitectónicas.
- Paridad exacta únicamente para tres entradas ARM y nueve tensores raw. No equivalencia universal ni evaluación nueva de precisión de 18 imágenes en B.
- Cien iteraciones sobre tres imágenes por pasada; no son sesiones independientes. Cercanía A/B descriptiva, sin prueba de equivalencia estadística ni mejoras demostradas.
- Pipeline secuencial con imágenes en RAM. No incluye adquisición/PNG ni mide throughput concurrente; no prueba tiempo real.
- Bytes lógicos de fronteras, no tráfico físico DRAM.
- INA260 mide VCC_SOM; energía aproximada por trapecios y reposo retenido. No potencia de carrier completo/enchufe ni energía aislada DPU.
- Vivado vectorless de baja confianza, alcance on-chip diferente; no validar esa estimación contra INA260 como si fueran la misma magnitud.
- WNS +0,006 ns; I2C con delays ausentes y advertencias de metodología. No Fmax experimental ni cierre incondicional.

## Evidencia y documentación

- El manifiesto original de campaña incluye una entrada de su propio archivo que no coincide. Los otros 18 hashes coinciden. Se conserva el original; el manifiesto del paquete excluye su propio SHA256SUMS.
- La traza XAT confirma fingerprint B, IP y VART; no se localizó una captura textual independiente B de `xmutil listapps`/`xclbinid`/smoke test. El UUID está en metadatos del artefacto y protocolo. No se solicita volver a medir.
- El protocolo copiado nombra `results/platform.json` y comandos de reproducción que no están en esa campaña final. Se conserva como original; usar las rutas del protocolo consolidado. No presentar esos archivos como disponibles.
- El DTBO final deriva del histórico funcional y difiere de la preparación anterior desde fuentes oficiales. Solo el contenido de `deployment/final_v41` corresponde al paquete final.
- `results/hardware/validation_summary.json`, inventarios A, auditorías y planes anteriores son capturas históricas: sus estados pendientes no son el estado actual.

## Antes del envío

- Completar autoría, afiliaciones, ORCID, roles CRediT, financiación y conflictos con los autores; no inventarlos.
- Confirmar la referencia definitiva de CASE, explicitar extensión y conciliar su inventario cuando haya evidencia.
- Completar procedencia/permisos de datos y ubicación pública de reproducibilidad. Por ahora el paquete es local.
- Compilar `manuscript/main.tex` en Overleaf y comprobar ≤12 páginas incluyendo figuras, tablas y referencias. La compilación manual queda a cargo del usuario; no se certificó paginación aquí.
- Preparar carta editorial y revisores propuestos según requisitos de Elektron; no se inventan declaraciones ni nombres.
