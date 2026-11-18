#!/usr/bin/env bash
set -euo pipefail
case_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec docker run --rm --network none --cpus 4 --memory 8g --shm-size 512m --user "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges -e HOME=/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -e PYTHONPATH=/workspace/deps:/workspace/upstream_leaky:/workspace -e STEP="${1:-prepare}" \
 -v /etc/passwd:/etc/passwd:ro -v "$case_dir:/workspace" \
 -v "$case_dir/../case_yolov8n/salidas/vai30_partition:/reference:ro" \
 -w /workspace --entrypoint /bin/bash \
 xilinx/vitis-ai-pytorch-cpu@sha256:f55bd069ffd56c6358cae29df19e6085f2bcf8ea5e045744aa412fd72db521ed -c '
 set -eo pipefail
 ulimit -c 0
 source /opt/vitis_ai/conda/etc/profile.d/conda.sh
 conda activate vitis-ai-pytorch
 if [ "$STEP" = prepare ]; then python prepare.py;
 elif [ "$STEP" = compile ]; then
  mkdir -p outputs/compiled
  xcompiler -i outputs/quantized/TinyRaw_int.xmodel -o outputs/compiled/yolov7tiny.xmodel -f 0x101000056010407
 else python inspect_quantize.py "$STEP"; fi
 '
