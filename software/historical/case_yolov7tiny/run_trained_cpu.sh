#!/usr/bin/env bash
set -euo pipefail
case_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec docker run --rm --network none --cpus 4 --memory 8g --shm-size 512m --user "$(id -u):$(id -g)" \
 --cap-drop ALL --security-opt no-new-privileges -e HOME=/tmp -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -e PYTHONPATH=/workspace/deps:/workspace/upstream_leaky:/workspace -e STEP="${1:?calib,test,compile}" \
 -v /etc/passwd:/etc/passwd:ro -v "$case_dir:/workspace" -v "$case_dir/../dataset:/dataset:ro" \
 -w /workspace --entrypoint /bin/bash \
 xilinx/vitis-ai-pytorch-cpu@sha256:f55bd069ffd56c6358cae29df19e6085f2bcf8ea5e045744aa412fd72db521ed -c '
 set -eo pipefail
 ulimit -c 0
 source /opt/vitis_ai/conda/etc/profile.d/conda.sh
 conda activate vitis-ai-pytorch
 if [ "$STEP" = compile ]; then
  mkdir -p trained/compiled
  xcompiler -i trained/quantized/TinyRaw_int.xmodel -o trained/compiled/yolov7tiny.xmodel -f 0x101000056010407
  python audit_trained.py
 else python quantize_trained.py "$STEP"; fi
 '
