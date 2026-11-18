#!/usr/bin/env python3
"""Run the exact platform B model on one user-supplied image."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmodel", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    software = Path(__file__).resolve().parents[1]
    assets = software.parent / "reproducibility" / "release_assets.json"
    model = json.loads(assets.read_text())["platform_b_xmodel"]
    try:
        digest = hashlib.sha256(args.xmodel.read_bytes()).hexdigest()
    except OSError as exc:
        parser.error(str(exc))
    if digest != model["sha256"]:
        parser.error("XMODEL SHA-256 does not match the platform B campaign.")
    if not args.image.is_file():
        parser.error("Provide an existing image with --image.")

    import cv2
    import numpy as np
    import vitis_ai_library
    import xir
    from postprocess import decode, nms, preprocess_image

    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("OpenCV could not decode the image.")
    height, width = image.shape[:2]
    gain = 1024 / max(height, width)
    resized_h, resized_w = int(height * gain), int(width * gain)
    if min(resized_h, resized_w) < 1:
        parser.error("Image aspect ratio is too extreme for 1024-pixel preprocessing.")

    graph = xir.Graph.deserialize(str(args.xmodel))
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    fingerprints = [
        sg.get_attr("dpu_fingerprint")
        for sg in subgraphs if sg.get_attr("device") == "DPU"
    ]
    if fingerprints != [int(model["fingerprint"], 16)]:
        parser.error("Expected one DPU subgraph with the platform B fingerprint.")

    runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
    inputs, outputs = runner.get_inputs(), runner.get_outputs()
    destination = np.asarray(inputs[0])
    if destination.shape != (1, 1024, 1024, 3) or destination.dtype != np.int8:
        raise RuntimeError("Unexpected XMODEL input layout or dtype.")
    scale = 2.0 ** inputs[0].get_tensor().get_attr("fix_point")
    image_rgb = preprocess_image(image).transpose(0, 2, 3, 1)
    destination[:] = np.clip(np.floor(image_rgb * scale + 0.5), -128, 127).astype(np.int8)
    if runner.wait(runner.execute_async(inputs, outputs)) != 0:
        raise RuntimeError("Graph Runner execution failed.")

    values = sorted((np.asarray(out) for out in outputs), key=lambda value: -value.shape[1])
    raw = [value.transpose(0, 3, 1, 2) for value in values]
    if [value.shape for value in raw] != [(1, 18, size, size) for size in (128, 64, 32)]:
        raise RuntimeError("Unexpected YOLOv7-tiny output shapes.")
    head = json.loads((software / "artifacts" / "y7_decoder.json").read_text())
    anchors = np.array(head["anchors"], dtype=np.float32)
    strides = np.array(head["strides"], dtype=np.float32)
    detections = nms(decode(raw, anchors, strides), 0.25, 0.6)
    offset_x = int(round((1024 - resized_w) / 2 - 0.1))
    offset_y = int(round((1024 - resized_h) / 2 - 0.1))
    detections[:, [0, 2]] = np.clip(
        (detections[:, [0, 2]] - offset_x) / (resized_w / width), 0, width
    )
    detections[:, [1, 3]] = np.clip(
        (detections[:, [1, 3]] - offset_y) / (resized_h / height), 0, height
    )
    print(json.dumps({
        "columns": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
        "class_names": ["fish"],
        "detections": detections.tolist(),
    }, allow_nan=False))


if __name__ == "__main__":
    main()
