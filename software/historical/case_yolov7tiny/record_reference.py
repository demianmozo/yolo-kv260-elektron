"""Record the corrected FP32 reference configuration and code used to produce it."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent;run=B/'runs/reference'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B/'data/manifest.csv')=='971b6269a988f05c1bdc609c27c3da7a0b002930acb7680b32b9d2e6d1f8c48a'
assert sha(B/'pretrained/yolov7-tiny.pt')=='0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77'
assert sha(B.parent/'case_yolov8n/inspeccion/runs/detect/train-2/weights/best.pt')=='6ad6dabb90618f1ddec98066bccf27cf987144a0d32f8dedf0413d3e969a6cf2'
info=dict(status='corrected_FP32_reference',source_commit='eef4f2c92851e4d72c0630ee1f3615144353a73e',architecture='cfg/deploy/yolov7-tiny.yaml',classes=['fish'],resolution=[1024,1024],seed=42,batch=4,accumulate=1,precision='FP32, AMP/TF32 disabled',optimizer='SGD',loss_scaling='unchanged upstream per-batch loss multiplied by actual batch size; one step per batch',effective_weight_decay=.0005,early_stop_patience=40,selection='val_mAP50_95',split_sha256=sha(B/'data/manifest.csv'),pretrained_sha256=sha(B/'pretrained/yolov7-tiny.pt'),verification=json.loads((run/'verification.json').read_text()),environment_lock_sha256=sha(B/'requirements-train.lock.txt'),defective_archive=str(B/'archive/defective'),quantization_performed=False,compilation_performed=False)
info['code_sha256']={str(p.relative_to(B)):sha(p) for p in [B/'train_entry.py',B/'training_checks.py',B/'fix_training.py',B/'training_src/train.py',B/'training_src/test.py',B/'training_src/utils/loss.py',B/'upstream_leaky/cfg/deploy/yolov7-tiny.yaml',B/'evaluate_reference.py',B/'reference_metrics.py']}
(run/'reproducibility.json').write_text(json.dumps(info,indent=2));print('PASS manifest, official initialization, original YOLOv8 and reference metadata')
