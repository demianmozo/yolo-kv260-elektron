"""Record concrete versions, source changes and immutable input hashes."""
import json,hashlib,csv,zipfile
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
changes=[]
for p in (B/'training_src').rglob('*.py'):
 src=B/'upstream_leaky'/p.relative_to(B/'training_src')
 if p.read_bytes()!=src.read_bytes():changes.append(dict(path=str(p.relative_to(B/'training_src')),original_sha256=sha(src),modified_sha256=sha(p)))
(B/'data/trainer_changes.json').write_text(json.dumps(changes,indent=2))
rows=list(csv.DictReader((B/'data/manifest.csv').open()))
for r in rows:
 assert sha(Path(r['path']))==r['sha256'];assert sha(Path(r['label_path']))==r['label_sha256']
z=zipfile.ZipFile(B.parent/'dataset.zip')
assert all(z.read('images/train/'+r['name'])==Path(r['path']).read_bytes() for r in rows)
assert all(z.read('labels/train/'+Path(r['label_path']).name)==Path(r['label_path']).read_bytes() for r in rows)
assert sha(B.parent/'case_yolov8n/inspeccion/runs/detect/train-2/weights/best.pt')=='6ad6dabb90618f1ddec98066bccf27cf987144a0d32f8dedf0413d3e969a6cf2'
info=dict(source_commit='eef4f2c92851e4d72c0630ee1f3615144353a73e',cfg='upstream_leaky/cfg/deploy/yolov7-tiny.yaml',cfg_sha256=sha(B/'upstream_leaky/cfg/deploy/yolov7-tiny.yaml'),seed=42,train_images=59,val_images=18,test=None,manifest_sha256=sha(B/'data/manifest.csv'),pretrained=json.loads((B/'pretrained/compatibility.json').read_text()),checkpoint=json.loads((B/'trained/validation_setup.json').read_text()),training=dict(precision='FP32; AMP/TF32 disabled',batch_size=4,nominal_accumulated_batch=64,optimizer='SGD',max_epochs=150,early_stopping_patience=20,completed_epochs=28,selection='official fitness = 0.1*mAP50 + 0.9*mAP50_95 on internal val',learning_rate=.01,warmup_epochs=3,augmentation='official geometric defaults, HSV/mosaic/mixup/paste-in disabled for acoustic imagery',fixed_anchors=True),quantization=dict(tool='Vitis AI 3.0 CPU',fingerprint='0x101000056010407',leaky_slope_fp32=.1,leaky_slope_int8=.1015625,calibration_split='train only, 59 unique images'),original_images_labels_and_yolov8_checkpoint_verified_unchanged=True)
(B/'trained/run.json').write_text(json.dumps(info,indent=2));print('PASS immutable originals, pinned configuration and training provenance')
