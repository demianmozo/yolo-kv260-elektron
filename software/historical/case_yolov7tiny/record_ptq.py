import hashlib,json,csv
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'runs/reference/ptq'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader((B/'data/manifest.csv').open()));cal=json.loads((O/'calibration.json').read_text());train={r['sha256'] for r in rows if r['split']=='train'};val={r['sha256'] for r in rows if r['split']=='val'}
assert len(cal['images'])==59 and {r['sha256'] for r in cal['images']}==train and not train&val
op=json.loads((O/'operating_point.json').read_text());assert sha(B/'runs/reference/weights/best.pt')==op['checkpoint_sha256']
files=['prepare_reference_ptq.py','quantize_reference_ptq.py','run_reference_ptq.sh','audit_reference_ptq.py','ptq_metrics.py','ptq_board_post.py','check_ptq_post.py','run_reference_kria.py','verify_board_inputs.py','package_reference_kria.py','summarize_ptq_profile.py']
result=dict(checkpoint='runs/reference/weights/best.pt',checkpoint_sha256=op['checkpoint_sha256'],checkpoint_component='EMA',seed=42,batch=1,resolution=[1024,1024],class_name='fish',calibration_images=59,validation_images=18,calibration_train_only=True,split_sha256=sha(B/'data/manifest.csv'),container='xilinx/vitis-ai-pytorch-cpu@sha256:f55bd069ffd56c6358cae29df19e6085f2bcf8ea5e045744aa412fd72db521ed',vitis_ai='3.0',fingerprint='0x101000056010407',operating_confidence=op['confidence'],NMS_iou=.6,max_detections=300,AP_min_confidence=.001,quantized_leaky_slope=.1015625,original_leaky_slope=.1,quant_info_sha256=sha(O/'quantized/quant_info.json'),compiled_sha256=sha(O/'compiled/yolov7tiny.xmodel'),scripts_sha256={f:sha(B/f) for f in files},QAT=False,ARM_reference_inputs='hardware_diagnostic/*_input.npy; OpenCV 4.5.2 resize rounding differs by one level from host 4.12.0; raw parity checked with identical ARM tensors')
(O/'reproducibility.json').write_text(json.dumps(result,indent=2));print('PASS: checkpoint hash unchanged, calibration provenance and generated artifact hashes recorded')
