"""Evaluate original YOLOv8n on the exact saved YOLOv7 validation input tensors."""
from pathlib import Path
import os,json,hashlib
os.environ['YOLO_CONFIG_DIR']='/tmp/case-yolo-config';os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np,torch
from ultralytics import YOLO
B=Path(__file__).resolve().parent;O=B/'diagnostic';torch.set_num_threads(4)
p=B.parent/'case_yolov8n/inspeccion/runs/detect/train-2/weights/best.pt';net=YOLO(p).model.cpu().float().eval();assert net.names=={0:'fish'}
inputs=np.load(O/'val_inputs.npz');assert len(inputs.files)==18
with torch.inference_mode():
 for name in inputs.files:
  x=torch.from_numpy(inputs[name][None].astype(np.float32)/255);assert x.shape==(1,3,1024,1024)
  pred=net(x)[0].transpose(1,2);assert pred.shape[-1]==5
  # Common NMS input: joint YOLOv8 score as class score, objectness=1.
  equivalent=torch.cat([pred[:,:,:4],torch.ones_like(pred[:,:,4:5]),pred[:,:,4:5]],-1)
  np.savez_compressed(O/'predictions'/('y8_val_'+name+'.npz'),pred=equivalent.numpy())
print(json.dumps(dict(checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),images=18,preprocessing='exact shared uint8 tensors from YOLOv7 square validation',exposure='these 18 images belonged to original YOLOv8 training dataset; not independent test'),indent=2))
