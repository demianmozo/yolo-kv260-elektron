import sys,json,hashlib,copy
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch,numpy as np
from functools import partial
torch.load=partial(torch.load,weights_only=False)
from tiny_raw import TinyRaw
from trained_common import entries,preprocess
from utils.datasets import LoadImagesAndLabels
from utils.general import non_max_suppression

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
o=B/'trained';o.mkdir(exist_ok=True);p=B/'runs/fish/weights/best.pt'
ck=torch.load(p,map_location='cpu',weights_only=False);net=ck['ema'].float().eval()
assert net.names==['fish'] and net.model[-1].nc==1
assert sum(isinstance(m,torch.nn.Conv2d) for m in net.modules())==58
acts=[m for m in net.modules() if isinstance(m,torch.nn.LeakyReLU)];assert len(acts)==55 and all(m.negative_slope==.1 for m in acts)
assert not any(isinstance(m,torch.nn.SiLU) for m in net.modules())
np.savez(o/'best_state.npz',**{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
raw=TinyRaw(net).eval().cuda(); decoder=copy.deepcopy(net.model[-1]).eval();decoder.m=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)]);net.cuda();net.model[-1].grid=[torch.zeros(1) for _ in range(3)]
rows=entries(B,'val');loader=LoadImagesAndLabels(str(B/'data/val.txt'),1024,1,augment=False,rect=False,stride=32)
raw_out=o/'fp32_val';raw_out.mkdir(exist_ok=True)
max_abs=0.; relnum=0.;relden=0.;detection_abs=0.
with torch.no_grad():
 for i,r in enumerate(rows):
  x,_=preprocess(r['path']);j=next(j for j,n in enumerate(loader.img_files) if Path(n).name==r['name'])
  np.testing.assert_array_equal(x[0],loader[j][0].numpy().astype(np.float32)/255)
  xt=torch.from_numpy(x).cuda();origin=net(xt)[0].cpu();a=[v.cpu() for v in raw(xt)]
  decoded=decoder([v.clone() for v in a])[0];dif=decoded-origin
  max_abs=max(max_abs,float(dif.abs().max()));relnum+=float(dif.double().square().sum());relden+=float(origin.double().square().sum())
  torch.testing.assert_close(decoded,origin,rtol=1e-5,atol=.001)
  da=non_max_suppression(origin.clone(),.25,.6)[0];db=non_max_suppression(decoded.clone(),.25,.6)[0]
  assert da.shape==db.shape;torch.testing.assert_close(da,db,rtol=1e-5,atol=.001)
  if len(da):detection_abs=max(detection_abs,float((da-db).abs().max()))
  np.savez(raw_out/(r['name']+'.npz'),*[v.numpy() for v in a])
  if i==0:np.save(o/'probe_input.npy',x)
info=dict(best_checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),epoch=ck['epoch']+1,seed=42,classes=['fish'],resolution=1024,raw_decode_max_abs=max_abs,raw_decode_relative_l2=(relnum/relden)**.5,final_detections_max_abs=detection_abs,preprocessing='verified exactly against official square val loader for all 18 images; cv2 AREA resize + gray114 letterbox + RGB/NCHW/255',val_images=len(rows))
(o/'validation_setup.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
