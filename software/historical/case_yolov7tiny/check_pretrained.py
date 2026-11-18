from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch
from models.yolo import Model
torch.set_num_threads(4)
p=B/'pretrained/yolov7-tiny.pt';ck=torch.load(p,map_location='cpu',weights_only=False);pre=ck['model'].float();torch.manual_seed(42)
model=Model(str(B/'upstream_leaky/cfg/deploy/yolov7-tiny.yaml'),ch=3,nc=1).float()
assert type(pre.model[-1]).__name__=='Detect'
assert len([m for m in pre.modules() if isinstance(m,torch.nn.LeakyReLU)])==55
assert all(m.negative_slope==.1 for m in pre.modules() if isinstance(m,torch.nn.LeakyReLU))
source=pre.state_dict(); target=model.state_dict(); transfer={k:v for k,v in source.items() if k in target and v.shape==target[k].shape and 'anchor' not in k}
missing=[k for k in target if k not in transfer];assert all(k.startswith('model.77.') for k in missing),missing
model.load_state_dict(transfer,strict=False)
info=dict(source_url='https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),transferred=len(transfer),total=len(target),not_transferred=missing,reason='80 COCO classes to one fish class; retain fixed deploy anchors',conv_count=sum(isinstance(m,torch.nn.Conv2d) for m in model.modules()),leaky_count=sum(isinstance(m,torch.nn.LeakyReLU) for m in model.modules()))
(B/'pretrained/compatibility.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
