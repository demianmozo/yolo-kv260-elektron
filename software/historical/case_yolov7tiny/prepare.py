import json,hashlib
from pathlib import Path
from collections import Counter
import torch,numpy as np
from models.yolo import Model
from tiny_raw import TinyRaw

torch.set_num_threads(4);torch.manual_seed(42)
base=Path('/workspace');out=base/'outputs';out.mkdir(exist_ok=True)
assert 'nonlinear_type: leaky_relu' in (base/'target.prototxt').read_text()
net=Model(str(base/'upstream_leaky/cfg/deploy/yolov7-tiny.yaml'),ch=3,nc=1).cpu().float().eval()
raw=TinyRaw(net).eval()
assert not any(isinstance(m,torch.nn.SiLU) for m in raw.modules())
acts=[m for m in raw.modules() if isinstance(m,torch.nn.LeakyReLU)]
assert acts and all(m.negative_slope==.1 for m in acts)
x=torch.from_numpy(np.load('/reference/probe_input.npy'))
with torch.no_grad():
    decoded,heads=net(x)
    actual=raw(x)
    for a,b in zip(actual,heads):
        reference=b.permute(0,1,4,2,3).reshape(a.shape)
        torch.testing.assert_close(a,reference,rtol=0,atol=0)
np.save(out/'probe_input.npy',x.numpy())
np.savez(out/'fp32_raw.npz',*[a.numpy() for a in actual])
with (out/'initialized_raw.pt').open('xb') as f:torch.save(raw,f,pickle_protocol=2)
weights=hashlib.sha256()
for k,v in raw.state_dict().items():weights.update(k.encode());weights.update(v.numpy().tobytes())
info=dict(seed=42,nc=1,class_name='fish',trained=False,input=[1,3,1024,1024],output_shapes=[list(a.shape) for a in actual],parameters=sum(p.numel() for p in raw.parameters()),modules=dict(Counter(type(m).__name__ for m in raw.modules())),weights_sha256=weights.hexdigest(),fp32_raw_error_max=0)
(out/'model.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2),flush=True)
