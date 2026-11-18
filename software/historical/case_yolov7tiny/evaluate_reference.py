"""FP32 inference only on the frozen train/val lists; original YOLOv7 decoder."""
import sys,argparse,json
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch,numpy as np
from trained_common import entries,preprocess
p=argparse.ArgumentParser();p.add_argument('--run',default='reference');a=p.parse_args();run=B/'runs'/a.run
O=run/'evaluation';(O/'predictions').mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ck=torch.load(run/'weights/best.pt',map_location='cpu',weights_only=False);net=ck['ema'].float().eval().cuda();net.model[-1].grid=[torch.zeros(1) for _ in range(3)]
assert net.names==['fish'] and net.model[-1].nc==1
with torch.inference_mode():
 for split in ['train','val']:
  rows=entries(B,split)
  for i,r in enumerate(rows):
   x,_=preprocess(r['path']);assert x.shape==(1,3,1024,1024)
   y=net(torch.from_numpy(x).cuda())[0];assert y.shape==(1,64512,6)
   np.savez_compressed(O/'predictions'/('y7_'+split+'_'+r['name']+'.npz'),pred=y.cpu().numpy())
  print('FP32',split,len(rows),'images',flush=True)
(O/'checkpoint.json').write_text(json.dumps(dict(run=a.run,epoch=ck['epoch']+1,EMA=True,optimizer_steps=ck['optimizer_steps'],resolution=1024,selection='val_mAP50_95'),indent=2))
