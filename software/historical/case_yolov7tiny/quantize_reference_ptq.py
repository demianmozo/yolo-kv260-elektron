import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np
from models.yolo import Model
from tiny_raw import TinyRaw
from trained_common import entries,preprocess
from pytorch_nndct.apis import torch_quantizer
p=argparse.ArgumentParser();p.add_argument('mode',choices=['calib','test','board']);a=p.parse_args()
B=Path('/workspace');o=B/'runs/reference/ptq';torch.set_num_threads(4);torch.manual_seed(42)
net=Model(str(B/'upstream_leaky/cfg/deploy/yolov7-tiny.yaml'),ch=3,nc=1).float().eval()
st=np.load(o/'best_state.npz');net.load_state_dict({k:torch.from_numpy(st[k]) for k in st.files},strict=True)
model=TinyRaw(net).eval();x=torch.from_numpy(np.load(o/'probe_input.npy'))
q=torch_quantizer('test' if a.mode=='board' else a.mode,model,(x,),device=torch.device('cpu'),output_dir=str(o/'quantized'))
rows=entries(B,'train' if a.mode=='calib' else 'val')
assert len(rows)==(59 if a.mode=='calib' else 18)
if a.mode=='board':rows=[r for r in rows if int(r['part']) in [66,74,83]]
if a.mode!='calib':(o/('int8_board' if a.mode=='board' else 'int8_val')).mkdir(exist_ok=True)
with torch.no_grad():
 for i,r in enumerate(rows):
  path=Path('/dataset/images/train')/r['name'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
  arr,_=preprocess(path)
  if a.mode=='board':arr=np.load(o/'hardware_diagnostic'/('sample_P'+r['part']+'_input.npy'))
  out=q.quant_model(torch.from_numpy(arr))
  if a.mode!='calib':
   np.savez(o/('int8_board' if a.mode=='board' else 'int8_val')/(r['name']+'.npz'),*[v.numpy() for v in out])
   if a.mode=='test' and i==0:np.savez(o/'quant_raw.npz',*[v.numpy() for v in out])
  if (i+1)%5==0:print(a.mode,i+1,'/',len(rows),flush=True)
 if a.mode=='calib':
  q.export_quant_config();(o/'calibration.json').write_text(json.dumps(dict(split='train',images=[dict(name=r['name'],sha256=r['sha256']) for r in rows],synthetic=False,preprocessing='same square validation preprocessing'),indent=2))
 elif a.mode=='test':
  q.export_xmodel(output_dir=str(o/'quantized'),deploy_check=False)
print('PASS',a.mode,flush=True)
