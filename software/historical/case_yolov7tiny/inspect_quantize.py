import argparse
from pathlib import Path
import torch,numpy as np
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('mode',choices=('inspect','calib','test'));a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
out=Path('/workspace/outputs')
model=torch.load(str(out/'initialized_raw.pt'),map_location='cpu').eval()
x=torch.from_numpy(np.load(out/'probe_input.npy'))
if a.mode=='inspect':
 from pytorch_nndct import Inspector
 Inspector('0x101000056010407').inspect(model,(x,),device=torch.device('cpu'),output_dir=str(out/'inspector'),image_format=None)
else:
 from pytorch_nndct.apis import torch_quantizer
 q=torch_quantizer(a.mode,model,(x,),device=torch.device('cpu'),output_dir=str(out/'quantized'))
 with torch.no_grad():
  if a.mode=='calib':
   for i in range(4):q.quant_model(torch.rand_like(x))
   q.export_quant_config()
  else:
   y=q.quant_model(x)
   np.savez(out/'quant_raw.npz',*[v.numpy() for v in y])
   q.export_xmodel(output_dir=str(out/'quantized'),deploy_check=False)
   import xir
   g=xir.Graph.deserialize(str(out/'quantized/TinyRaw_int.xmodel'))
   print('XIR',Counter(o.get_type() for o in g.get_ops()),flush=True)
print('PASS',a.mode,flush=True)
