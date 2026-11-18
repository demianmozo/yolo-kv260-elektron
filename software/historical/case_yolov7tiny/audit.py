from collections import Counter
import xir,json
from pathlib import Path
p=Path('/workspace/outputs')
g=xir.Graph.deserialize(str(p/'quantized/TinyRaw_int.xmodel'))
slopes=Counter(str(o.get_attrs()) for o in g.get_ops() if o.get_type()=='leaky-relu')
print('LeakyReLU XIR:',slopes,flush=True)
g=xir.Graph.deserialize(str(p/'compiled/yolov7tiny.xmodel'))
rows=[]
for s in g.get_root_subgraph().toposort_child_subgraph():
 device=s.get_attr('device');ops=Counter(o.get_type() for o in s.get_ops())
 if device=='DPU':assert s.get_attr('dpu_fingerprint')==0x101000056010407
 if device=='CPU':assert set(ops)=={'fix2float'},ops
 rows.append(dict(name=s.get_name(),device=device,ops=dict(ops)))
print(json.dumps(rows,indent=2))
nonlinear=Counter(str(o.get_attrs().get('nonlinear')) for o in g.get_ops() if o.get_type()=='conv2d-fix')
print('Convoluciones/activaciones fusionadas:',nonlinear)
assert sum(s['device']=='DPU' for s in rows)==1
assert sum(s['device']=='CPU' for s in rows)==3
(p/'audit.json').write_text(json.dumps(dict(subgraphs=rows,nonlinear=dict(nonlinear)),indent=2))
