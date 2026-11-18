import xir,json
from pathlib import Path
from collections import Counter
B=Path('/workspace/trained');g=xir.Graph.deserialize(str(B/'compiled/yolov7tiny.xmodel'))
rows=[]
for s in g.get_root_subgraph().toposort_child_subgraph():
 device=s.get_attr('device');ops=Counter(o.get_type() for o in s.get_ops())
 if device=='DPU':assert s.get_attr('dpu_fingerprint')==0x101000056010407
 if device=='CPU':assert set(ops)=={'fix2float'}
 rows.append(dict(name=s.get_name(),device=device,ops=dict(ops)))
assert Counter(r['device'] for r in rows)==dict(USER=1,DPU=1,CPU=3)
info=dict(subgraphs=rows,fingerprint='0x101000056010407');(B/'audit.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
