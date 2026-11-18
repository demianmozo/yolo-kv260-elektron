import json,shutil,tarfile
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'runs/reference/ptq';P=O/'kria';P.mkdir(exist_ok=True)
m=json.loads((O/'metrics.json').read_text());assert m['mAP50_drop_points']<=3
for source,name in [(O/'compiled/yolov7tiny.xmodel','yolov7tiny.xmodel'),(O/'decoder.json','decoder.json'),(B/'run_reference_kria.py','run_reference_kria.py'),(B/'ptq_board_post.py','ptq_board_post.py')]:shutil.copy2(source,P/name)
import csv
rows=[r for r in csv.DictReader((B/'data/manifest.csv').open()) if r['split']=='val']
for k in [0,8,17]:
 r=rows[k];stem='sample_P'+r['part'];shutil.copy2(r['path'],P/(stem+'.png'));shutil.copy2(O/'int8_board'/(r['name']+'.npz'),P/(stem+'.npz'))
with tarfile.open(O/'reference_kria.tar.gz','w:gz') as t:
 for p in sorted(P.iterdir()):t.add(p,arcname=p.name)
print(O/'reference_kria.tar.gz')
