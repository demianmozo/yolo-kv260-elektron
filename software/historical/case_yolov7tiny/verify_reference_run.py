"""Verify identical initialization and exact training/validation/checkpoint accounting."""
from pathlib import Path
import argparse,sys,json,hashlib,csv
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch,yaml
p=argparse.ArgumentParser();p.add_argument('--run',default='reference');a=p.parse_args();run=B/'runs'/a.run
torch.set_num_threads(4)
old=torch.load(B/'archive/defective/run/weights/init.pt',map_location='cpu',weights_only=False)
new=torch.load(run/'weights/init.pt',map_location='cpu',weights_only=False)
assert old.state_dict().keys()==new.state_dict().keys()
for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
assert new.nc==1 and new.names==['fish']
assert sum(isinstance(m,torch.nn.Conv2d) for m in new.modules())==58
assert sum(isinstance(m,torch.nn.LeakyReLU) and m.negative_slope==.1 for m in new.modules())==55
before=yaml.safe_load((B/'archive/defective/run/opt.yaml').read_text());after=yaml.safe_load((run/'opt.yaml').read_text())
for key in ['weights','cfg','data','hyp','seed','batch_size','img_size','rect','noautoanchor','multi_scale','adam','workers','label_smoothing']:
 assert before[key]==after[key],key
assert (run/'hyp.yaml').read_bytes()==(B/'archive/defective/run/hyp.yaml').read_bytes()
records=[json.loads(l) for l in (run/'epoch_accounting.jsonl').read_text().splitlines()];assert len(records)>0
for i,r in enumerate(records,1):
 assert r['epoch']==i and r['batches']==15 and r['images']==59
 assert r['actual_updates']==r['expected_updates']==15 and r['cumulative_updates']==15*i
 assert r['validation_calls']==r['last_saves']==1
last=torch.load(run/'weights/last.pt',map_location='cpu',weights_only=False);best=torch.load(run/'weights/best.pt',map_location='cpu',weights_only=False)
assert last['epoch']+1==len(records) and last['updates']==last['optimizer_steps']==15*len(records)
best_index=max(range(len(records)),key=lambda i:records[i]['val_mAP50_95'])
assert best['epoch']==best_index and best['optimizer_steps']==best['updates']==15*(best_index+1)
assert best['best_fitness']==records[best_index]['val_mAP50_95']
assert len((run/'results.txt').read_text().splitlines())==len(records)
for r in csv.DictReader((B/'data/manifest.csv').open()):
 for path,key in [('path','sha256'),('label_path','label_sha256')]:assert hashlib.sha256(Path(r[path]).read_bytes()).hexdigest()==r[key]
info=dict(initial_state_exactly_identical=True,seed=42,split_unchanged=True,epochs=len(records),batches_per_epoch=15,images_per_epoch=59,updates_per_epoch=15,total_optimizer_steps=last['optimizer_steps'],best_epoch=best_index+1,best_updates=best['updates'],best_val_mAP50_95=best['best_fitness'],checkpoint_sha256=hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest(),selection='val_mAP50_95',patience=after['patience'],accumulate=1,validation_and_last_save_once_per_epoch=True,original_images_and_labels_unchanged=True)
(run/'verification.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
