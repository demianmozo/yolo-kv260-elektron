"""Immutable source audit and blocked internal validation; no acoustic reconstruction."""
import csv, hashlib, json, re, struct, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'dataset'; OUT=BASE/'data'; OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
images=sorted((ROOT/'images/train').glob('*.png')); labels=list((ROOT/'labels/train').glob('*.txt'))
assert len(images)==84 and len(labels)==84
assert {p.stem for p in images}=={p.stem for p in labels}
candidates=(OUT/'source_search_paths.txt').read_text().splitlines()
parents=sorted({str(Path(p).parent) for p in candidates if Path(p).name=='project_Fish_ID_Training_Data_1_P_39.ID'})
rows=[]; provenance={}; seen={}
for p in images:
 m=re.fullmatch(r'(project_Fish_ID_Training_Data_(\d+))_P_(\d+)_CH(\d+)(_auto)?',p.stem); assert m
 seq,series,part,channel,auto=m.groups();part=int(part)
 q=ROOT/'labels/train'/(p.stem+'.txt');a=np.loadtxt(q,ndmin=2)
 assert a.shape[1]==5 and np.isfinite(a).all() and (a[:,0]==0).all()
 assert (a[:,1:]>=0).all() and (a[:,1:]<=1).all() and (a[:,3:]>0).all()
 assert (a[:,1:3]-a[:,3:]/2>=-1e-6).all() and (a[:,1:3]+a[:,3:]/2<=1+1e-6).all()
 assert len(np.unique(a,axis=0))==len(a)
 im=Image.open(p); pix=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
 split='excluded_annotation_conflict' if part==39 else 'train' if part<=60 else 'gap' if part<=65 else 'val'
 possible=[]
 for parent in parents:
  f=Path(parent)/(seq+'_P_'+str(part)+'.ID')
  if f.exists():
   possible.append(str(f))
   with f.open('rb') as h:b=h.read(48)
   if len(b)==48:
    v=struct.unpack('<f7I2f2I',b)
    provenance[str(f)]=dict(version=v[0],start_date_raw=v[1],start_time_raw=v[2],end_date_raw=v[3],end_time_raw=v[4],pings=v[5],sounder=v[6],channels=v[7],association='name-only; NOT confirmed PNG source')
 rows.append(dict(name=p.name,path=str(p),sha256=sha(p),pixel_sha256=pix,label_path=str(q),label_sha256=sha(q),boxes=len(a),width=im.width,height=im.height,sequence=seq,part=part,channel=channel,campaign='',transect='',source_file='',source_status='unconfirmed; homonymous raw candidates',source_candidates=json.dumps(possible),png_metadata=json.dumps(im.info),split=split))
 seen.setdefault(pix,[]).append(p.name)
rows.sort(key=lambda r:(r['part'],r['name']))
with (OUT/'manifest.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'source_candidates.json').write_text(json.dumps(provenance,indent=2))
dups=[v for v in seen.values() if len(v)>1];assert len(dups)==1
p39=[r for r in rows if r['part']==39];aa=[np.loadtxt(r['label_path'],ndmin=2) for r in p39]
def xyxy(a):return np.concatenate((a[:,1:3]-a[:,3:]/2,a[:,1:3]+a[:,3:]/2),1)
a,b=map(xyxy,aa); inter=np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0).prod(2)
iou=inter/((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter)
review=Image.new('RGB',(1200,750),'white')
for k,r in enumerate(p39):
 im=Image.open(r['path']).convert('RGB');d=ImageDraw.Draw(im)
 for n,box in enumerate(xyxy(aa[k])):
  z=box*np.array([im.width,im.height,im.width,im.height]);d.rectangle(tuple(z),outline='red',width=2);d.text(tuple(z[:2]),str(n+1),fill='black')
 im.thumbnail((590,710));review.paste(im,(600*k,35));ImageDraw.Draw(review).text((600*k+5,10),r['name']+' '+str(r['boxes'])+' boxes',fill='black')
review.save(OUT/'p39_review.png')
for split in ['train','val']:
 chosen=[r for r in rows if r['split']==split]
 # Copy labels into derived view; image symlinks avoid touching original files or caches.
 for kind in ['images','labels']:(OUT/kind/split).mkdir(parents=True,exist_ok=True)
 paths=[]
 for r in chosen:
  dst=OUT/'images'/split/r['name']
  if not dst.exists():dst.symlink_to(r['path'])
  (OUT/'labels'/split/Path(r['label_path']).name).write_bytes(Path(r['label_path']).read_bytes())
  paths.append(str(dst))
 (OUT/(split+'.txt')).write_text('\n'.join(paths)+'\n')
assert not ({r['pixel_sha256'] for r in rows if r['split']=='train'} & {r['pixel_sha256'] for r in rows if r['split']=='val'})
summary=dict(images=84,unique_images=len(seen),duplicates=dups,p39_boxes=[r['boxes'] for r in p39],p39_exact_shared_boxes=len(set(map(tuple,aa[0]))&set(map(tuple,aa[1]))),p39_iou50_matches_per_side=[int((iou.max(1)>=.5).sum()),int((iou.max(0)>=.5).sum())],p39_status='both excluded, visual expert review pending; no merge',splits={s:dict(images=sum(r['split']==s for r in rows),boxes=sum(r['boxes'] for r in rows if r['split']==s)) for s in ['train','val','gap','excluded_annotation_conflict']},test=None,split_rule='internal blocked holdout: P1-60 train except P39; P61-65 embargo; P66-83 val. One name-derived acoustic group; no independent test. Campaign and temporal independence NOT established.',original_yaml=(ROOT/'fish.yaml').read_text(),manifest_sha256=sha(OUT/'manifest.csv'))
(OUT/'split.json').write_text(json.dumps(summary,indent=2));(OUT/'fish.yaml').write_text('train: '+str(OUT/'train.txt')+'\nval: '+str(OUT/'val.txt')+'\nnc: 1\nnames: [fish]\n')
print(json.dumps(summary,indent=2))
