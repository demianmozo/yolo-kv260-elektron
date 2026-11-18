"""Freeze FP32 validation F1 operating point, then compare PTQ with identical protocol."""
import sys,json,copy,hashlib
from pathlib import Path
import numpy as np,torch
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
from trained_common import entries,preprocess
from utils.general import non_max_suppression,box_iou,xywh2xyxy,scale_coords
from utils.metrics import ap_per_class
O=B/'runs/reference/ptq';O.mkdir(exist_ok=True);torch.set_num_threads(4)
def match(p,g):
 c=torch.zeros((len(p),10),dtype=torch.bool)
 if len(p) and len(g):
  v,ix=box_iou(p[:,:4],g).max(1);seen=set()
  for j in (v>.5).nonzero().flatten().tolist():
   k=int(ix[j])
   if k not in seen:seen.add(k);c[j]=v[j]>torch.linspace(.5,.95,10)
 return c.numpy()
def stats(mode):
 ts=[];ss=[];n=0
 if mode=='int8':
  ck=torch.load(B/'runs/reference/weights/best.pt',map_location='cpu',weights_only=False)
  dec=copy.deepcopy(ck['ema'].float().eval().model[-1]);dec.m=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
 for r in entries(B,'val'):
  if mode=='fp32':out=torch.from_numpy(np.load(B/'runs/reference/evaluation/predictions'/('y7_val_'+r['name']+'.npz'))['pred'].copy())
  else:
   a=np.load(O/'int8_val'/(r['name']+'.npz'))
   with torch.no_grad():out=dec([torch.from_numpy(a['arr_'+str(j)]) for j in range(3)])[0]
  p=non_max_suppression(out,.001,.6)[0]
  _,(h0,w0,h,w,ratio,pad)=preprocess(r['path']);lab=np.loadtxt(r['label_path'],ndmin=2).astype(np.float32)
  g=xywh2xyxy(torch.from_numpy(lab[:,1:]))*torch.tensor([w,h]*2)+torch.tensor(list(pad)*2)
  scale_coords((1024,1024),p[:,:4],(h0,w0),((h/h0,w/w0),pad));scale_coords((1024,1024),g,(h0,w0),((h/h0,w/w0),pad))
  ts.append(match(p,g));ss.append(p[:,4].numpy());n+=len(g)
 t=np.concatenate(ts);s=np.concatenate(ss);order=np.argsort(-s);t=t[order];s=s[order]
 _,_,ap,_,_=ap_per_class(t,s,np.zeros(len(s)),np.zeros(n),plot=False)
 return t,s,n,float(ap[:,0].mean()),float(ap.mean())
def counts(t,s,n,threshold):
 tp=int(t[s>=threshold,0].sum());fp=int((s>=threshold).sum())-tp;fn=n-tp;p=tp/max(tp+fp,1);r=tp/n
 return dict(confidence=threshold,P=p,R=r,F1=2*p*r/max(p+r,1e-12),TP=tp,FP=fp,FN=fn)
t,s,n,a,b=stats('fp32');op=O/'operating_point.json'
if not op.exists():
 # Evaluate at every distinct score; tied scores are included together.
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];tp=t[:,0].cumsum();f=2*tp[ends]/(ends+1+n);k=ends[int(f.argmax())];threshold=float(s[k])
 op.write_text(json.dumps(dict(selection='maximum empirical F1 on FP32 val, IoU>0.5; ties prefer higher confidence',checkpoint_sha256=hashlib.sha256((B/'runs/reference/weights/best.pt').read_bytes()).hexdigest(),**counts(t,s,n,threshold)),indent=2))
threshold=json.loads(op.read_text())['confidence'];result={}
for mode in (['fp32','int8'] if (O/'int8_val').exists() else ['fp32']):
 if mode=='int8':t,s,n,a,b=stats(mode)
 result[mode]=dict(images=18,labels=n,mAP50=a,mAP50_95=b,at_025=counts(t,s,n,.25),at_operating=counts(t,s,n,threshold))
if 'int8' in result:
 diffs=[]
 for j in range(3):
  sumd=sume=total=0;mx=0;groups={k:[0.,0,0.] for k in ['xy','wh','objectness','class']}
  for r in entries(B,'val'):
   f=np.load(O/'fp32_val'/(r['name']+'.npz'))['arr_'+str(j)].astype(np.float64);q=np.load(O/'int8_val'/(r['name']+'.npz'))['arr_'+str(j)].astype(np.float64);d=q-f
   shaped=d.reshape(1,3,6,*d.shape[2:])
   for name,sl in [('xy',slice(0,2)),('wh',slice(2,4)),('objectness',slice(4,5)),('class',slice(5,6))]:
    v=shaped[:,:,sl];groups[name][0]+=float((v*v).sum());groups[name][1]+=v.size;groups[name][2]=max(groups[name][2],float(abs(v).max()))
   sumd+=float((d*d).sum());sume+=float((f*f).sum());total+=d.size;mx=max(mx,float(abs(d).max()))
  diffs.append(dict(scale=j,shape=list(f.shape),max_abs=mx,RMSE=(sumd/total)**.5,relative_L2=(sumd/sume)**.5,channels={k:dict(RMSE=(v[0]/v[1])**.5,max_abs=v[2]) for k,v in groups.items()}))
 result['raw_outputs']=diffs;result['mAP50_drop_points']=100*(result['fp32']['mAP50']-result['int8']['mAP50'])
(O/'metrics.json').write_text(json.dumps(result,indent=2));print(op.read_text());print(json.dumps(result,indent=2))
