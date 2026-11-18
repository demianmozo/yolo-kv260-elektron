"""Identical NMS/matching for YOLOv7 train/val and YOLOv8 val; threshold curves."""
import sys,json,csv,base64,io
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import numpy as np,torch,cv2
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trained_common import entries,preprocess
from utils.general import non_max_suppression,box_iou,xywh2xyxy,scale_coords
from utils.metrics import ap_per_class
O=B/'diagnostic';torch.set_num_threads(4);thresholds=[.001,.01,.05,.10,.25]
def match(pred,gt):
 correct=torch.zeros((len(pred),10),dtype=torch.bool)
 if len(pred) and len(gt):
  v,idx=box_iou(pred[:,:4],gt).max(1);seen=set()
  for j in (v>.5).nonzero().flatten():
   d=int(idx[j]);j=int(j)
   if d not in seen:seen.add(d);correct[j]=v[j]>torch.linspace(.5,.95,10)
 return correct
summary={};table=[];curves={};visual=[]
for model,split,objonly in [('y7','train',False),('y7','val',False),('y8','val',False),('y7','train',True),('y7','val',True)]:
 key=model+'_'+split+('_objectness_only' if objonly else '');ts=[];scores=[];nlabels=0;caps=0;counts={t:[0,0,0] for t in thresholds};per=[]
 for r in entries(B,split):
  a=np.load(O/'predictions'/(model+'_'+split+'_'+r['name']+'.npz'))['pred'];out=torch.from_numpy(a.copy())
  if objonly:out[:,:,5]=1. # diagnostic counterfactual, no checkpoint change
  pred=non_max_suppression(out,.001,.6)[0];caps+=int(len(pred)==300)
  x,(h0,w0,h,w,ratio,pad)=preprocess(r['path']);lab=np.loadtxt(r['label_path'],ndmin=2).astype(np.float32)
  gt=xywh2xyxy(torch.from_numpy(lab[:,1:]))*torch.tensor([w,h]*2)+torch.tensor(list(pad)*2);pn=pred.clone();gn=gt.clone()
  scale_coords((1024,1024),pn[:,:4],(h0,w0),((h/h0,w/w0),pad));scale_coords((1024,1024),gn,(h0,w0),((h/h0,w/w0),pad))
  correct=match(pn,gn);ts.append(correct.numpy());scores.append(pred[:,4].numpy());nlabels+=len(lab)
  for t in thresholds:
   keep=pred[:,4]>=t;tp=int(match(pn[keep],gn)[:,0].sum());counts[t][0]+=tp;counts[t][1]+=int(keep.sum())-tp;counts[t][2]+=len(lab)-tp
  per.append(dict(name=r['name'],detections_at_0001=len(pred),TP_at_0001=int(correct[:,0].sum()),labels=len(lab)))
  if model=='y7' and not objonly and int(r['part']) in [1,30,66,83]:
   img=(x[0].transpose(1,2,0)*255).round().astype(np.uint8)[:,:,::-1];ok,data=cv2.imencode('.png',img);assert ok
   visual.append(dict(name=split+' P'+r['part'],image='data:image/png;base64,'+base64.b64encode(data).decode(),gt=gt.tolist(),pred=pred.tolist(),original_wh=[w0,h0],input_content_wh=[w,h],original_sizes=(lab[:,3:5]*np.array([w0,h0])).tolist()))
 tp=np.concatenate(ts);conf=np.concatenate(scores);order=np.argsort(-conf);tp=tp[order];conf=conf[order]
 _,_,ap,_,_=ap_per_class(tp,conf,np.zeros(len(conf)),np.zeros(nlabels),plot=False)
 recall=tp[:,0].cumsum()/nlabels;precision=tp[:,0].cumsum()/np.arange(1,len(tp)+1);curves[key]=(recall,precision,conf)
 summary[key]=dict(images=len(per),labels=nlabels,mAP50=float(ap[:,0].mean()),mAP50_95=float(ap.mean()),images_hitting_max_det_300=caps,thresholds={},per_image=per)
 for t,(a,b,c) in counts.items():
  row=dict(model=key,confidence=t,P=a/max(a+b,1),R=a/max(a+c,1),TP=a,FP=b,FN=c);summary[key]['thresholds'][str(t)]=row;table.append(row)
with (O/'thresholds.csv').open('w') as f:wri=csv.DictWriter(f,fieldnames=list(table[0]));wri.writeheader();wri.writerows(table)
(O/'metrics.json').write_text(json.dumps(dict(protocol='shared square1024 inputs; native scores; official YOLOv7 NMS IoU .6, maxdet300, conf>=.001; native image clipping; identical greedy IoU>.5 matching; AP uses all predictions >=.001, never prefiltered at .25',results=summary),indent=2))
fig,ax=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for key in ['y7_train','y7_val','y8_val']:
 rec,prec,conf=curves[key];ax[0,0].plot(rec,prec,label=key,linewidth=1);ax[0,1].plot(conf,rec,label=key)
ax[0,0].set(xlabel='Recall',ylabel='Precision',title='PR (IoU > 0.5)',xlim=(0,1),ylim=(0,1));ax[0,0].legend()
ax[0,1].set(xlabel='Confianza',ylabel='Recall',title='Recall por confianza',xscale='log',xlim=(.001,1),ylim=(0,1));ax[0,1].legend()
lines=[l.split() for l in (B/'runs/fish/results.txt').read_text().splitlines()];v=np.array([[float(x) for x in l[2:]] for l in lines]);epochs=np.arange(1,len(v)+1)
for axes,k,title in [(ax[0,2],0,'Box loss'),(ax[1,0],1,'Objectness loss'),(ax[1,1],2,'Classification loss')]:
 axes.plot(epochs,v[:,k],label='train (OTA)');axes.plot(epochs,v[:,10+k],label='val (ComputeLoss)');axes.axvline(8,color='k',ls=':',label='best=8');axes.set(title=title,xlabel='Epoca registrada');axes.legend()
for key in ['train','val']:
 wh=np.concatenate([np.loadtxt(r['label_path'],ndmin=2)[:,3:5] for r in entries(B,key)]);ax[1,2].scatter(wh[:,0],wh[:,1],s=4,alpha=.2,label=key)
ax[1,2].set(title='Cajas normalizadas',xlabel='Ancho',ylabel='Alto');ax[1,2].legend();fig.savefig(O/'curves.png',dpi=150);plt.close(fig)
# Interactive views keep all >=.001 predictions inspectable without truncating the list.
html='''<!doctype html><meta charset="utf-8"><title>Diagnóstico FP32</title><style>body{font:15px sans-serif;margin:20px;background:#eee}canvas{width:min(95vw,850px);height:auto;background:black}article{display:inline-block;vertical-align:top;margin:10px}#tip{position:fixed;background:white;border:1px solid;padding:8px;pointer-events:none;display:none}label{margin-right:20px}</style><h2>YOLOv7-tiny EMA, época 8 — train y val</h2><p>Verde: etiquetas reales. Rojo: predicciones. Tamaños en píxeles de entrada 1024. Al pasar el cursor: confianza y tamaños originales. Se muestran todas las predicciones después de NMS (máximo 300), desde 0,001.</p><label>Confianza <select id="thr"><option>.001</option><option>.01</option><option>.05</option><option>.10</option><option>.25</option></select></label><label><input type="checkbox" id="gt" checked>Etiquetas reales y tamaños</label><label><input type="checkbox" id="pr" checked>Predicciones</label><label><input type="checkbox" id="txt" checked>Confianza sobre cada caja</label><div id="cards"></div><div id="tip"></div><script>const data=DATA;const cards=document.getElementById('cards');for(const [i,d] of data.entries()){const a=document.createElement('article');a.innerHTML='<h3>'+d.name+'</h3><canvas width="1024" height="1024"></canvas>';cards.appendChild(a);d.c=a.querySelector('canvas');d.im=new Image();d.im.onload=()=>draw(d);d.im.src=d.image;d.c.onmousemove=e=>{const b=d.c.getBoundingClientRect(),x=(e.clientX-b.left)*1024/b.width,y=(e.clientY-b.top)*1024/b.height;let lines=[];d.pred.forEach(p=>{if(p[4]>=+thr.value&&x>=p[0]&&x<=p[2]&&y>=p[1]&&y<=p[3])lines.push('Pred conf='+p[4].toFixed(5)+' tamaño entrada='+Math.round(p[2]-p[0])+'×'+Math.round(p[3]-p[1])+' px')});d.gt.forEach((p,j)=>{if(x>=p[0]&&x<=p[2]&&y>=p[1]&&y<=p[3])lines.push('GT '+(j+1)+' original='+d.original_sizes[j].map(Math.round).join('×')+' px; entrada='+Math.round(p[2]-p[0])+'×'+Math.round(p[3]-p[1]))});tip.textContent=lines.join(' | ');tip.style.display=lines.length?'block':'none';tip.style.left=Math.min(e.clientX+10,innerWidth-450)+'px';tip.style.top=e.clientY+15+'px'};d.c.onmouseleave=()=>tip.style.display='none'}function box(c,p,color,text){c.strokeStyle=color;c.lineWidth=1;c.strokeRect(p[0],p[1],p[2]-p[0],p[3]-p[1]);c.font='11px monospace';c.fillStyle=color;c.fillText(text,p[0],Math.max(11,p[1]))}function draw(d){const c=d.c.getContext('2d');c.drawImage(d.im,0,0);if(gt.checked)d.gt.forEach((p,j)=>box(c,p,'#00ff55','GT'+(j+1)+' '+Math.round(p[2]-p[0])+'×'+Math.round(p[3]-p[1])));if(pr.checked)d.pred.filter(p=>p[4]>=+thr.value).forEach(p=>box(c,p,'#ff5555',txt.checked?p[4].toFixed(3):''))}for(const e of [thr,gt,pr,txt])e.onchange=()=>data.forEach(draw);</script>'''
(O/'visual_review.html').write_text(html.replace('DATA',json.dumps(visual)))
# Static panels for direct inspection at the requested lower threshold.
for d in visual:
 im=cv2.imdecode(np.frombuffer(base64.b64decode(d['image'].split(',')[1]),np.uint8),1);left=im.copy();right=im.copy()
 for j,p in enumerate(d['gt']):
  a=tuple(np.array(p[:2],int));b=tuple(np.array(p[2:],int));cv2.rectangle(left,a,b,(0,255,0),1);cv2.putText(left,f'{j+1}:{p[2]-p[0]:.0f}x{p[3]-p[1]:.0f}',a,cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,0),1)
 for p in d['pred']:
  a=tuple(np.array(p[:2],int));b=tuple(np.array(p[2:4],int));cv2.rectangle(right,a,b,(0,0,255),1);cv2.putText(right,f'{p[4]:.3f}',a,cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
 cv2.imwrite(str(O/(d['name'].replace(' ','_')+'.png')),np.concatenate([left,right],axis=1))
print(json.dumps({k:{a:b for a,b in v.items() if a not in ['per_image']} for k,v in summary.items()},indent=2))
