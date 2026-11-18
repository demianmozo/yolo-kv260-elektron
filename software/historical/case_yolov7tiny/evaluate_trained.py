"""FP32/INT8 on the frozen internal validation block, identical decoding and NMS."""
import sys,json,copy
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import numpy as np,torch,cv2
from utils.general import non_max_suppression,box_iou,xywh2xyxy,scale_coords
from utils.metrics import ap_per_class
from trained_common import entries,preprocess

torch.set_num_threads(4);o=B/'trained';ck=torch.load(B/'runs/fish/weights/best.pt',map_location='cpu',weights_only=False)
decoder=copy.deepcopy(ck['ema'].float().eval().model[-1]);decoder.m=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
def match(pred,gt):
 correct=torch.zeros((len(pred),10),dtype=torch.bool);ids={}
 if len(pred) and len(gt):
  values,idx=box_iou(pred[:,:4],gt).max(1);seen=set()
  for j in (values>.5).nonzero().flatten():
   d=int(idx[j]);j=int(j)
   if d not in seen:
    seen.add(d);correct[j]=values[j]>torch.linspace(.5,.95,10);ids[j]=d
 return correct,ids
stats={}; rows=entries(B,'val');visual=[]
with torch.no_grad():
 for mode in ['fp32','int8']:
  ts=[];cs=[];pc=[];tc=[];TP=FP=FN=0;per=[]
  for i,r in enumerate(rows):
   raw=np.load(o/(mode+'_val')/(r['name']+'.npz'));predicted=decoder([torch.from_numpy(raw['arr_'+str(j)]) for j in range(3)])[0]
   pred=non_max_suppression(predicted.clone(),conf_thres=.001,iou_thres=.6)[0]
   x,(h0,w0,h,w,ratio,pad)=preprocess(r['path']);labels=np.loadtxt(r['label_path'],ndmin=2).astype(np.float32)
   gt=xywh2xyxy(torch.from_numpy(labels[:,1:]))*torch.tensor([w*ratio[0],h*ratio[1]]*2)+torch.tensor(list(pad)*2)
   # Official validation clips boxes in original image coordinates before matching.
   pred_native=pred.clone();gt_native=gt.clone()
   scale_coords((1024,1024),pred_native[:,:4],(h0,w0),((h/h0,w/w0),pad))
   scale_coords((1024,1024),gt_native,(h0,w0),((h/h0,w/w0),pad))
   correct,_=match(pred_native,gt_native);ts.append(correct.numpy());cs.append(pred[:,4].numpy());pc.append(pred[:,5].numpy());tc.append(labels[:,0])
   selected=pred[pred[:,4]>=.25];hits,ids=match(pred_native[pred[:,4]>=.25],gt_native);tp=int(hits[:,0].sum());fp=len(selected)-tp;fn=len(gt)-tp;TP+=tp;FP+=fp;FN+=fn
   per.append(dict(name=r['name'],TP=tp,FP=fp,FN=fn))
   if mode=='fp32' and i in [0,8,17]:
    im=(x[0].transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1].copy()
    for j,box in enumerate(gt):cv2.rectangle(im,tuple(box[:2].int().tolist()),tuple(box[2:].int().tolist()),(0,200,0) if j in ids.values() else (0,255,255),1)
    for j,box in enumerate(selected):cv2.rectangle(im,tuple(box[:2].int().tolist()),tuple(box[2:4].int().tolist()),(255,0,0) if j in ids else (0,0,255),2)
    cv2.putText(im,f"P{r['part']} TP={tp} FP={fp} FN={fn}",(20,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    visual.append(cv2.resize(im,(512,512)))
  p,rec,ap,f1,classes=ap_per_class(np.concatenate(ts),np.concatenate(cs),np.concatenate(pc),np.concatenate(tc),plot=False)
  stats[mode]=dict(images=len(rows),labels=TP+FN,precision_at_025=TP/max(TP+FP,1),recall_at_025=TP/max(TP+FN,1),mAP50=float(ap[:,0].mean()),mAP50_95=float(ap.mean()),TP=TP,FP=FP,FN=FN,per_image=per)
cv2.imwrite(str(o/'val_review.png'),np.concatenate(visual,axis=1))
result=dict(split='internal blocked validation P66-83; NO independent test',checkpoint_sha256=json.loads((o/'validation_setup.json').read_text())['sha256'],thresholds=dict(AP_conf=.001,count_conf=.25,NMS_iou=.6,matching_iou=.5,max_det=300),results=stats)
(o/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='per_image'} for k,v in stats.items()},indent=2))
