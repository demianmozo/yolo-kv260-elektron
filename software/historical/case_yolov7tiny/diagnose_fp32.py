"""Read-only checkpoint diagnosis. No training, optimizer step, calibration or compilation."""
import sys,json,copy,csv,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch,numpy as np
from functools import partial
torch.load=partial(torch.load,weights_only=False)
from trained_common import entries,preprocess
from utils.datasets import LoadImagesAndLabels
from utils.general import xywh2xyxy,scale_coords,non_max_suppression
from models.yolo import Detect

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
O=B/'diagnostic';O.mkdir(exist_ok=True);(O/'predictions').mkdir(exist_ok=True)
ck=torch.load(B/'runs/fish/weights/best.pt',map_location='cpu');last=torch.load(B/'runs/fish/weights/last.pt',map_location='cpu');init=torch.load(B/'runs/fish/weights/init.pt',map_location='cpu')
info={'best_epoch':ck['epoch']+1,'best_ema_updates':ck['updates'],'last_epoch':last['epoch']+1,'last_ema_updates':last['updates'],'best_fitness':float(np.asarray(ck['best_fitness']).item()),'input_shapes':[],'classes':ck['ema'].names,'nc':ck['ema'].model[-1].nc}
assert info['classes']==['fish'] and info['nc']==1
net=ck['ema'].float().eval().cuda();head=net.model[-1];head.grid=[torch.zeros(1) for _ in range(3)]
assert isinstance(head,Detect);decoder=copy.deepcopy(head).eval();decoder.m=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
anchors=head.anchor_grid.detach().cpu().reshape(-1,2).numpy();info['anchors_px']=anchors.tolist();info['stride']=head.stride.tolist()
# The one-class classification rows receive no BCE in upstream ComputeLossOTA.
cls_deltas={}
for key,model in [('best_model',ck['model']),('best_ema',ck['ema']),('last_model',last['model'])]:
 ds=[]
 for a,b in zip(init.model[-1].m,model.model[-1].m):
  ds.append(dict(weight_max_abs=float((a.weight[5::6].cpu()-b.weight[5::6].cpu()).abs().max()),bias_max_abs=float((a.bias[5::6].cpu()-b.bias[5::6].cpu()).abs().max())))
 cls_deltas[key]=ds
info['classification_rows_vs_init']=cls_deltas
allrows={s:entries(B,s) for s in ['train','val']}; inputs={}; shape_stats={};loaded={};maxdecode=0.;maxraw=0.;maxpost=0.;roundtrip=0.;conf_components={}
for split,rows in allrows.items():
 loader=LoadImagesAndLabels(str(B/'data'/(split+'.txt')),1024,1,augment=False,rect=False,stride=32)
 loaded[split]=dict(images=len(loader),labels=sum(len(x) for x in loader.labels),class_ids=sorted(set(np.concatenate(loader.labels)[:,0].tolist())))
 wh=[];norm=[];obj=[];cls=[];joint=[]
 with torch.inference_mode():
  for i,r in enumerate(rows):
   x,shape=preprocess(r['path']);h0,w0,h,w,ratio,pad=shape
   j=next(j for j,n in enumerate(loader.img_files) if Path(n).name==r['name'])
   image,targets,_,shapes=loader[j];np.testing.assert_array_equal(image.numpy().astype(np.float32)/255,x[0]);assert x.shape==(1,3,1024,1024)
   labels=np.loadtxt(r['label_path'],ndmin=2).astype(np.float32);wh.append(labels[:,3:5]*np.array([w,h]));norm.append(labels[:,3:5])
   gt=xywh2xyxy(torch.from_numpy(labels[:,1:]))*torch.tensor([w,h]*2)+torch.tensor(list(pad)*2)
   loader_gt=xywh2xyxy(targets[:,2:6])*1024;torch.testing.assert_close(gt,loader_gt,atol=.001,rtol=1e-5)
   # Audit inverse transformation against original label coordinates.
   back=gt.clone();back[:,[0,2]]=(back[:,[0,2]]-pad[0])/(w/w0);back[:,[1,3]]=(back[:,[1,3]]-pad[1])/(h/h0)
   roundtrip=max(roundtrip,float((back-xywh2xyxy(torch.from_numpy(labels[:,1:]))*torch.tensor([w0,h0]*2)).abs().max()))
   captured=[];hooks=[m.register_forward_hook(lambda m,a,v:captured.append(v.detach().clone())) for m in head.m]
   out,raw=net(torch.from_numpy(x).cuda())
   for hook in hooks:hook.remove()
   ext,_=decoder([z.clone() for z in captured]);maxdecode=max(maxdecode,float((out-ext).abs().max()));torch.testing.assert_close(out,ext,atol=0,rtol=0)
   for a,b in zip(captured,raw):
    ref=b.permute(0,1,4,2,3).reshape(a.shape);maxraw=max(maxraw,float((a-ref).abs().max()));assert torch.equal(a,ref)
   if i==0:
    a=non_max_suppression(out.clone(),.001,.6)[0];b=non_max_suppression(ext.clone(),.001,.6)[0];assert torch.equal(a,b)
   arr=out.cpu().numpy();np.savez_compressed(O/'predictions'/('y7_'+split+'_'+r['name']+'.npz'),pred=arr)
   # Counterfactual diagnostic only: retain boxes/objectness, bypass unused class score.
   obj.extend(arr[0,:,4].tolist());cls.extend(arr[0,:,5].tolist());joint.extend((arr[0,:,4]*arr[0,:,5]).tolist())
   if split=='val': inputs[r['name']]=(x[0]*255).round().astype(np.uint8)
   if (i+1)%20==0:print('evaluated',split,i+1,flush=True)
 wh=np.concatenate(wh);norm=np.concatenate(norm);rat=wh[:,None]/anchors[None];quality=np.minimum(rat,1/rat).min(2)
 shape_stats[split]=dict(normalized_percentiles={key:np.percentile(values,[0,5,25,50,75,95,100]).tolist() for key,values in [('width',norm[:,0]),('height',norm[:,1]),('area',norm.prod(1))]},input_pixel_percentiles={key:np.percentile(wh[:,i],[0,5,50,95,100]).tolist() for i,key in enumerate(['width','height'])},BPR=float((quality.max(1)>.25).mean()),anchors_per_target=float((quality>.25).sum(1).mean()),targets=len(wh))
 conf_components[split]={k:np.percentile(v,[0,50,90,99,99.9,100]).tolist() for k,v in [('objectness',obj),('class_probability',cls),('product',joint)]}
np.savez_compressed(O/'val_inputs.npz',**inputs)
info.update(loaded=loaded,shapes=shape_stats,confidence_components=conf_components,decoder=dict(type='pinned models.yolo.Detect.forward; learned convolutions replaced by identity only in external decoder',raw_max_abs=maxraw,pre_nms_max_abs=maxdecode,post_nms_sample_max_abs=maxpost),label_roundtrip_max_px=roundtrip,input_shapes=[[1,3,1024,1024]],matching_preprocessing='exactly identical to official square non-augmented loader on all 77 images')
# Reproduce the optimizer schedule without computing a gradient or changing weights.
nb=15;nw=45;events=[];discarded=0;used=0
for epoch in range(28):
 pending=0;steps=[];sizes=[]
 for i in range(nb):
  ni=i+nb*epoch;acc=max(1,np.interp(ni,[0,nw],[1,16]).round()) if ni<=nw else 16
  pending+=4 if i<14 else 3
  if ni%acc==0:steps.append(ni);sizes.append(pending);used+=pending;pending=0
 discarded+=pending;events.append(dict(epoch=epoch+1,steps=len(steps),effective_images=sizes,discarded_images=pending))
assert sum(e['steps'] for e in events[:last['epoch']+1])==last['updates'];assert sum(e['steps'] for e in events[:8])==ck['updates']
info['early_stop_audit']='check duplicated before and after validation; epoch 28 trained but skipped validation/save; last saved epoch 27'
info['optimizer_schedule']=dict(epochs=events,total_updates=sum(e['steps'] for e in events),last_saved_updates=last['updates'],best_updates=ck['updates'],images_contributing=used,images_discarded_before_optimizer=discarded,total_presented=59*28,reason='optimizer.zero_grad at epoch start; no flush of partial accumulation at epoch end; 15 batches/epoch vs accumulate=16 after warmup')
(O/'audit.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k not in ['optimizer_schedule','confidence_components','shapes']},indent=2))
