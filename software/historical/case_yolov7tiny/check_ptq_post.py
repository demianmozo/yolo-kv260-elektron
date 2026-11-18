import sys,json,copy
from pathlib import Path
import numpy as np,torch,cv2
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
from trained_common import entries,preprocess
from ptq_board_post import preprocess_image,decode,nms
from utils.general import non_max_suppression
O=B/'runs/reference/ptq';torch.set_num_threads(4)
ck=torch.load(B/'runs/reference/weights/best.pt',map_location='cpu',weights_only=False);d=copy.deepcopy(ck['ema'].float().eval().model[-1]);d.m=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
anchors=d.anchor_grid.numpy().reshape(3,3,2);strides=d.stride.numpy();threshold=json.loads((O/'operating_point.json').read_text())['confidence'];mx=0.;dm=0.;tested=0
for mode in ['fp32','int8']:
 if not (O/(mode+'_val')).exists():continue
 for r in entries(B,'val'):
  np.testing.assert_array_equal(preprocess_image(cv2.imread(r['path'])),preprocess(r['path'])[0])
  a=np.load(O/(mode+'_val')/(r['name']+'.npz'));raw=[a['arr_'+str(j)] for j in range(3)]
  with torch.no_grad():ref=d([torch.from_numpy(x.copy()) for x in raw])[0].numpy()[0]
  out=decode(raw,anchors,strides);mx=max(mx,float(abs(out-ref).max()));np.testing.assert_allclose(out,ref,rtol=2e-6,atol=.001)
  for c in [.25,threshold]:
   official=non_max_suppression(torch.from_numpy(ref[None].copy()),.001,.6)[0].numpy();official=official[official[:,4]>=c];actual=nms(out,c)
   assert len(actual)==len(official),(mode,r['name'],c,len(actual),len(official))
   # Ignore ordering of exactly tied confidence scores, compare nearest complete detections.
   err=np.abs(actual[:,None,:]-official[None,:,:]).max(axis=2);err=float(max(err.min(axis=0).max(),err.min(axis=1).max())) if len(actual) else 0
   dm=max(dm,err);assert err<.002,(mode,r['name'],c,err)
  tested+=1
(O/'decoder.json').write_text(json.dumps(dict(anchors=anchors.tolist(),strides=strides.tolist(),confidence=threshold)))
result=dict(images_tested=tested,preprocessing_exact=True,decode_max_abs=mx,detection_max_abs=dm)
(O/'postprocessing_verification.json').write_text(json.dumps(result,indent=2));print(result)
