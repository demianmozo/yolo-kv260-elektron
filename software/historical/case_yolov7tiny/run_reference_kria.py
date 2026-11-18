"""Verify three validation ecograms, then benchmark the complete sequential pipeline."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,subprocess
from pathlib import Path
import numpy as np,cv2,xir,vitis_ai_library
from ptq_board_post import preprocess_image,decode,nms
cv2.setNumThreads(1)
B=Path(__file__).resolve().parent;cfg=json.loads((B/'decoder.json').read_text());fp=0x101000056010407
hw=json.loads(subprocess.check_output(['xdputil','query'],universal_newlines=True));assert [int(k['fingerprint'],16) for k in hw['kernels']]==[fp]
g=xir.Graph.deserialize(str(B/'yolov7tiny.xmodel'));devices=[]
for sub in g.get_root_subgraph().toposort_child_subgraph():
 dev=sub.get_attr('device');devices.append(dev)
 if dev=='DPU':assert sub.get_attr('dpu_fingerprint')==fp
 if dev=='CPU':assert {op.get_type() for op in sub.get_ops()}=={'fix2float'}
assert devices.count('DPU')==1 and devices.count('CPU')==3
r=vitis_ai_library.GraphRunner.create_graph_runner(g);ins,outs=r.get_inputs(),r.get_outputs();dst=np.asarray(ins[0]);assert dst.shape==(1,1024,1024,3) and dst.dtype==np.int8
scale=2.**ins[0].get_tensor().get_attr('fix_point');anchors=np.array(cfg['anchors'],np.float32);strides=np.array(cfg['strides'],np.float32)
files=sorted(B.glob('sample_*.png'))
assert len(files)==3

def run(path,confidence):
 t=[time.perf_counter()];im=cv2.imread(str(path));assert im is not None;t.append(time.perf_counter())
 x=preprocess_image(im);t.append(time.perf_counter())
 dst[:]=np.clip(np.floor(x.transpose(0,2,3,1)*scale+.5),-128,127).astype(np.int8);t.append(time.perf_counter())
 assert r.wait(r.execute_async(ins,outs))==0;t.append(time.perf_counter())
 values=sorted([np.asarray(o).copy() for o in outs],key=lambda a:-a.shape[1]);raw=[v.transpose(0,3,1,2) for v in values];t.append(time.perf_counter())
 pred=decode(raw,anchors,strides);t.append(time.perf_counter());dets=nms(pred,confidence)
 h0,w0=im.shape[:2];gain=1024/max(h0,w0);h,w=int(h0*gain),int(w0*gain)
 dets[:,[0,2]]=np.clip((dets[:,[0,2]]-(1024-w)/2)/(h/h0),0,w0);dets[:,[1,3]]=np.clip((dets[:,[1,3]]-(1024-h)/2)/(h/h0),0,h0)
 t.append(time.perf_counter());return np.diff(t)*1000,raw,dets

checks=[]
for p in files:
 np.save(B/(p.stem+'_input.npy'),preprocess_image(cv2.imread(str(p))))
 _,raw,dets=run(p,cfg['confidence']);ref=np.load(p.with_suffix('.npz'));errors=[]
 for j,v in enumerate(raw):
  expected=ref['arr_'+str(j)];d=v.astype(np.float64)-expected
  errors.append(dict(scale=j,shape=list(v.shape),max_abs=float(abs(d).max()),RMSE=float(np.sqrt((d*d).mean()))))
  errors[-1]['different_values']=int(np.count_nonzero(d))
 np.savez(B/(p.stem+'_hardware.npz'),*raw)
 checks.append(dict(image=p.name,outputs=errors,detections=len(dets)))
(B/'raw_checks.json').write_text(json.dumps(checks,indent=2))
assert all(e['max_abs']==0 for c in checks for e in c['outputs']), 'Hardware differs from simulation; benchmark stopped'
print('PASS: three ecograms, all raw outputs exactly equal to INT8 simulation',flush=True)
keys=['image_read','preprocessing','input_quantize_copy','graph_runner','output_copy','decode','nms_and_coordinates'];results={}
for confidence in [cfg['confidence'],.25]:
 for p in files:run(p,confidence)
 measurements=[]
 for _ in range(5):
  for p in files:measurements.append(run(p,confidence)[0].tolist())
 a=np.asarray(measurements);summary={k:dict(mean_ms=float(a[:,j].mean()),median_ms=float(np.median(a[:,j]))) for j,k in enumerate(keys)}
 summary['total']=dict(mean_ms=float(a.sum(1).mean()),median_ms=float(np.median(a.sum(1))))
 results[str(confidence)]=dict(warmup=3,repeats=15,latency=summary,samples_ms=measurements)
result=dict(fingerprint=hex(fp),batch=1,input=[1024,1024],DPU=1,CPU=3,cpu_ops=['fix2float']*3,checks=checks,benchmarks=results,graph_runner_scope='DPU + runtime synchronization + 3 fix2float; DPU-only measured separately by vaitrace',disk_read='warm page cache',opencv_threads=1)
(B/'board_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['latency'] for k,v in results.items()},indent=2),flush=True)
del r
