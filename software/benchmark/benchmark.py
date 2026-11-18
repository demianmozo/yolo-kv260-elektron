"""Common reproducible KV260 benchmark; same preloaded images and postprocess protocol."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,csv,threading,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2,xir,vitis_ai_library
from postprocess import preprocess_image,decode,decode8,nms
p=argparse.ArgumentParser();p.add_argument('--model',choices=['y7','y8'],required=True);p.add_argument('--profile',action='store_true');p.add_argument('--repeats',type=int,default=100);p.add_argument('--warmup',type=int,default=10);a=p.parse_args()
assert a.repeats>=100 and a.warmup>=10
B=Path(__file__).resolve().parents[1];O=B/'results';O.mkdir(exist_ok=True);tag=a.model+('_profiled' if a.profile else '_normal');cfg=json.loads((B/'protocol.json').read_text());cv2.setNumThreads(1)
images=[cv2.imread(str(B/'artifacts'/name)) for name in cfg['samples']];assert all(im is not None for im in images)
model=B/'artifacts'/(a.model+'.xmodel');g=xir.Graph.deserialize(str(model));subs=g.get_root_subgraph().toposort_child_subgraph();owner={op.get_output_tensor().name:i for i,s in enumerate(subs) for op in s.get_ops()}
def tensor(t):return dict(name=t.name,shape=list(t.dims),dtype=str(t.dtype),bytes=t.get_data_size())
rows=[];edges=[]
for i,s in enumerate(subs):
 dev=s.get_attr('device');attrs={}
 if dev=='DPU':assert s.get_attr('dpu_fingerprint')==int(cfg['fingerprint'],16)
 for k in ['reg_id_to_size','reg_id_to_context_type','reg_id_to_hw_segment','workload']:
  if s.has_attr(k):attrs[k]=s.get_attr(k)
 rows.append(dict(index=i,name=s.get_name(),device=dev,ops=[dict(name=op.get_name(),type=op.get_type()) for op in s.toposort()],inputs=[tensor(t) for t in s.get_input_tensors()],outputs=[tensor(t) for t in s.get_output_tensors()],attrs=attrs))
 for t in s.get_input_tensors():
  j=owner[t.name];edges.append(dict(source=j,destination=i,source_device=subs[j].get_attr('device'),destination_device=dev,tensor=tensor(t)))
(O/(a.model+'_graph.json')).write_text(json.dumps(dict(subgraphs=rows,edges=edges),indent=2))
r=vitis_ai_library.GraphRunner.create_graph_runner(g);ins,outs=r.get_inputs(),r.get_outputs();dst=np.asarray(ins[0]);scale=2.**ins[0].get_tensor().get_attr('fix_point');assert dst.shape==(1,1024,1024,3) and dst.dtype==np.int8
head=json.loads((B/'artifacts/y7_decoder.json').read_text());anchors=np.array(head['anchors'],np.float32);strides=np.array(head['strides'],np.float32)
phases={'value':'setup'};stop=threading.Event();sensor_rows=[]
channels=[]
for d in sorted(Path('/sys/class/hwmon').glob('hwmon*')):
 for f in sorted(d.glob('power*_input')):
  channels.append((str(f),f,(d/'name').read_text().strip(),(f.with_name(f.name.replace('_input','_label'))).read_text().strip() if f.with_name(f.name.replace('_input','_label')).exists() else 'unlabelled'))
def monitor():
 while not stop.is_set():
  t=time.perf_counter()
  for name,path,chip,label in channels:
   try:sensor_rows.append(dict(time_s=t,phase=phases['value'],sensor=name,chip=chip,label=label,power_W=float(path.read_text())/1e6))
   except OSError:pass
  stop.wait(.5)
def memory():
 keys=['MemTotal','MemAvailable','CmaTotal','CmaFree'];d={}
 for path in ['/proc/meminfo','/proc/self/status']:
  for line in Path(path).read_text().splitlines():
   k=line.split(':')[0]
   if k in keys+['VmRSS','VmHWM']:d[k+'_kB']=int(line.split()[1])
 return d

def run(im):
 t=[time.perf_counter()];x=preprocess_image(im);t.append(time.perf_counter())
 dst[:]=np.clip(np.floor(x.transpose(0,2,3,1)*scale+.5),-128,127).astype(np.int8);t.append(time.perf_counter())
 assert r.wait(r.execute_async(ins,outs))==0;t.append(time.perf_counter())
 values=sorted([np.asarray(o).copy() for o in outs],key=lambda v:(-v.shape[1],-v.shape[3]));raw=[v.transpose(0,3,1,2) for v in values];t.append(time.perf_counter())
 pred=decode(raw,anchors,strides) if a.model=='y7' else decode8(raw);t.append(time.perf_counter());d=nms(pred,.25,.6);t.append(time.perf_counter())
 # Output coordinates to source image; timed with NMS, same as the retained protocol.
 h0,w0=im.shape[:2];gain=1024/max(h0,w0);h,w=int(h0*gain),int(w0*gain)
 d[:,[0,2]]=np.clip((d[:,[0,2]]-(1024-w)/2)/(h/h0),0,w0);d[:,[1,3]]=np.clip((d[:,[1,3]]-(1024-h)/2)/(h/h0),0,h0);t[-1]=time.perf_counter()
 return np.diff(t)*1000,raw,d
mem_before=memory();phases['value']='warmup'
for i in range(a.warmup):run(images[i%len(images)])
thread=None
if channels and not a.profile:
 phases['value']='idle_before';thread=threading.Thread(target=monitor,daemon=True);thread.start();time.sleep(15)
keys=['preprocessing_ms','input_quantize_copy_ms','graph_runner_ms','output_copy_ms','decode_ms','nms_coordinates_ms'];samples=[];phases['value']='inference';start=time.perf_counter()
for i in range(a.repeats):
 times,raw,dets=run(images[i%len(images)]);samples.append(dict(iteration=i,image=cfg['samples'][i%len(images)],**dict(zip(keys,map(float,times))),total_ms=float(times.sum()),detections=len(dets)))
 if i==0:probe=raw
 if (i+1)%10==0:print(tag,i+1,'/',a.repeats,flush=True)
end=time.perf_counter();phases['value']='idle_after'
if thread:time.sleep(15);stop.set();thread.join()
np.savez(O/(tag+'_raw_probe.npz'),*probe)
for name,data in [(tag+'_samples.csv',samples),(tag+'_power.csv',sensor_rows)]:
 if data:
  with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={}
for key in keys+['total_ms']:
 v=np.array([s[key] for s in samples]);summary[key]=dict(mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)),p95=float(np.percentile(v,95)))
result=dict(model=a.model,profiled=a.profile,checkpoint_scope='frozen XMODEL',xmodel_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),warmup=a.warmup,repeats=a.repeats,statistics_ms=summary,measured_interval_s=[start,end],memory_before=mem_before,memory_after=memory(),power_channels=[dict(path=c[0],chip=c[2],label=c[3]) for c in channels],opencv=cv2.__version__,numpy=np.__version__,cpu_threads=1)
(O/(tag+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);del r
