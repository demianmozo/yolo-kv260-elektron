"""One batch-1 image; Graph Runner outputs versus Vitis AI quantized simulation."""
from pathlib import Path
import json,subprocess,time
import numpy as np
import xir,vitis_ai_library
B=Path(__file__).resolve().parent;fp=0x101000056010407
hardware=json.loads(subprocess.check_output(['xdputil','query'],universal_newlines=True))
assert [int(k['fingerprint'],16) for k in hardware['kernels']]==[fp]
g=xir.Graph.deserialize(str(B/'yolov7tiny.xmodel'));devices=[]
for s in g.get_root_subgraph().toposort_child_subgraph():
 device=s.get_attr('device');devices.append(device)
 if device=='DPU':assert s.get_attr('dpu_fingerprint')==fp
 if device=='CPU':assert {o.get_type() for o in s.get_ops()}=={'fix2float'}
assert devices.count('DPU')==1 and devices.count('CPU')==3
r=vitis_ai_library.GraphRunner.create_graph_runner(g);ins,outs=r.get_inputs(),r.get_outputs();assert len(ins)==1 and len(outs)==3
x=np.load(B/'probe_input.npy').transpose(0,2,3,1);dst=np.asarray(ins[0]);assert dst.shape==(1,1024,1024,3) and dst.dtype==np.int8
scale=2.**ins[0].get_tensor().get_attr('fix_point');dst[:]=np.clip(np.floor(x*scale+.5),-128,127).astype(np.int8)
t=time.perf_counter();assert r.wait(r.execute_async(ins,outs))==0;elapsed=1000*(time.perf_counter()-t)
reference=np.load(B/'quant_raw.npz');expected={tuple(reference['arr_'+str(i)].transpose(0,2,3,1).shape):reference['arr_'+str(i)].transpose(0,2,3,1) for i in range(3)}
values=[np.asarray(o).copy() for o in outs];metrics=[]
for v in values:
 e=expected[tuple(v.shape)];diff=v.astype(np.float64)-e
 metrics.append(dict(shape=list(v.shape),max_abs=float(np.max(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),different_values=int(np.count_nonzero(diff))))
# Save in canonical scale order, NCHW, for CPU decoding verification.
ordered=sorted(values,key=lambda v:-v.shape[1]);np.savez(B/'hardware_raw.npz',*[v.transpose(0,3,1,2) for v in ordered])
result=dict(fingerprint=hex(fp),DPU=1,CPU=3,cpu_ops=['fix2float']*3,batch=1,input=[1024,1024],graph_runner_first_run_ms=elapsed,outputs=metrics,exact=all(m['max_abs']==0 for m in metrics))
(B/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
assert result['exact'],'Hardware differs from quantized simulation; inspect numerical differences before accepting'
del r
