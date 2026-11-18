"""Latencia secuencial Graph Runner, batch=1; no incluye decoder/NMS ni imagen/SSH."""
import argparse
import json
import time
from pathlib import Path
import subprocess
import numpy as np
import xir
import vitis_ai_library

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repeats',type=int,default=10)
p.add_argument('--warmup',type=int,default=3)
a=p.parse_args();assert 1<=a.repeats<=30 and 1<=a.warmup<=5
base=Path(__file__).resolve().parent
fp=0x101000056010407
hardware=json.loads(subprocess.check_output(['xdputil','query'],universal_newlines=True))
assert [int(k['fingerprint'],16) for k in hardware['kernels']]==[fp]
t0=time.perf_counter()
g=xir.Graph.deserialize(str(base/'yolov7tiny.xmodel'))
for s in g.get_root_subgraph().toposort_child_subgraph():
    if s.get_attr('device')=='DPU': assert s.get_attr('dpu_fingerprint')==fp
    elif s.get_attr('device')=='CPU': assert {o.get_type() for o in s.get_ops()}=={'fix2float'}
runner=vitis_ai_library.GraphRunner.create_graph_runner(g)
inputs,outputs=runner.get_inputs(),runner.get_outputs()
assert len(inputs)==1 and len(outputs)==3
initialization_ms=(time.perf_counter()-t0)*1000
sample=np.load(base/'probe_input.npy').transpose(0,2,3,1)
dst=np.asarray(inputs[0]);assert dst.dtype==np.int8 and sample.shape==dst.shape
scale=2.**inputs[0].get_tensor().get_attr('fix_point')
refs=np.load(base/'quant_raw.npz')
expected={tuple(refs['arr_'+str(i)].transpose(0,2,3,1).shape):refs['arr_'+str(i)].transpose(0,2,3,1) for i in range(3)}
assert len(expected)==3

def run():
    start=time.perf_counter()
    dst[:]=np.clip(np.floor(sample*scale+.5),-128,127).astype(np.int8)
    prepared=time.perf_counter()
    assert runner.wait(runner.execute_async(inputs,outputs))==0
    finished=time.perf_counter()
    values=[np.asarray(o).copy() for o in outputs]
    copied=time.perf_counter()
    return [1000*(prepared-start),1000*(finished-prepared),1000*(copied-finished),1000*(copied-start)],values

for _ in range(a.warmup): _,values=run()
for value in values: np.testing.assert_array_equal(value,expected[tuple(value.shape)])
measurements=[]
for i in range(a.repeats):
    times,values=run();measurements.append(times)
    if (i+1)%5==0: print('Medidos',i+1,'/',a.repeats,flush=True)
for value in values: np.testing.assert_array_equal(value,expected[tuple(value.shape)])
rows=np.asarray(measurements)
summary={}
for i,key in enumerate(('input_quantize_copy','graph_runner','output_copy','total_buffers')):
    summary[key]={'median_ms':float(np.median(rows[:,i])),'mean_ms':float(rows[:,i].mean()),
                  'min_ms':float(rows[:,i].min()),'max_ms':float(rows[:,i].max())}
print(json.dumps(dict(batch=1,resolution=[1024,1024],warmup=a.warmup,repeats=a.repeats,initialization_ms=initialization_ms,
                     latency=summary,sequential_fps_buffers=1000/float(rows[:,3].mean()),
                     output_error_max=0,excluded=['image loading/preprocessing','DFL/decoding','NMS','SSH/network'],samples_ms=measurements),indent=2),flush=True)
del runner
