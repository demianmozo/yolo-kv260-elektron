import gzip,pickle,json,statistics
from pathlib import Path
O=Path(__file__).resolve().parent/'runs/reference/ptq';x=pickle.load(gzip.open(O/'trace.xat','rb'));frames=[];pending={}
for e in x['vart']:
 if e.get('section')!='TRACE':continue
 cls=e['classname'];t=float(e['ts'])
 if cls=='dpu-runner':frames.append(dict(dpu_hardware_ms=0.,dpu_controller_ms=0.,cpu_fix2float_ms=0.,CPU_calls=0));continue
 if cls not in ['dpu-controller','cpu-task']:continue
 key=(e['pid'],cls)
 if e['event_state']=='1':pending[key]=t
 else:
  ms=(t-pending.pop(key))*1000
  if cls=='dpu-controller':frames[-1]['dpu_controller_ms']+=ms;frames[-1]['dpu_hardware_ms']+=int(e['hwconuter'])/300000.
  else:frames[-1]['cpu_fix2float_ms']+=ms;frames[-1]['CPU_calls']+=1
assert len(frames)==39 and not pending and all(f['CPU_calls']==3 for f in frames)
normal=json.loads((O/'board_results.json').read_text());profiled=json.loads((O/'board_results_profiled.json').read_text());result={}
for k,(lo,hi) in zip(normal['benchmarks'],[(6,21),(24,39)]):
 row={name:statistics.mean(f[name] for f in frames[lo:hi]) for name in ['dpu_hardware_ms','dpu_controller_ms','cpu_fix2float_ms']}
 for kind,obj in [('normal',normal),('profiled',profiled)]:
  row[kind+'_total_ms']=obj['benchmarks'][k]['latency']['total']['mean_ms'];row[kind+'_graph_runner_ms']=obj['benchmarks'][k]['latency']['graph_runner']['mean_ms']
 row['profiling_total_overhead_percent']=100*(row['profiled_total_ms']/row['normal_total_ms']-1);result[k]=row
(O/'profile_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
