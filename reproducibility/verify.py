#!/usr/bin/env python3
"""Portable read-only verification: standard library only; no experimental execution."""
import csv,hashlib,json,math,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
for line in (R/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split(maxsplit=1);assert digest(R/name)==expected,name
def percentile(values,p):
 a=sorted(values);index=(len(a)-1)*p;lo=int(index);hi=min(lo+1,len(a)-1)
 return a[lo]+(a[hi]-a[lo])*(index-lo)
for mode in ['normal','profiled']:
 rows=list(csv.DictReader((R/f'evidence/platform_B/y7_{mode}_samples.csv').open()))
 expected=json.loads((R/f'evidence/platform_B/y7_{mode}.json').read_text())
 assert len(rows)==expected['repeats']==100
 for key,ref in expected['statistics_ms'].items():
  a=[float(r[key]) for r in rows];actual=dict(mean=statistics.mean(a),median=statistics.median(a),std=statistics.stdev(a),p95=percentile(a,.95))
  for k,v in actual.items():assert math.isclose(v,ref[k],rel_tol=1e-9,abs_tol=1e-8),(mode,key,k)
normal=json.loads((R/'evidence/platform_B/y7_normal.json').read_text());rows=list(csv.DictReader((R/'evidence/platform_B/y7_normal_power.csv').open()))
active=[r for r in rows if r['phase']=='inference'];t0,t1=normal['measured_interval_s']
t=[t0]+[float(r['time_s']) for r in active]+[t1];p=[float(active[0]['power_W'])]+[float(r['power_W']) for r in active]+[float(active[-1]['power_W'])]
energy=sum((b-a)*(x+y)/2 for a,b,x,y in zip(t,t[1:],p,p[1:]));ref=json.loads((R/'evidence/platform_B/new_platform_power_energy.json').read_text())
assert math.isclose(energy/100,ref['total_energy_J_per_image'],rel_tol=1e-9)
print('OK: source snapshot hashes; 100 normal + 100 profiled samples; telemetry energy. No inference or hardware executed.')
