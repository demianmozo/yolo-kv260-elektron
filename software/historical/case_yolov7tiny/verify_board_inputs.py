import sys,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;sys.path[:0]=[str(B/'training_src')]
from trained_common import entries,preprocess
O=B/'runs/reference/ptq';results=[]
for r in entries(B,'val'):
 p=O/'hardware_diagnostic'/('sample_P'+r['part']+'_input.npy')
 if not p.exists():continue
 host=preprocess(r['path'])[0];arm=np.load(p);a=np.load(O/'hardware_diagnostic'/('sample_P'+r['part']+'_hardware.npz'));q=np.load(O/'int8_board'/(r['name']+'.npz'))
 error=[]
 for j in range(3):
  x=a['arr_'+str(j)];y=q['arr_'+str(j)];np.testing.assert_array_equal(x,y);error.append(float(abs(x-y).max()))
 results.append(dict(image=r['name'],resize_different_values=int(np.count_nonzero(host-arm)),resize_max_abs=float(abs(host-arm).max()),input_int8_different_values=int(np.count_nonzero(np.floor(host*64+.5)-np.floor(arm*64+.5))),DPU_vs_same_input_simulation_max_abs=error))
assert len(results)==3
(O/'input_parity.json').write_text(json.dumps(dict(host_opencv='4.12.0',ARM_opencv='4.5.2',input_fix_point=6,results=results),indent=2));print(json.dumps(results,indent=2))
