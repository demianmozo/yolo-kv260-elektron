"""Isolated official trainer entry; only trusted local .pt files may be loaded."""
import os,sys,runpy
from pathlib import Path
os.environ['WANDB_MODE']='disabled';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'training_src'))
import torch
from functools import partial
torch.load=partial(torch.load,weights_only=False)
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.70,0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
runpy.run_path(str(B/'training_src/train.py'),run_name='__main__')
