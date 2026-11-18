"""Make an isolated runnable copy of the pinned official trainer; original snapshot intact."""
from pathlib import Path
import shutil,json,hashlib
B=Path(__file__).resolve().parent; src=B/'upstream_leaky';dst=B/'training_src'
if dst.exists(): raise SystemExit('training_src exists; inspect before regeneration')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def patch(file,a,b):
 p=dst/file;s=p.read_text();assert a in s,(file,a);p.write_text(s.replace(a,b))
# Modern NumPy/PyTorch compatibility only.
for p in dst.rglob('*.py'):
 s=p.read_text();import re
 s=re.sub(r'np\.int\b','int',s);s=re.sub(r'np\.float\b','float',s)
 s=s.replace('torch.load(', 'torch.load(') # trusted checkpoint policy supplied in entry point
 s=s.replace('gj.clamp_(0, gain[3] - 1)', 'gj.clamp_(0, int(gain[3]) - 1)').replace('gi.clamp_(0, gain[2] - 1)', 'gi.clamp_(0, int(gain[2]) - 1)')
 s=s.replace('torch.ones(size=(len(b),))','torch.ones(size=(len(b),), device=b.device)')
 p.write_text(s)
patch(Path('train.py'),'init_seeds(2 + rank)','init_seeds(opt.seed)')
patch(Path('train.py'),'plots = not opt.evolve','plots = False')
patch(Path('train.py'),'enabled=cuda','enabled=False')
patch(Path('train.py'),"rect=True, rank=-1","rect=False, rank=-1")
patch(Path('train.py'),'model.half().float()  # pre-reduce anchor precision','model.float()  # retain original FP32 weights and anchors')
patch(Path('train.py'),"nw = max(round(hyp['warmup_epochs'] * nb), 1000)","nw = max(round(hyp['warmup_epochs'] * nb), 1)")
patch(Path('train.py'),'start_epoch, best_fitness = 0, 0.0','start_epoch, best_fitness = 0, -1.0\n    best_epoch = -1')
patch(Path('train.py'),'best_fitness = fi\n','best_fitness = fi\n                best_epoch = epoch\n')
patch(Path('train.py'),"else model).half()","else model).float()")
patch(Path('train.py'),"deepcopy(ema.ema).half()","deepcopy(ema.ema).float()")
p=dst/'train.py';s=p.read_text();start=s.index('                if (best_fitness == fi) and (epoch >= 200):');end=s.index('                if wandb_logger.wandb:',start);s=s[:start]+s[end:];s=s.replace('        # end epoch ',"        if epoch - best_epoch >= opt.patience:\n            logger.info('Early stopping: epoch %d, best epoch %d', epoch + 1, best_epoch + 1)\n            break\n\n        # end epoch ");s=s.replace('                strip_optimizer(f)  # strip optimizers','                pass  # retain full FP32 best checkpoint and optimizer for reproducibility')
s=s.replace("parser.add_argument('--epochs'", "parser.add_argument('--seed', type=int, default=42)\n    parser.add_argument('--patience', type=int, default=20)\n    parser.add_argument('--epochs'")
p.write_text(s)
patch(Path('test.py'),'half_precision=True','half_precision=False')
# Frozen hyperparameters: color represents acoustic amplitude, so no HSV jitter;
# no synthetic composites/paste-in with uncertain or densely overlapping fish labels.
p=dst/'data/hyp.scratch.tiny.yaml';s=p.read_text()
for key in ['hsv_h','hsv_s','hsv_v','mosaic','mixup','paste_in']:
 s=re.sub(r'^'+key+r': [^\n]+',key+': 0.0',s,flags=re.M)
(B/'data/hyp.yaml').write_text(s)
from fix_training import fix_reference_trainer
fix_reference_trainer(dst/"train.py")
changes=[]
for p in dst.rglob('*.py'):
 original=src/p.relative_to(dst)
 if p.read_bytes()!=original.read_bytes():changes.append(dict(path=str(p.relative_to(dst)),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),modified_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(B/'data/trainer_changes.json').write_text(json.dumps(changes,indent=2))
