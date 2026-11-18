"""A missing step must be detected; checks actual optimizer calls, not loop intentions."""
import torch
from training_checks import StepCounter
p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.01);counter=StepCounter(opt)
for epoch in range(2):
 before=counter.total
 for batch_size in [4]*14+[3]:
  (p.square()*batch_size).backward();opt.step();opt.zero_grad(set_to_none=True)
 assert counter.verify_epoch(before,15,59,15,59)==15
assert counter.total==30
try:counter.verify_epoch(counter.total,15,59,15,59)
except RuntimeError as e:assert 'zero optimizer' in str(e)
else:raise AssertionError('Missing optimizer updates were not detected')
print('PASS 15 actual steps per epoch including final partial batch; zero-step guard aborts')
