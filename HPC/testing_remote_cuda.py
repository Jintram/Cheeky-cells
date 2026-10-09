
################################################################################

# Sanity check for a (remote) conda + pytorch install: is CUDA visible, can we
# compute on it, and can we run a short training loop?
# 
# I copied these commands manually into an interactive node on the HPC.
# (After starting the appropriate conda env and python.)
#
# Written by Claude Opus 5.5

################################################################################

import time
import torch
import torch.nn as nn

################################################################################
# Environment info

print(f'torch {torch.__version__}, built for CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}')
print(f'cuda available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available -- likely a CPU-only pytorch build, or a driver/toolkit mismatch (check `nvidia-smi`).')

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {props.name}, {props.total_memory/1e9:.1f} GB, compute capability {props.major}.{props.minor}')
device = torch.device('cuda:0')

################################################################################
# Basic tensor ops: result on GPU should match CPU

a, b = torch.randn(2000, 2000), torch.randn(2000, 2000)
c_gpu = (a.to(device) @ b.to(device)).cpu()
max_diff = (c_gpu - a @ b).abs().max().item()
print(f'matmul GPU vs CPU max abs diff: {max_diff:.2e}')
assert max_diff < 1e-1, 'GPU matmul result deviates from CPU'

################################################################################
# Rudimentary train loop on synthetic data; loss should drop
# Uses the repo's UNet if the package is installed, otherwise a tiny conv net.

print('model: tiny conv net')
model = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 2, 3, padding=1))
model = model.to(device)

# Task: segment pixels brighter than 0 in a smoothed noise image
X = nn.functional.avg_pool2d(torch.randn(16, 1, 136, 136), 9, stride=1).to(device)  # -> 128x128
Y = (X[:, 0] > 0).long()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

torch.cuda.synchronize(); t0 = time.time()
for step in range(50):
    loss = loss_fn(model(X), Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == 0: first_loss = loss.item()
    if step % 10 == 0 or step == 49: print(f'  step {step:3d}  loss {loss.item():.4f}')
torch.cuda.synchronize()

print(f'50 steps in {time.time()-t0:.1f}s, peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
assert loss.item() < first_loss, 'loss did not decrease'
print('ALL CHECKS PASSED')
