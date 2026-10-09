

### Minimal pytorch environment test

On the server, the following command will spin up a node 
interactively:

```sh
srun --partition=capacity --gpus=1 --cpus-per-task=1 --time=00:30:00 --pty bash
```

I performed my installation on this interactive node 
(exit the interactive node later by `exit`).

To get information regarding the required CUDA version, use:
```sh
nvidia-smi
```

This returns: 

```
Fri Oct  9 13:12:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.57.04              KMD Version: 610.57.04     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:04:00.0 Off |                    0 |
| N/A   39C    P8             15W /   72W |       0MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

So `13.3` is the version we want.
To install pytorch using conda, use:

```sh
CONDA_OVERRIDE_CUDA=13.3 conda create -n test2 -c conda-forge python "pytorch=*=cuda*" torchvision
```

And activate the environment using:

```sh
conda activate test2
```


