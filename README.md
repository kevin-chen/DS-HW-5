# DS HW 5

NYU HPC Nvidia Clusters

## Commands
- On every connection to cluster run:
```
source setup
```

## Example Code
- Vector Addition
  1. compile program (this creates an executable)
      ```
      nvcc -o VA-GPU-11 VA-GPU-11.cu
      ```
  2. start x11 server (wait for resources)
  3. run executable by scheduling the job
      ```
      srun --x11 --nodes=1 --gres=gpu:1 --pty VA-GPU-11
      ```

## Setup X11 Server
1. Download XQuartz on mac (or nothing on Vital)
2. ssh -Y greene-solo (kc3585@greene.hpc.nyu.edu)
3. (Don't do this) Start interactive session with X11 forwarding enabled
```
srun --x11 --pty /bin/bash
```
