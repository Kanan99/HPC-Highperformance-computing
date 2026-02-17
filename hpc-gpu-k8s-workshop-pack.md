# Workshop Pack: HPC + GPUs + Kubernetes + PyTorch Multi‑GPU

**Format:** lecture notes + concept cards + labs (copy/paste YAML + Python).  
**Goal:** by the end you can (1) run GPU workloads on Kubernetes, (2) scale PyTorch from 1 GPU → multi‑GPU → multi‑node, and (3) understand what makes HPC “HPC” (MPI/RDMA/topology) and where Kubernetes fits.

**Suggested duration:** 1–2 days (pick labs that match your time).

---

## 0) Pre‑reqs (what you need ready)

### Cluster / machine requirements
- Kubernetes cluster with at least 1 node that has an NVIDIA GPU (or multiple GPUs for multi‑GPU labs).
- GPU node has:
  - NVIDIA driver installed
  - NVIDIA container runtime/toolkit installed
  - NVIDIA Kubernetes device plugin deployed (so `nvidia.com/gpu` is schedulable)
- Optional for advanced labs:
  - NVIDIA GPU Operator (to enable time‑slicing config easily)
  - RDMA-capable NICs + RDMA device plugin (for multi‑node perf)

**References**
- Kubernetes “Schedule GPUs” task (resource rules and sample Pod). citeturn0search0  
- Kubernetes Device Plugin framework (how hardware becomes schedulable). citeturn0search10  
- NVIDIA Kubernetes device plugin repo. citeturn0search4  
- NVIDIA GPU Operator “Time‑Slicing GPUs” (oversubscription concept). citeturn0search1  

---

## 1) Concept cards (printable mental models)

### HPC vs “cloud batch”
**HPC** usually means:
- **tightly coupled** parallel jobs (processes start together, communicate frequently)
- **MPI** is common
- performance is often limited by network latency/bandwidth and topology

A classic HPC scheduler is **Slurm**, designed for batch/queued jobs with strong allocation semantics and lots of plugins. citeturn2search21turn2search1

### FLOPS (why people talk about it)
**FLOPS** = floating point operations per second. Useful for rough compute comparison, but real performance also depends on memory bandwidth and communication. citeturn1search2

### RDMA (why multi‑node GPU training cares)
**RDMA** allows direct memory access between machines without involving the OS, enabling low-latency, high-throughput networking. citeturn1search3

### Kubernetes core objects (minimum to start)
- **Pod**: smallest deployable unit (one or more containers that run together)
- **Deployment**: keeps a set of Pods running and does rolling updates
- **Job**: runs a task to completion (batch)

Kubernetes is a platform for managing containerized workloads and services with declarative configuration and automation. citeturn0search0turn0search10turn8search0  

---

## 2) Module A — Kubernetes intro for GPU/HPC folks

### Lecture notes
Kubernetes schedules **Pods** onto **Nodes**. The scheduler filters feasible nodes then scores them. citeturn4view2  
For HPC/batch, vanilla K8s gives you Jobs, but queueing and “gang scheduling” often need add-ons (we’ll cover Kueue/Volcano later). citeturn2search3turn2search10turn2search18

### Lab A1 — First Pod (no GPU)
**Objective:** learn “kubectl apply / logs / describe”.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-pod
spec:
  restartPolicy: Never
  containers:
  - name: hello
    image: busybox:1.36
    command: ["sh", "-c", "echo hello from k8s && sleep 5"]
```

**Commands**
```bash
kubectl apply -f hello-pod.yaml
kubectl logs pod/hello-pod
kubectl describe pod/hello-pod
```

---

## 3) Module B — GPU scheduling on Kubernetes

### Lecture notes
Kubernetes uses **device plugins** to advertise GPUs as schedulable resources. citeturn0search10  
For NVIDIA GPUs, you usually request `nvidia.com/gpu` in Pod resources. Kubernetes requires GPU `limits` and `requests` (if set) to match. citeturn0search0

### Lab B1 — “nvidia-smi in a Pod”
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi
spec:
  restartPolicy: Never
  containers:
  - name: nvidia-smi
    image: nvidia/cuda:12.3.2-base-ubuntu22.04
    command: ["bash", "-lc", "nvidia-smi && sleep 5"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

```bash
kubectl apply -f nvidia-smi.yaml
kubectl logs pod/nvidia-smi
```

### Lab B2 — Node labeling for “different GPU types”
K8s recommends labeling nodes so workloads can target specific GPU models. citeturn0search0

```bash
kubectl label node <node-name> accelerator=nvidia-h100
```

```yaml
# Add to spec:
nodeSelector:
  accelerator: nvidia-h100
```

---

## 4) Module C — Single node, multi‑GPU training (PyTorch DDP)

### Lecture notes
**DistributedDataParallel (DDP)** replicates the model on each GPU/process and synchronizes gradients.  
DDP does **not** shard your input automatically; you must shard data (e.g., with `DistributedSampler`). citeturn0search8

**torchrun** is the recommended launcher for single-node and multi-node distributed training. citeturn0search2

### Lab C1 — Minimal DDP training script (single node)
Create `train_ddp.py`:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

def setup():
    # torchrun sets these environment variables
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)

def main():
    local_rank = setup()

    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST("/data", train=True, download=True, transform=transform)

    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=128, sampler=sampler, num_workers=2, pin_memory=True)

    model = SmallCNN().cuda()
    model = DDP(model, device_ids=[local_rank])
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for x, y in dl:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        if dist.get_rank() == 0:
            print(f"epoch {epoch} done")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run on a GPU box (outside K8s first):
```bash
torchrun --standalone --nproc-per-node=4 train_ddp.py
```

**Why this matters:** this is the same pattern you’ll run inside Kubernetes once GPUs are allocated.

**References**
- DDP tutorial + API notes about data sharding responsibility. citeturn0search6turn0search8  
- torchrun docs. citeturn0search2  

### Lab C2 — Run the same DDP inside Kubernetes (single node)
Use a `Job` (batch) and request multiple GPUs:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: mnist-ddp
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: <your-image-with-train_ddp.py>
        command: ["bash", "-lc", "torchrun --standalone --nproc-per-node=4 train_ddp.py"]
        resources:
          limits:
            nvidia.com/gpu: 4
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        emptyDir: {}
```

---

## 5) Module D — Memory scaling: FSDP (parameter/optimizer sharding)

### Lecture notes
**FSDP** shards parameters/gradients/optimizer state across ranks to reduce memory footprint versus DDP replicas. citeturn0search9turn0search3  

### Lab D1 — Wrap model with FSDP (conceptual sample)
This is a minimal sketch (you’ll typically use auto-wrapping policies in practice).

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# after dist.init_process_group() and setting device
model = MyBigModel().cuda()
model = FSDP(model)  # shard params across ranks
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

**References**
- FSDP getting started + API. citeturn0search7turn0search3  

---

## 6) Module E — Multi‑node distributed training: torchrun and/or MPI Operator

### Option 1: Multi‑node torchrun (PyTorch-native)
torchrun supports multi-node and elastic rendezvous modes. citeturn0search2turn0search12

**Workshop idea:** run 2 nodes × 4 GPUs (total 8 ranks) and compare step time vs 1 node.

### Option 2: MPI style on Kubernetes (Kubeflow MPIJob)
Kubernetes + MPI integration is discussed by StackHPC, and Kubeflow provides MPIJob for MPI/allreduce style runs. citeturn1search0turn7search0

**MPIJob skeleton (conceptual):**
```yaml
apiVersion: kubeflow.org/v1
kind: MPIJob
metadata:
  name: mpi-train
spec:
  slotsPerWorker: 4
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: launcher
            image: <your-image>
            command: ["bash", "-lc", "mpirun -np 8 python train_ddp.py"]
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: worker
            image: <your-image>
            resources:
              limits:
                nvidia.com/gpu: 4
```

---

## 7) Module F — GPU sharing: MIG vs time-slicing (utilization vs isolation)

### Lecture notes
- **Time-slicing** (software oversubscription) allows multiple workloads to share a GPU by interleaving execution. NVIDIA GPU Operator documents time-slicing as oversubscription through device plugin options. citeturn0search1  
- **MIG** (hardware partitioning) offers stronger isolation by splitting a physical GPU into isolated instances (requires compatible GPUs). NVIDIA documents MIG support for Kubernetes. citeturn0search11  

### Lab F1 — Discussion exercise (choose a policy)
Given 3 workload types:
1) short inference bursts
2) long training runs
3) interactive notebooks

Decide: full GPU, MIG, or time-slicing for each. Explain tradeoffs:
- isolation/QoS
- utilization
- scheduling complexity

(If you want hands-on, we can add a concrete GPU Operator values/config example tailored to your cluster.)

---

## 8) Module G — RDMA on Kubernetes (for HPC-style performance)

### Lecture notes
Multi‑node GPU training can bottleneck on communication. RDMA helps by providing low-latency, high throughput transfers. citeturn1search3  

The Mellanox RDMA shared device plugin runs as a DaemonSet and exposes RDMA resources to pods. citeturn2search0

### Lab G1 — Request RDMA resources (conceptual)
When RDMA resources are exposed, you may request them similarly to GPUs (exact resource names depend on plugin/config).  
Example reference for exposed resource naming in managed clusters: citeturn2search20

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
    rdma/fabric0: 1
```

---

## 9) Module H — Batch queueing and gang scheduling on Kubernetes

### Why you need this for HPC/AI
If a job needs 8 GPUs across multiple nodes, you often want “all or nothing” scheduling (gang scheduling) to avoid partial starts.

Kubernetes has added work in this area (gang scheduling concepts/plugins), and there are established add-ons:
- **Kueue**: Kubernetes-native job queueing system for batch/HPC/AI. citeturn2search3turn2search7  
- **Volcano**: batch scheduler / unified scheduling and is historically linked to kube-batch. citeturn2search10turn1search0  
- Kubernetes also documents gang scheduling behavior with a scheduler plugin. citeturn2search18  

### Lab H1 — Kueue: run a queued Job (starter)
Follow Kueue docs for running a Kubernetes Job with Kueue enabled. citeturn2search19  
(We can add a complete YAML bundle if you tell me your K8s version and whether you want namespace quotas/cohorts.)

---

## 10) Where Kubernetes fits (and where Slurm fits)

**Kubernetes strengths:** ecosystem, portability, mixed workloads, “platform” semantics. citeturn8search0  
**HPC caution:** Kubernetes was built around loosely-coupled services; HPC tightly-coupled jobs may need extra scheduling/networking pieces (IBM/HPCwire discusses pros/cons). citeturn1search1  
**Slurm strengths:** classic batch scheduling, accounting, reservations/fairshare/topology via plugins. citeturn2search21turn2search1

---

## Suggested workshop flow (pick 1-day or 2-day)

### 1-day “core”
1) Module A (K8s intro) + Lab A1  
2) Module B (GPU scheduling) + Lab B1  
3) Module C (DDP) + Lab C1  
4) Module C2 (DDP in K8s)  
5) Module H (queueing concept) quick demo with Kueue/Volcano

### 2-day “full”
Day 1: A → B → C (plus C2)  
Day 2: D (FSDP) → E (multi-node) → F (sharing) → G (RDMA) → H (queueing/gang)

---

## Reading list (aligned to your original links)

- StackHPC: MPI + Kubernetes landscape and scheduler discussion. citeturn1search0  
- HPCwire/IBM: Pros and cons of Kubernetes for HPC. citeturn1search1  
- FLOPS background. citeturn1search2  
- RDMA background. citeturn1search3  
- Kubernetes GPU scheduling + device plugins. citeturn0search0turn0search10  
- Mellanox RDMA device plugin. citeturn2search0  
- Slurm overview / quickstart. citeturn2search1turn2search21  
- PyTorch DDP / torchrun / FSDP tutorials. citeturn0search6turn0search2turn0search7turn0search9  

---

*Last updated: 2026-02-17 (Asia/Baku).*
