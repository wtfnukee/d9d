---
DEP: 0011
Title: Pipeline Activation Offloading
Author: Egor Konovalov @wtfnukee
Status: Accepted
Type: Feature
Created: 2026-10-25
---

# Pipeline Activation Offloading

## Abstract

In a pipeline schedule, a rank runs several forward passes before the matching backward passes arrive. The activations that autograd saves for these microbatches stay in GPU memory for the whole gap. The first ranks wait longest, so their activation memory decides how large a model fits.

This DEP adds activation offloading to the pipelining engine, as in the [GLM-5 technical report](https://arxiv.org/abs/2602.15763) and [PipeOffload](https://arxiv.org/abs/2503.01328). After the forward pass of a microbatch, the stage copies its saved activations to pinned host memory and frees them on the device. Shortly before the backward pass, it copies them back. The copies run on a separate CUDA stream, so they overlap with compute.

A compiler pass adds three actions to the composed program: offload, reload and release. A runtime component records the saved tensors with `torch.autograd.graph.saved_tensors_hooks`. Schedules and models do not change. One flag turns the feature on: `pipelining.offload_activations`.

## Motivation

`ForwardComputeHandler` keeps the stage inputs and outputs of each microbatch until its backward pass. These tensors are small. The autograd graph behind the outputs holds the large ones: the intermediate activations of every layer. Nothing in the pipelining engine can reach them today.

How long the graph lives depends on the schedule. Under 1F1B, rank 0 runs `pp_size` forward passes before its first backward pass, while the last rank runs the backward pass right after the forward pass. Interleaved and Zero Bubble schedules stretch the gap further on the first ranks. The only lever today is activation checkpointing in the model, which recomputes every layer whether or not the schedule needs the memory.

## Design Proposal

### Recording

`ForwardComputeHandler.run()` is the only place that calls the stage module. When the program offloads a microbatch, it wraps the call in `ActivationOffloader.recording()`:

```python
with offloader.recording(microbatch_index):
    output = self._module(inputs, shared)
offloader.keep_resident(microbatch_index, pytree.tree_leaves((inputs, output)))
```

The pack hook copies nothing. It decides whether the saved tensor stays on the device. If not, it records the storage of the tensor and returns a handle with the dtype, shape, stride and storage offset. The following tensors stay on the device:

*   **Parameters and their views**: `F.linear` saves `weight.t()`, a plain tensor whose view base is the parameter. Under FSDP2, the unsharded weight is an `nn.Parameter` too.
*   **Buffers and the local shards of `DTensor` parameters**: They are module state and stay on the device for the whole step. The offloader compares storages, so views of them stay too.
*   **Tensors outside the GPU**.
*   **The inputs, shared inputs and outputs of the stage**: The forward cache keeps the inputs and outputs until the backward pass, and the executor keeps the shared inputs for the whole step, so copying them frees nothing. `keep_resident()` marks their storages after the forward pass.

A saved `DTensor` that is not module state is offloaded through its local shard. The handle keeps its mesh, placements, shape and stride, and the unpack hook wraps the rebuilt shard with `DTensor.from_local()`.

Views of one storage, such as QKV splits, transposes and reshapes, share one record. Each storage is copied once, and only the bytes its saved views read: a saved slice of a large buffer costs the slice.

Activation checkpointing works with the offloader. The non-reentrant `torch.utils.checkpoint` saves the inputs of each checkpointed layer as ordinary saved tensors, so the pack hook records them like any other activation. With checkpointing on, the offloader moves the layer inputs, which are most of what the stage still holds.

### Copies

`offload()` puts the recorded storages into one pinned host arena. Each storage gets a slice aligned to 256 bytes. The copy stream waits for the compute stream, copies each storage into its slice and calls `record_stream()`. The offloader then drops its references, and the caching allocator reuses the blocks once the copy stream has read them.

`reload()` allocates one device arena of the same size on the compute stream. The copy stream waits for the compute stream, because the allocator may have just handed the block over from earlier work. Then it runs a single host-to-device copy and records an event. The first unpack of the microbatch makes the compute stream wait on that event. Every rebuilt tensor is a view into the device arena. The alignment keeps each storage offset divisible by the element size.

`release()` drops the record after the last backward pass. Pinned host memory comes from the PyTorch caching host allocator, which reuses it across steps.

All stages of a rank share one copy stream, so the copies run in program order.

### Actions

Three actions join the VM, with the new work type `ActionWorkType.offload`:

*   **`ActivationOffloadAction`**: Calls `ActivationOffloader.offload()` through `PipelineStage.activation_offloader`.
*   **`ActivationReloadAction`**: Calls `ActivationOffloader.reload()`.
*   **`ActivationReleaseAction`**: Calls `ActivationOffloader.release()`.

`ForwardComputeAction` gains a `record_activations` field, which the placement pass sets on the forward passes it offloads. The printed program does not show it; the offload actions next to the forward pass do.

### Placement

`add_activation_offload_ops()` rewrites a composed program, including its communication actions. For each microbatch on each stage of a rank, it finds the forward pass, the first backward pass and the last backward pass:

1.  It skips the microbatch if fewer than two compute actions lie between the forward pass and the first backward pass. One compute action hides the copy to the host and another one hides the copy back.
2.  It adds the offload after the forward pass, and after the send of its outputs if the send follows directly.
3.  It adds the reload before the last compute action ahead of the first backward pass.
4.  It adds the release after the last backward pass. Under a split backward, this is the weight pass, which unpacks the saved tensors a second time.

The pass never places an action inside a `ComposeAction`. Programs without a backward pass do not change. `PipelineProgramCache` runs the pass once per microbatch count, next to `compose()`.

### Configuration

```python
class PipeliningConfig(BaseModel):
    schedule: AnyPipelineScheduleConfig = ...
    offload_activations: bool = False
```

`build_schedule()` and `PipelineScheduleExecutor` take `offload_activations`. With the flag set, `build_schedule()` creates one copy stream per rank and an `ActivationOffloader` for each stage. Jobs without pipeline parallelism run each backward pass right after its forward pass, so the flag changes nothing there.

## Usage

```yaml
trainer:
  pipelining:
    schedule:
      schedule: 1f1b
      num_stages_per_rank: 2
      zero_bubble: true
    offload_activations: true
```

## Backward Compatibility

The flag is off by default, and the new arguments of `build_schedule()`, `PipelineScheduleExecutor`, `PipelineStage` and `PipelineStage.forward_one_chunk()` have defaults. `ForwardComputeHandler` and `PipelineProgramCache` are internal and take a new required argument.

## Performance

Measured on 8x A100-SXM4-40GB with NVLink between the GPUs, in bf16. The stage modules hold a Llama-7B-shaped stack: 32 layers in total, hidden size 4096, FFN size 11008, 32 heads, attention through `F.scaled_dot_product_attention`. Each step runs 16 microbatches of 1 sequence of 4096 tokens. Step times are medians over 5 steps.

| Schedule | Checkpointing | Offloading | Step time | Peak memory, busiest rank | Peak memory, rank 0 |
|:--|:--|:--|--:|--:|--:|
| 1F1B | on | off | 2950 ms | 6.35 GiB | 6.16 GiB |
| 1F1B | on | on | 2940-2959 ms | 5.32 GiB | 4.97 GiB |
| 1F1B | off | on | 14.7-22.2 s | 9.40 GiB | 9.06 GiB |
| 1F1B | off | off | out of memory | - | - |
| Zero Bubble, 2 stages per rank | on | off | 4196 ms | 12.35 GiB | 6.28 GiB |
| Zero Bubble, 2 stages per rank | on | on | 4208-4211 ms | 12.13 GiB | 5.87 GiB |
| Zero Bubble, 2 stages per rank | off | on | 12.6-12.9 s | 21.57 GiB | 7.71 GiB |
| Zero Bubble, 2 stages per rank | off | off | 3286 ms | 29.56 GiB | 23.49 GiB |

Pinned copies reach 12.2 GiB/s from one GPU to the host and 11.5 GiB/s back. When all 8 GPUs copy at once, each gets 3-6 GiB/s, because GPUs share their links to the host.

*   **With checkpointing**: Each rank offloads 1.0-1.5 GiB per step, and the copies hide behind compute. Step time stays within 0.3%. Peak memory of the busiest rank drops by 16% under 1F1B. Under Zero Bubble the busiest rank is the last one, which offloads least, so its peak drops by 2% while rank 0 drops by 7%.
*   **Without checkpointing**: Each rank moves 37 GiB per step in each direction, which takes 5-6 s of copying at the shared bandwidth. The copies cannot hide behind a 3-4 s step, so the step is 3-7.5x slower than with checkpointing. The range of 1F1B step times comes from two machines with different host bandwidth.
*   **Copies**: The offloader moves its copies at about 9 GiB/s with checkpointing and about 7 GiB/s without, close to the bandwidth that the link gives each GPU. Merging the copies of a microbatch into fewer transfers would not raise this, so the offloader keeps one copy per storage.

The copies share the host link with any other traffic that crosses it. In this benchmark, pipeline P2P runs over NVLink. On hosts where P2P or expert-parallel RDMA traffic crosses PCIe, measure the step time with offloading on and off.
