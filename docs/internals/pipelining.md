# Pipelining Internals

## About

This page explains the internals of the `d9d.pipelining` module. It is for developers who want to add new stage layouts or schedules, or change the execution engine.

!!! warning "Internal API"
    If you use the standard d9d training loop, you do not need to call this package. d9d sets up pipelining from its configuration. This page is for users who extend d9d.

## Architecture

### The Idea

d9d separates the **schedule structure** from the **runtime execution**.

1.  You write a builder (e.g. `1F1B`, `DualPipe`) that produces a list of logical actions per rank (e.g. `Forward(Stage=0, MB=1)`, `Backward(Stage=0, MB=0)`). You can let d9d add `Send`/`Recv` actions to your compute-only schedule. It derives them from data dependencies and orders them so the schedule does not deadlock.
2.  A simple virtual machine iterates over the action list and executes each action.

With this split, a research schedule such as Zero Bubble or DualPipeV is a list of actions. You do not write a state machine or recursive calls.

### Core Components

#### PipelineStage (`infra/stage/stage.py`)

Wraps a user `nn.Module`. It does not decide *when* to run. It gives the actions and the executor the atomic operations of a pipeline stage, such as the forward and backward passes.

It consists of:

*   **Computation handlers**:
    *   `ForwardComputeHandler`: Runs the forward pass and caches inputs and outputs for the backward pass.
    *   `BackwardComputeHandler`: Runs the backward pass. It can split the backward pass into `backward_input` (dI) and `backward_weight` (dW) for Zero Bubble schedules.
*   **Communication handlers**: Own the P2P buffers for the forward and backward passes. Buffer sizes are inferred per microbatch, so the microbatches in a pack can differ in shape.

#### Actions (`infra/schedule/component/runtime/action.py`)

The atomic instructions for the pipeline virtual machine.

*   `ForwardComputeAction`: Runs the forward pass for a microbatch.
*   `BackwardFullInputComputeAction`: Runs the backward pass. It computes gradients for the inputs only, or for the inputs and the weights.
*   `BackwardWeightComputeAction`: Computes the deferred weight gradients (used in Zero Bubble schedules).
*   `ForwardSendAction` / `ForwardReceiveAction` / `BackwardSendAction` / `BackwardReceiveAction`: Network I/O.
*   `ActivationOffloadAction` / `ActivationReloadAction` / `ActivationReleaseAction`: Move the saved activations of a microbatch to host memory, back to the device, and drop them after the last backward pass (see [Activation Offloading](#activation-offloading)).
*   `ComposeAction`: Groups several actions into one. Schedules such as DualPipeV use it to pair a forward action and a backward action for overlap. The executor currently runs the grouped actions in order.

Actions are declarative and immutable.

#### Programs

A program is a `dict[int, list[ActionBase]]`: it maps each rank to its sequential list of actions.

A program depends only on the microbatch count and the fixed pipeline topology. The `PipelineProgramCache` composes a program once per microbatch count and reuses it across steps. For each program, it keeps the action list of this rank and whether the program contains backward work.

#### Executor (`infra/schedule/component/runtime/executor.py`)

The `PipelineScheduleExecutor` is the runtime engine.

Each step, it receives a **pack** (a sequence of ready per-microbatch inputs) and a callback for the outputs of each microbatch. Then it:

1.  Gets the program for the pack length from the `PipelineProgramCache`.
2.  Configures the stage buffers for the pack. It sizes the P2P buffers of each microbatch separately, so microbatches can differ in shape. It skips this step when the shapes match the previous step.
3.  Iterates over the action list of this rank and applies each action.
4.  Waits for all pending sends to complete.

### Activation Offloading

The `add_activation_offload_ops` pass rewrites a composed program, after `add_communication_ops`. For each microbatch of each stage on a rank, it adds an offload after the forward pass and its send, a reload before the last compute action ahead of the first backward pass, and a release after the last backward pass. It skips a microbatch with fewer than two compute actions between the forward and the first backward pass, and sets `record_activations` on the forward passes it offloads. `PipelineProgramCache` runs the pass once per microbatch count.

At run time, each stage owns an `ActivationOffloader`. `ForwardComputeHandler` runs the module inside `ActivationOffloader.recording()`, a `torch.autograd.graph.saved_tensors_hooks` scope. The pack hook keeps parameters, buffers and their views on the device and records the storages of the other saved tensors, using the local shard of a `DTensor`. Views of one storage share one record, which covers only the bytes they read. The offload copies these byte ranges into slices of one pinned host arena on a copy stream that all stages of the rank share. The reload copies the arena back to one device buffer in a single transfer, and the unpack hook rebuilds each saved tensor as a view into it, wrapping it back into a `DTensor` where needed. See [DEP-0011](https://github.com/d9d-project/d9d/blob/main/deps/0011-dep-pipeline-activation-offload.md).

### Comparison with PyTorch

The d9d pipelining implementation borrows concepts from the `torch.distributed.pipelining` API, such as the Zero Bubble implementation. It restructures the code for clarity, type safety and modularity.

The main differences are a strict separation of concerns and composition over inheritance:

1.  **Decomposed stage logic**:
    *   **PyTorch**: A single `_PipelineStageBase` class manages P2P buffer allocation, gradient accumulation state, and forward/backward execution.
    *   **d9d**: The `PipelineStage` class is a thin orchestrator. It delegates the work to dedicated handlers.

2.  **Polymorphic actions vs. enumeration**:
    *   **PyTorch**: Represents schedule instructions with one generic `_Action` NamedTuple and an Enum (`_ComputationType.FORWARD`, `_ComputationType.SEND_F`, etc.).
    *   **d9d**: Uses a class hierarchy for actions (`ForwardComputeAction`, `ForwardSendAction`, `ComposeAction`). Each action implements `apply`, so the executor does not branch on an enum. Compiler passes such as `add_communication_ops` use `match`/`case` on the action classes. Each action carries its own fields (e.g., the `full_backward` flag), and the type checker covers them.

3.  **Builders vs. schedule classes**:
    *   **PyTorch**: Often couples the schedule definition with the runtime object. For example, the `Schedule1F1B` class both generates the order and executes it.
    *   **d9d**: Separates the **program builder**, which generates the actions, from the **executor**, which runs them. You can inspect a schedule before execution, or swap the scheduling algorithm without changing the executor.

## Building Custom Schedules

To build a new schedule, implement a `PipelineProgramBuilder`.

### Implement the Builder

```python
from d9d.pipelining.infra.schedule.component.program import (
    PipelineProgramBuilder,
    ScheduleStyle,
    add_communication_ops,
    build_stage_to_host_rank_topology,
)
from d9d.pipelining.infra.schedule.component.runtime import (
    ActionBase,
    ForwardComputeAction,
)


class MyScheduleBuilder(PipelineProgramBuilder):
    def __init__(self, stages_per_rank: int):
        self._stages_per_rank = stages_per_rank

    @property
    def num_stages_per_rank(self) -> int:
        return self._stages_per_rank

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.loop

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        num_stages = self._stages_per_rank * pp_size

        # Map logical stages to ranks.
        stage_to_rank = build_stage_to_host_rank_topology(
            pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.loop
        )

        actions: dict[int, list[ActionBase]] = {rank: [] for rank in range(pp_size)}

        # 1. Generate the compute schedule.
        for rank in range(pp_size):
            # ... custom logic that decides the order of forward and backward actions ...
            actions[rank].append(ForwardComputeAction(stage_idx=..., microbatch_idx=...))

        # 2. Add the communication actions derived from the data dependencies between stages.
        return add_communication_ops(
            compute_actions=actions, stage_to_rank=stage_to_rank, num_stages=num_stages
        )
```

### Register the Builder

1.  Add a config class for your schedule to `factory/config.py`, and add it to the `AnyPipelineScheduleConfig` union.
2.  In `factory/registry.py`, register a function that builds your builder from the config with `@PIPELINE_PROGRAM_REGISTRY.register_program(MyScheduleConfig)`.

## API Reference

::: d9d.pipelining.infra.stage

::: d9d.pipelining.infra.schedule.component.runtime

::: d9d.pipelining.infra.schedule.component.program

::: d9d.pipelining.infra.schedule.program

::: d9d.pipelining.training
