import pytest
import torch
import torch.nn.functional as F
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.infra.schedule.component.runtime import PipelineScheduleExecutor
from d9d.pipelining.infra.stage import ActivationOffloader, PipelineStage
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

# The CUDA caching allocator rounds every allocation up to this size.
_MIN_BLOCK_BYTES = 512


class _Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(4 * hidden, hidden)
        self.scale = nn.Buffer(torch.ones(4 * hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)) * self.scale)


class _SaveViews(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        flat = x.reshape(-1)
        ctx.save_for_backward(flat[: flat.numel() // 2], flat[flat.numel() // 2 :], x.t())
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        first, second, transposed = ctx.saved_tensors
        assert first.numel() + second.numel() == transposed.numel()
        return grad * 2


def _build(hidden: int = 64) -> tuple[_Model, torch.Tensor]:
    torch.manual_seed(0)
    model = _Model(hidden).cuda()
    x = torch.randn(32, hidden, device="cuda", requires_grad=True)
    return model, x


def _grads(model: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]


def _reference_grads(hidden: int = 64) -> list[torch.Tensor]:
    model, x = _build(hidden)
    model(x).sum().backward()
    return _grads(model, x)


def _forward_offloaded(offloader: ActivationOffloader, model: nn.Module, x: torch.Tensor, microbatch: int = 0):
    with offloader.recording(microbatch):
        out = model(x)
    offloader.keep_resident(microbatch, [x, out])
    return out


@pytest.mark.local
def test_round_trip_matches_reference_gradients():
    model, x = _build()
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    out = _forward_offloaded(offloader, model, x)
    offloader.offload(0)
    offloader.reload(0)
    out.sum().backward()
    offloader.release(0)

    for actual, expected in zip(_grads(model, x), _reference_grads(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert offloader.num_live_records == 0


@pytest.mark.local
def test_offload_frees_device_memory():
    model, x = _build(hidden=1024)
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    x_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    before_forward = torch.cuda.memory_allocated() - _MIN_BLOCK_BYTES
    out = _forward_offloaded(offloader, model, x)
    torch.cuda.synchronize()
    after_forward = torch.cuda.memory_allocated()

    offloader.offload(0)
    torch.cuda.synchronize()
    # Blocks freed under record_stream() return to the pool when the next allocation processes their events.
    del x_probe
    x_probe = torch.empty(1, device="cuda")
    after_offload = torch.cuda.memory_allocated()

    # Only the output and the probe stay: the saved activations of the hidden layer left the device.
    assert after_offload - before_forward <= out.untyped_storage().nbytes() + _MIN_BLOCK_BYTES
    del x_probe
    assert after_offload < after_forward

    offloader.reload(0)
    out.sum().backward()
    offloader.release(0)


@pytest.mark.local
def test_parameters_and_buffers_stay_on_device():
    model, x = _build()
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    _forward_offloaded(offloader, model, x)

    state_ptrs = {t.untyped_storage().data_ptr() for t in [*model.parameters(), *model.buffers()]}
    captured = offloader._records[0].storages
    assert captured
    assert not any(storage.data_ptr() in state_ptrs for storage in captured if storage is not None)


@pytest.mark.local
def test_views_of_one_storage_share_one_copy():
    x = torch.randn(16, 16, device="cuda", requires_grad=True)
    offloader = ActivationOffloader(nn.Module(), torch.cuda.Stream())

    with offloader.recording(0):
        out = _SaveViews.apply(x * 1)
    offloader.keep_resident(0, [x, out])

    assert len(offloader._records[0].storages) == 1

    offloader.offload(0)
    offloader.reload(0)
    out.sum().backward()
    offloader.release(0)
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))


@pytest.mark.local
def test_saved_tensors_survive_a_second_unpack():
    model, x = _build()
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    out = _forward_offloaded(offloader, model, x)
    offloader.offload(0)
    offloader.reload(0)
    # A split backward unpacks the saved tensors twice: once for the inputs, once for the weights.
    out.sum().backward(retain_graph=True)
    out.sum().backward()
    offloader.release(0)

    for actual, expected in zip(_grads(model, x), _reference_grads(), strict=True):
        torch.testing.assert_close(actual, expected * 2)


@pytest.mark.local
def test_backward_without_reload_raises():
    model, x = _build()
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    out = _forward_offloaded(offloader, model, x)
    offloader.offload(0)

    with pytest.raises(RuntimeError, match="must reload it"):
        out.sum().backward()


@pytest.mark.local
def test_lifecycle_guards():
    model, x = _build()
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    with pytest.raises(ValueError, match="No activations are recorded"):
        offloader.offload(0)

    _forward_offloaded(offloader, model, x)
    with pytest.raises(ValueError, match="already recorded"), offloader.recording(0):
        pass
    with pytest.raises(ValueError, match="needs one of"):
        offloader.reload(0)

    offloader.offload(0)
    with pytest.raises(ValueError, match="needs one of"):
        offloader.offload(0)
    with pytest.raises(ValueError, match="needs one of"):
        offloader.release(0)

    offloader.reload(0)
    offloader.release(0)
    assert offloader.num_live_records == 0


class _CheckpointedModel(nn.Module):
    def __init__(self, hidden: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_Model(hidden) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x


@pytest.mark.local
def test_round_trip_with_activation_checkpointing():
    torch.manual_seed(0)
    reference = _CheckpointedModel(1024, num_layers=4).cuda()
    x_ref = torch.randn(32, 1024, device="cuda", requires_grad=True)
    reference(x_ref).sum().backward()

    torch.manual_seed(0)
    model = _CheckpointedModel(1024, num_layers=4).cuda()
    x = torch.randn(32, 1024, device="cuda", requires_grad=True)
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    x_probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    before_forward = torch.cuda.memory_allocated() - _MIN_BLOCK_BYTES
    out = _forward_offloaded(offloader, model, x)

    # Non-reentrant checkpointing saves each layer input as an ordinary saved tensor, so all of them are recorded.
    assert len(offloader._records[0].storages) == 4
    offloader.offload(0)
    torch.cuda.synchronize()
    del x_probe
    x_probe = torch.empty(1, device="cuda")

    # Only the output and the probe stay: the inputs of the checkpointed layers left the device.
    assert torch.cuda.memory_allocated() - before_forward <= out.untyped_storage().nbytes() + _MIN_BLOCK_BYTES
    del x_probe

    offloader.reload(0)
    out.sum().backward()
    offloader.release(0)

    for actual, expected in zip(_grads(model, x), _grads(reference, x_ref), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.local
def test_inplace_change_of_saved_tensor_raises():
    x = torch.randn(16, 16, device="cuda", requires_grad=True)
    offloader = ActivationOffloader(nn.Module(), torch.cuda.Stream())

    with offloader.recording(0):
        hidden = x * 1
        out = hidden.sigmoid() * hidden
    offloader.keep_resident(0, [x, out])

    with torch.no_grad():
        hidden.add_(1)

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.local
def test_executor_rejects_stages_without_offloader():
    stage = PipelineStage(
        info=PipelineStageInfo(num_stages=1, current_stage=0),
        module=nn.Linear(4, 4),
        group=None,
        stage_to_host_topology={0: 0},
    )

    with pytest.raises(ValueError, match="have no activation offloader"):
        PipelineScheduleExecutor(dist_context=None, stages=[stage], program_builder=None, offload_activations=True)


@pytest.mark.local
def test_narrow_view_copies_only_its_bytes():
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)
    offloader = ActivationOffloader(nn.Module(), torch.cuda.Stream())

    with offloader.recording(0):
        # pow() saves its input: two rows of a 4 MiB storage.
        out = (x * 2)[100:102].pow(2).sum()
    offloader.keep_resident(0, [x, out])

    assert len(offloader._records[0].storages) == 1
    offloader.offload(0)
    assert offloader._records[0].host_arena.numel() == 2 * 256 * 4

    offloader.reload(0)
    out.backward()
    offloader.release(0)

    expected = torch.zeros_like(x)
    expected[100:102] = 8 * x.detach()[100:102]
    torch.testing.assert_close(x.grad, expected)


class _DTensorModel(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.weight = nn.Parameter(distribute_tensor(torch.randn(64, 64, device="cuda") / 8, mesh, [Shard(0)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = DTensor.from_local(x, self.mesh, [Replicate()], run_check=False) @ self.weight.t()
        return (torch.tanh(hidden) * hidden).to_local().sum(dim=-1)


@pytest.mark.distributed
def test_dtensor_activations_are_offloaded(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)["dp_replicate"]

    torch.manual_seed(0)
    reference = _DTensorModel(mesh)
    x_ref = torch.randn(32, 64, device="cuda", requires_grad=True)
    reference(x_ref).sum().backward()

    torch.manual_seed(0)
    model = _DTensorModel(mesh)
    x = torch.randn(32, 64, device="cuda", requires_grad=True)
    offloader = ActivationOffloader(model, torch.cuda.Stream())

    out = _forward_offloaded(offloader, model, x)
    record = offloader._records[0]
    weight_ptr = model.weight.to_local().untyped_storage().data_ptr()
    captured = [storage.data_ptr() for storage in record.storages if storage is not None]
    # The DTensor activations are recorded through their local shards, the DTensor weight is not.
    assert len(captured) > 0
    assert weight_ptr not in captured

    offloader.offload(0)
    assert all(record.storages[i] is None for i in record.arena_bases)
    offloader.reload(0)
    out.sum().backward()
    offloader.release(0)

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(model.weight.grad.to_local(), reference.weight.grad.to_local())
