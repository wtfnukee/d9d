import dataclasses
import functools
import math
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement

# Each storage gets its own aligned slice of the arena, so every rebuilt tensor keeps a storage offset that its
# element size divides, and kernels keep the alignment that the caching allocator gives a fresh block.
_ARENA_ALIGNMENT_BYTES = 256


def _align(num_bytes: int) -> int:
    return (num_bytes + _ARENA_ALIGNMENT_BYTES - 1) // _ARENA_ALIGNMENT_BYTES * _ARENA_ALIGNMENT_BYTES


def _storage_as_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def _byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    # The bytes of its storage that a strided view reads: from its first element to one past its last one.
    itemsize = tensor.element_size()
    start = int(tensor.storage_offset()) * itemsize
    if tensor.numel() == 0:
        return start, start
    last = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return start, start + (int(last) + 1) * itemsize


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # to_local() would add an autograd node inside the hooks, so read the shard the DTensor stores directly.
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor  # noqa: SLF001 - see above


def _is_parameter_or_view(tensor: torch.Tensor) -> bool:
    # F.linear saves weight.t(): a plain tensor whose view base is the parameter. Under FSDP2 the unsharded
    # weight is an nn.Parameter too.
    base = tensor._base if tensor._base is not None else tensor  # noqa: SLF001 - PyTorch has no public view base
    return isinstance(base, nn.Parameter)


@dataclasses.dataclass(frozen=True, slots=True)
class _DTensorLayout:
    """What turns a rebuilt local shard back into the saved ``DTensor``.

    Attributes:
        mesh: The device mesh of the ``DTensor``.
        placements: The placements of the ``DTensor``.
        shape: The global shape of the ``DTensor``.
        stride: The global stride of the ``DTensor``.
    """

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    shape: torch.Size
    stride: tuple[int, ...]


class _RecordState(StrEnum):
    recording = "recording"
    resident = "resident"
    offloaded = "offloaded"
    reloaded = "reloaded"


@dataclasses.dataclass(frozen=True, slots=True)
class _SavedTensorHandle:
    """The packed form of a saved tensor: where its storage lives and how to view it.

    Attributes:
        microbatch_index: The microbatch whose record holds the storage.
        storage_index: The index of the storage in the record.
        dtype: The dtype of the saved tensor, or of its local shard for a ``DTensor``.
        shape: The shape of the saved tensor, or of its local shard for a ``DTensor``.
        stride: The stride of the saved tensor, or of its local shard for a ``DTensor``.
        storage_offset: The storage offset of the saved tensor, or of its local shard, in elements.
        dtensor: The layout that turns the local shard back into the saved ``DTensor``, or ``None`` for a plain
            tensor.
        source: A weak reference to the saved tensor, alive while other code still holds it.
        version: The version counter of the saved tensor when autograd saved it.
    """

    microbatch_index: int
    storage_index: int
    dtype: torch.dtype
    shape: torch.Size
    stride: tuple[int, ...]
    storage_offset: int
    dtensor: _DTensorLayout | None
    source: weakref.ref[torch.Tensor]
    version: int


@dataclasses.dataclass(slots=True)
class _MicrobatchRecord:
    """The storages captured in the forward pass of one microbatch.

    Attributes:
        state: The position of the record in its lifecycle.
        storages: The captured device storages, indexed by storage index. An offloaded storage is ``None``.
        storage_index_by_ptr: The storage index of each captured device storage, keyed by its data pointer.
        byte_ranges: The ``(start, end)`` bytes of each captured storage that its saved views read.
        resident_indices: The storages that stay on the device, because the stage keeps them anyway.
        arena_bases: For each offloaded storage, the arena offset in bytes where its byte 0 would sit. The arena
            holds only the copied range, so a view at byte ``b`` of the storage sits at ``arena_bases[i] + b``.
        host_arena: The pinned host buffer that holds the offloaded storages.
        device_arena: The device buffer that the offloaded storages are reloaded into.
        reload_event: The event that the reload records on the copy stream.
        awaited_stream_ids: The streams that already wait on ``reload_event``.
    """

    state: _RecordState
    storages: list[torch.UntypedStorage | None] = dataclasses.field(default_factory=list)
    storage_index_by_ptr: dict[int, int] = dataclasses.field(default_factory=dict)
    byte_ranges: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    resident_indices: set[int] = dataclasses.field(default_factory=set)
    arena_bases: dict[int, int] = dataclasses.field(default_factory=dict)
    host_arena: torch.Tensor | None = None
    device_arena: torch.Tensor | None = None
    reload_event: torch.cuda.Event | None = None
    awaited_stream_ids: set[int] = dataclasses.field(default_factory=set)


class ActivationOffloader:
    """Moves the activations that autograd saves in the forward pass of a stage to host memory and back.

    The schedule calls ``offload()`` after the forward pass of a microbatch, ``reload()`` shortly before its
    backward pass and ``release()`` after its last backward pass. Parameters, buffers and the tensors the stage
    keeps for its own backward pass stay on the device.
    """

    def __init__(self, module: nn.Module, copy_stream: torch.cuda.Stream):
        """Constructs the ``ActivationOffloader`` object.

        Args:
            module: The module of the stage. Its buffers stay on the device.
            copy_stream: The CUDA stream that runs the copies between the device and the host.
        """
        self._module = module
        self._copy_stream = copy_stream
        self._records: dict[int, _MicrobatchRecord] = {}
        self._state_ptrs: frozenset[int] = frozenset()

    def _keeps_resident(self, tensor: torch.Tensor, local: torch.Tensor, storage: torch.UntypedStorage) -> bool:
        if not local.is_cuda:
            return True
        # Empty storages all report a null data pointer, so they would alias each other in the record.
        if storage.nbytes() == 0:
            return True
        if _is_parameter_or_view(tensor):
            return True
        # A view of a buffer or of a DTensor parameter shares its storage, so the storage pointer covers both.
        return storage.data_ptr() in self._state_ptrs

    def _require_record(self, microbatch_index: int, *states: _RecordState) -> _MicrobatchRecord:
        record = self._records.get(microbatch_index)
        if record is None:
            raise ValueError(f"No activations are recorded for microbatch_index ({microbatch_index}).")
        if record.state not in states:
            raise ValueError(
                f"The activations of microbatch_index ({microbatch_index}) are {record.state}, "
                f"but this call needs one of ({', '.join(states)})."
            )
        return record

    def _pack(self, microbatch_index: int, tensor: torch.Tensor) -> torch.Tensor | _SavedTensorHandle:
        # A DTensor offloads its local shard and keeps its layout in the handle.
        local = _local_tensor(tensor)
        storage = local.untyped_storage()
        if self._keeps_resident(tensor, local, storage):
            return tensor

        record = self._records[microbatch_index]
        storage_ptr = storage.data_ptr()
        storage_index = record.storage_index_by_ptr.get(storage_ptr)
        start, end = _byte_range(local)
        # Views of one storage (QKV splits, transposes, reshapes) share one copy of the bytes they read.
        if storage_index is None:
            storage_index = len(record.storages)
            record.storages.append(storage)
            record.byte_ranges.append((start, end))
            record.storage_index_by_ptr[storage_ptr] = storage_index
        elif start < end:
            low, high = record.byte_ranges[storage_index]
            record.byte_ranges[storage_index] = (min(low, start), max(high, end)) if low < high else (start, end)

        return _SavedTensorHandle(
            microbatch_index=microbatch_index,
            storage_index=storage_index,
            dtype=local.dtype,
            shape=local.shape,
            stride=local.stride(),
            storage_offset=int(local.storage_offset()),
            dtensor=(
                _DTensorLayout(
                    mesh=tensor.device_mesh,
                    placements=tensor.placements,
                    shape=tensor.shape,
                    stride=tensor.stride(),
                )
                if isinstance(tensor, DTensor)
                else None
            ),
            source=weakref.ref(tensor),
            version=tensor._version,  # noqa: SLF001 - autograd's own in-place check reads the same counter
        )

    def _unpack(self, packed: torch.Tensor | _SavedTensorHandle) -> torch.Tensor:
        if isinstance(packed, torch.Tensor):
            return packed

        record = self._records.get(packed.microbatch_index)
        if record is None:
            raise RuntimeError(
                f"A saved activation of microbatch_index ({packed.microbatch_index}) was used after release()."
            )

        # The handle replaces the saved tensor, so autograd cannot see in-place changes to it. Check the counter
        # while the tensor is alive: a dead tensor has no references left to change it through.
        source = packed.source()
        if source is not None and source._version != packed.version:  # noqa: SLF001 - see _pack()
            raise RuntimeError(
                f"A saved activation of microbatch_index ({packed.microbatch_index}) was modified by an inplace "
                f"operation: its version is ({source._version}), but the forward pass saved version "  # noqa: SLF001 - see _pack()
                f"({packed.version})."
            )

        return self._as_saved(self._rebuild_local(record, packed), packed.dtensor)

    @staticmethod
    def _as_saved(local: torch.Tensor, layout: _DTensorLayout | None) -> torch.Tensor:
        if layout is None:
            return local
        return DTensor.from_local(
            local, layout.mesh, layout.placements, run_check=False, shape=layout.shape, stride=layout.stride
        )

    def _rebuild_local(self, record: _MicrobatchRecord, packed: _SavedTensorHandle) -> torch.Tensor:
        # A view without elements reads no bytes, so the copied range may not cover its offset.
        if math.prod(packed.shape) == 0:
            return torch.empty_strided(packed.shape, packed.stride, dtype=packed.dtype, device="cuda")

        # A storage that is still on the device keeps its slot; offload() empties the slot when it moves it out.
        storage = record.storages[packed.storage_index]
        if storage is not None:
            storage_offset = packed.storage_offset
        elif record.device_arena is not None and record.reload_event is not None:
            stream = torch.cuda.current_stream()
            if stream.stream_id not in record.awaited_stream_ids:
                stream.wait_event(record.reload_event)
                record.awaited_stream_ids.add(stream.stream_id)
            storage = record.device_arena.untyped_storage()
            itemsize = packed.dtype.itemsize
            storage_offset = (record.arena_bases[packed.storage_index] + packed.storage_offset * itemsize) // itemsize
        else:
            raise RuntimeError(
                f"A saved activation of microbatch_index ({packed.microbatch_index}) is {record.state}. "
                "The program must reload it before its backward pass."
            )

        return torch.empty(0, dtype=packed.dtype, device=storage.device).set_(
            storage, storage_offset, packed.shape, packed.stride
        )

    @contextmanager
    def recording(self, microbatch_index: int) -> Iterator[None]:
        """Records the tensors that autograd saves inside the block, for one microbatch.

        Args:
            microbatch_index: The microbatch whose forward pass runs inside the block.

        Raises:
            ValueError: If the activations of this microbatch are already recorded.
        """
        if microbatch_index in self._records:
            raise ValueError(f"The activations of microbatch_index ({microbatch_index}) are already recorded.")

        # Read once per forward pass: sleep and wake move the module state to new storages.
        self._state_ptrs = frozenset(
            _local_tensor(state).untyped_storage().data_ptr()
            for state in [*self._module.buffers(), *self._module.parameters()]
        )
        self._records[microbatch_index] = _MicrobatchRecord(state=_RecordState.recording)

        with torch.autograd.graph.saved_tensors_hooks(functools.partial(self._pack, microbatch_index), self._unpack):
            yield

        self._records[microbatch_index].state = _RecordState.resident

    def keep_resident(self, microbatch_index: int, tensors: Sequence[torch.Tensor]):
        """Marks captured storages that the stage keeps for its backward pass anyway, so they are not copied.

        Args:
            microbatch_index: The microbatch.
            tensors: The tensors the stage keeps, such as its inputs and outputs.

        Raises:
            ValueError: If the activations of this microbatch are not recorded and resident.
        """
        record = self._require_record(microbatch_index, _RecordState.resident)
        for tensor in tensors:
            storage_index = record.storage_index_by_ptr.get(_local_tensor(tensor).untyped_storage().data_ptr())
            if storage_index is not None:
                record.resident_indices.add(storage_index)

    def offload(self, microbatch_index: int):
        """Starts copying the recorded activations of a microbatch to host memory and drops their device storage.

        Args:
            microbatch_index: The microbatch.

        Raises:
            ValueError: If the activations of this microbatch are not recorded and resident.
        """
        record = self._require_record(microbatch_index, _RecordState.resident)

        to_copy = [
            (storage_index, storage)
            for storage_index, storage in enumerate(record.storages)
            if storage_index not in record.resident_indices and storage is not None
        ]
        copy_ranges: list[tuple[int, int, int]] = []
        total_bytes = 0
        for storage_index, _ in to_copy:
            start, end = record.byte_ranges[storage_index]
            # Rounding the start down keeps every rebuilt view at a storage offset its element size divides.
            start = start // _ARENA_ALIGNMENT_BYTES * _ARENA_ALIGNMENT_BYTES
            copy_ranges.append((total_bytes, start, end))
            record.arena_bases[storage_index] = total_bytes - start
            total_bytes += _align(end - start)

        record.state = _RecordState.offloaded
        if total_bytes == 0:
            return

        # Pinned memory comes from the caching host allocator, which reuses it across steps.
        record.host_arena = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)

        compute_stream = torch.cuda.current_stream()
        # The copies must start after the forward pass that wrote the activations.
        self._copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self._copy_stream):
            for (storage_index, storage), (offset, start, end) in zip(to_copy, copy_ranges, strict=True):
                source = _storage_as_bytes(storage)
                record.host_arena[offset : offset + end - start].copy_(source[start:end], non_blocking=True)
                # The allocator must not reuse the block before the copy stream has read it.
                source.record_stream(self._copy_stream)
                record.storages[storage_index] = None

    def reload(self, microbatch_index: int):
        """Starts copying the offloaded activations of a microbatch back to the device.

        Args:
            microbatch_index: The microbatch.

        Raises:
            ValueError: If the activations of this microbatch are not offloaded.
        """
        record = self._require_record(microbatch_index, _RecordState.offloaded)
        record.state = _RecordState.reloaded
        if record.host_arena is None:
            return

        compute_stream = torch.cuda.current_stream()
        # Allocated on the compute stream, which frees it in release(). The copy stream waits for the work that
        # used the block before, because the allocator may have just handed it over from that work.
        record.device_arena = torch.empty_like(record.host_arena, device=torch.cuda.current_device())
        self._copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self._copy_stream):
            record.device_arena.copy_(record.host_arena, non_blocking=True)
            # release() can drop the arena before any unpack makes the compute stream wait, e.g. when the
            # backward pass prunes every node that saved into it. The allocator must not reuse it before the copy.
            record.device_arena.record_stream(self._copy_stream)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        record.reload_event = event
        record.host_arena = None

    def release(self, microbatch_index: int):
        """Drops all recorded activations of a microbatch after its last backward pass.

        Args:
            microbatch_index: The microbatch.

        Raises:
            ValueError: If the activations of this microbatch are not recorded.
        """
        self._require_record(microbatch_index, _RecordState.resident, _RecordState.reloaded)
        del self._records[microbatch_index]

    @property
    def num_live_records(self) -> int:
        """The number of microbatches whose activations are recorded and not released."""
        return len(self._records)

    def reset(self):
        """Drops the records of all microbatches, e.g. after a step that raised."""
        self._records.clear()
