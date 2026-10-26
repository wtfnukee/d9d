import dataclasses
from collections import defaultdict

import pytest
from d9d.pipelining.infra.schedule.component.program import PipelineProgramBuilder, add_activation_offload_ops
from d9d.pipelining.infra.schedule.component.runtime import (
    ActionBase,
    ActivationOffloadAction,
    ActivationReleaseAction,
    ActivationReloadAction,
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ComposeAction,
    ForwardComputeAction,
    ForwardSendAction,
)
from d9d.pipelining.infra.schedule.component.runtime.action import ActionWorkType
from d9d.pipelining.infra.schedule.program import (
    DualPipeVPipelineProgramBuilder,
    Interleaved1F1BPipelineProgramBuilder,
    LoopedBFSPipelineProgramBuilder,
    ZeroBubbleVPipelineProgramBuilder,
)

_OFFLOAD_TYPES = (ActivationOffloadAction, ActivationReloadAction, ActivationReleaseAction)


def _builders() -> list[tuple[str, PipelineProgramBuilder]]:
    return [
        ("inference", LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=True)),
        ("gpipe", LoopedBFSPipelineProgramBuilder(num_stages_per_rank=1, inference_mode=False)),
        ("looped_bfs_v2", LoopedBFSPipelineProgramBuilder(num_stages_per_rank=2, inference_mode=False)),
        ("1f1b_v1", Interleaved1F1BPipelineProgramBuilder(num_stages_per_rank=1)),
        ("1f1b_v2", Interleaved1F1BPipelineProgramBuilder(num_stages_per_rank=2)),
        ("zb1p_v2", Interleaved1F1BPipelineProgramBuilder(num_stages_per_rank=2, enable_zero_bubble=True)),
        ("zbv", ZeroBubbleVPipelineProgramBuilder()),
        ("dual_pipe_v", DualPipeVPipelineProgramBuilder()),
    ]


def _cases() -> list[tuple[str, PipelineProgramBuilder, int, int]]:
    cases = []
    for name, builder in _builders():
        for pp_size in (2, 4, 8):
            for num_microbatches in (1, 2, 4, 8, 16, 32):
                if isinstance(builder, DualPipeVPipelineProgramBuilder) and num_microbatches < 2 * pp_size:
                    continue
                cases.append((name, builder, pp_size, num_microbatches))
    return cases


def _without_recording(action: ActionBase) -> ActionBase:
    if isinstance(action, ForwardComputeAction):
        return dataclasses.replace(action, record_activations=False)
    if isinstance(action, ComposeAction):
        return ComposeAction(tuple(_without_recording(sub) for sub in action.actions))
    return action


def _positions(actions: list[ActionBase]) -> dict[tuple[type, int, int], list[int]]:
    # Where each kind of action of each (stage, microbatch) runs, looking inside ComposeAction.
    positions = defaultdict(list)
    for i, action in enumerate(actions):
        for sub in action.sub_actions:
            positions[type(sub), sub.stage_idx, sub.microbatch_idx].append(i)
    return positions


def _computes_between(actions: list[ActionBase], start: int, end: int) -> int:
    return sum(1 for i in range(start + 1, end) if actions[i].work_type is ActionWorkType.compute)


@pytest.mark.local
@pytest.mark.parametrize(("name", "builder", "pp_size", "num_microbatches"), _cases())
def test_offload_pass_invariants(name: str, builder: PipelineProgramBuilder, pp_size: int, num_microbatches: int):
    original = builder.compose(num_microbatches=num_microbatches, pp_size=pp_size)
    rewritten = add_activation_offload_ops(original)

    assert rewritten.keys() == original.keys()

    for rank, actions in rewritten.items():
        # The pass only adds actions and sets record_activations: undoing both gives back the original program.
        assert [_without_recording(a) for a in actions if not isinstance(a, _OFFLOAD_TYPES)] == original[rank]
        assert not any(
            isinstance(sub, _OFFLOAD_TYPES) for a in actions if isinstance(a, ComposeAction) for sub in a.actions
        )

        positions = _positions(actions)
        forwards = {sub for a in actions for sub in a.sub_actions if isinstance(sub, ForwardComputeAction)}

        for forward in forwards:
            key = (forward.stage_idx, forward.microbatch_idx)
            fwd = positions[ForwardComputeAction, *key]
            bwds = sorted(
                positions[BackwardFullInputComputeAction, *key] + positions[BackwardWeightComputeAction, *key]
            )
            offloads = positions[ActivationOffloadAction, *key]
            reloads = positions[ActivationReloadAction, *key]
            releases = positions[ActivationReleaseAction, *key]

            assert len(fwd) == 1
            if not bwds or _computes_between(actions, fwd[0], bwds[0]) < 2:
                # Nothing to hide the copies behind: the pass leaves the microbatch on the device.
                assert not forward.record_activations
                assert offloads == reloads == releases == []
                continue

            assert forward.record_activations
            assert len(offloads) == len(reloads) == len(releases) == 1
            offload, reload, release = offloads[0], reloads[0], releases[0]

            assert fwd[0] < offload < reload < bwds[0]
            assert release > bwds[-1]
            # Each copy has a compute action to hide behind.
            assert _computes_between(actions, offload, reload) >= 1
            assert _computes_between(actions, reload, bwds[0]) >= 1
            # The send of the forward outputs goes out before the copy to the host starts.
            for send in positions[ForwardSendAction, *key]:
                assert send < offload


@pytest.mark.local
def test_offload_pass_rejects_backward_before_forward():
    program = {0: [BackwardFullInputComputeAction(0, 0, full_backward=True), ForwardComputeAction(0, 0)]}

    with pytest.raises(ValueError, match="before its forward pass"):
        add_activation_offload_ops(program)
