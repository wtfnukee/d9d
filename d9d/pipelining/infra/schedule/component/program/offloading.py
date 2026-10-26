import bisect
import dataclasses
from collections import defaultdict

from ..runtime.action import (
    ActionBase,
    ActionWorkType,
    ActivationOffloadAction,
    ActivationReleaseAction,
    ActivationReloadAction,
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ComposeAction,
    ForwardComputeAction,
    ForwardSendAction,
)

# One compute action hides the copy to the host and another one hides the copy back. With fewer, the
# activations would come back almost as soon as they left, and the round trip would free nothing.
_MIN_COMPUTE_ACTIONS_BETWEEN = 2


@dataclasses.dataclass(slots=True)
class _Lifetime:
    """The positions of the actions that create and use the activations of one microbatch on one stage.

    Attributes:
        forward: The position of the forward pass.
        backwards: The positions of the backward passes that read the saved activations, in program order.
    """

    forward: int
    backwards: list[int] = dataclasses.field(default_factory=list)


def _collect_lifetimes(actions: list[ActionBase]) -> dict[tuple[int, int], _Lifetime]:
    lifetimes: dict[tuple[int, int], _Lifetime] = {}

    for position, action in enumerate(actions):
        for sub_action in action.sub_actions:
            match sub_action:
                case ForwardComputeAction():
                    lifetimes[sub_action.stage_idx, sub_action.microbatch_idx] = _Lifetime(forward=position)
                case BackwardFullInputComputeAction() | BackwardWeightComputeAction():
                    lifetime = lifetimes.get((sub_action.stage_idx, sub_action.microbatch_idx))
                    if lifetime is None:
                        raise ValueError(f"The program runs ({sub_action}) before its forward pass.")
                    lifetime.backwards.append(position)
                case _:
                    pass

    return lifetimes


def _with_recording(action: ActionBase, recorded: set[tuple[int, int]]) -> ActionBase:
    match action:
        case ForwardComputeAction() if (action.stage_idx, action.microbatch_idx) in recorded:
            return dataclasses.replace(action, record_activations=True)
        case ComposeAction():
            return ComposeAction(tuple(_with_recording(sub_action, recorded) for sub_action in action.actions))
        case _:
            return action


def _add_offload_ops_for_rank(actions: list[ActionBase]) -> list[ActionBase]:
    compute_positions = [
        position for position, action in enumerate(actions) if action.work_type is ActionWorkType.compute
    ]

    insert_before: dict[int, list[ActionBase]] = defaultdict(list)
    insert_after: dict[int, list[ActionBase]] = defaultdict(list)
    recorded: set[tuple[int, int]] = set()

    for (stage_idx, microbatch_idx), lifetime in sorted(_collect_lifetimes(actions).items()):
        if not lifetime.backwards:
            continue

        # The compute positions strictly between the forward pass and the first backward pass.
        first = bisect.bisect_right(compute_positions, lifetime.forward)
        end = bisect.bisect_left(compute_positions, lifetime.backwards[0])
        if end - first < _MIN_COMPUTE_ACTIONS_BETWEEN:
            continue

        # Start the copy to the host after the send, so the next stage gets its inputs first.
        offload_after = lifetime.forward
        while offload_after + 1 < len(actions) and actions[offload_after + 1] == ForwardSendAction(
            stage_idx, microbatch_idx
        ):
            offload_after += 1

        recorded.add((stage_idx, microbatch_idx))
        insert_after[offload_after].append(ActivationOffloadAction(stage_idx, microbatch_idx))
        # The latest slot that still leaves one compute action to hide the copy back to the device.
        insert_before[compute_positions[end - 1]].append(ActivationReloadAction(stage_idx, microbatch_idx))
        insert_after[lifetime.backwards[-1]].append(ActivationReleaseAction(stage_idx, microbatch_idx))

    result: list[ActionBase] = []
    for position, action in enumerate(actions):
        result.extend(insert_before[position])
        result.append(_with_recording(action, recorded))
        result.extend(insert_after[position])
    return result


def add_activation_offload_ops(program: dict[int, list[ActionBase]]) -> dict[int, list[ActionBase]]:
    """Injects activation offload, reload and release actions into a composed program.

    For each microbatch on each stage, the pass offloads the saved activations after the forward pass and
    reloads them before the backward pass. It sets ``record_activations`` on the forward passes it offloads. It
    skips microbatches with fewer than two compute actions between the forward pass and the first backward pass,
    because the copies would have nothing to hide behind. The release comes after the last backward pass, which
    is the weight pass under a split backward. The pass never places an action inside a ``ComposeAction``. It
    must run after ``add_communication_ops``.

    Args:
        program: The per-rank action lists, including the communication actions.

    Returns:
        A new program with the offload actions added. Programs without backward passes are unchanged.

    Raises:
        ValueError: If a backward action of a microbatch comes before its forward pass.
    """
    return {rank: _add_offload_ops_for_rank(actions) for rank, actions in program.items()}
