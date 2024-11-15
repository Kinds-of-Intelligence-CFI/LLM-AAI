from typing import Callable, Literal

import numpy as np
from mlagents_envs.base_env import ActionTuple

"""
Expects two lists of three, the first list is GOODGOAL left,forward,right the second is the same for BADGOAL
"""

action_name = Literal["FORWARDS", "BACKWARDS", "LEFT", "RIGHT", "NOOP"]

action_name_to_action_tuple: dict[
    action_name,
    ActionTuple
] = {
    "FORWARDS": ActionTuple(
        continuous=np.zeros((1, 0)),
        discrete=np.array([[1, 0]], dtype=np.int32),
    ),
    "BACKWARDS": ActionTuple(
        continuous=np.zeros((1, 0)),
        discrete=np.array([[2, 0]], dtype=np.int32),
    ),
    "LEFT": ActionTuple(
        continuous=np.zeros((1, 0)),
        discrete=np.array([[0, 2]], dtype=np.int32),
    ),
    "RIGHT": ActionTuple(
        continuous=np.zeros((1, 0)),
        discrete=np.array([[0, 1]], dtype=np.int32),
    ),
    "NOOP": ActionTuple(
        continuous=np.zeros((1, 0)),
        discrete=np.array([[0, 0]], dtype=np.int32),
    )
}

action_name_to_action_reps: dict[action_name, int] = {
    "FORWARDS": 4,
    "BACKWARDS": 4,
    "LEFT": 3,
    "RIGHT": 3
}

action_index_to_action_repetition_tuple: Callable[[action_name], tuple[ActionTuple, int]] = lambda x: (
    action_name_to_action_tuple[x],
    action_name_to_action_reps[x]
)