"""
A language that the LLM can use to specify paths in AAI, designed to be minimal in the number of commands
"""

from typing import Union, Literal, Callable, Any
from mlagents_envs.base_env import ActionTuple
from enum import Enum
from src.definitions.cardinal_directions import action_name_to_action_tuple
from src.definitions.constants import TEXT_COMMAND_DELIMITER,FRAMES_BETWEEN_OBS
import re
from src.definitions.constants import DEGREES_PER_ROTATE
from enum import Enum

# TODO: Bump this if we see LLMs regularly using all obs
MAX_OBS = 4

class YIELD_OBS:
    def __eq__(self, other: Any):
        return isinstance(other, YIELD_OBS)

Parser = Callable[
    [str],
    Union[
        tuple[Literal[False], str],
        tuple[Literal[True], list[ActionTuple]]
    ]
]


class ScriptCommands(str, Enum):
    Go = "Go"
    Turn = "Turn"
    Think = "Think"


# Specify language: Should be a string of "cmd(arg);" repeated many times
minimal_parser_act_spec = rf"({'|'.join([ScriptCommands.Go.value, ScriptCommands.Turn.value])})\(((\-)*[0-9]+)\)"
# Give the LLM maximum permission to write characters in its thought
minimal_parser_think_spec = rf"{ScriptCommands.Think.value}\([^\)]*\)"
minimal_parser_language_spec = rf"^(({minimal_parser_act_spec}|{minimal_parser_think_spec});)+$"

minimal_parser_fail_message = "Input does not match language spec: "
minimal_parser_too_many_obs_message = "Maximum obs exceeded in script: "


def _get_aai_commands_from_script_values(command: str, arg: int) -> list[ActionTuple]:
    if arg == 0:
        return []
    if command == ScriptCommands.Go.value:
        if arg > 0:
            return [
                action_name_to_action_tuple["FORWARDS"]
            ] * arg
        else:
            return [
                action_name_to_action_tuple["BACKWARDS"]
            ] * abs(arg)
    elif command == ScriptCommands.Turn.value:
        if arg > 0:
            return [
                action_name_to_action_tuple["RIGHT"]
            ] * (arg // DEGREES_PER_ROTATE)
        else:
            return [
                action_name_to_action_tuple["LEFT"]
            ] * (abs(arg) // DEGREES_PER_ROTATE)
    else:  # Defend against bad commands
        raise ValueError(f"Unrecognised command {command}")


def minimal_parser(script: str) -> Union[
    tuple[Literal[False], str],
    tuple[Literal[True], list[ActionTuple]]
]:
    # Remove some irrelevant characters we've seen LLMs add
    script = script.replace(" ", "")
    script = script.replace("\n", "")

    # Check the script matches the language
    if re.match(minimal_parser_language_spec, script) is None:
        return False, minimal_parser_fail_message + script
    else:
        actions: list[ActionTuple] = []
        command_strings = re.findall(rf"({minimal_parser_act_spec}|{minimal_parser_think_spec})", script)
        for command_tuple in command_strings:
            command_string = command_tuple[0]
            think_match = re.match(minimal_parser_think_spec, command_string)
            # Skip any think actions
            if think_match is not None:
                continue
            command_match = re.match(minimal_parser_act_spec, command_string)
            # Defend against a non-matching command leaking in
            if command_match is None:
                return False, f"Non matching command found {command_string}"
            actions += _get_aai_commands_from_script_values(command_match.group(1), int(command_match.group(2)))
        return True, actions
