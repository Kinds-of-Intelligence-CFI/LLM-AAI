""" Tests written for pytest """
import pytest
from mlagents_envs.base_env import ActionTuple

from src.llm_scripting.minimal_parser import minimal_parser_fail_message, minimal_parser, ScriptCommands
from src.definitions.constants import DEGREES_PER_ROTATE
from src.definitions.cardinal_directions import action_name_to_action_tuple


def test_should_return_false_and_a_message_if_invalid_script_is_input():
    bad_script = "This_is_not_a_valid_script"
    okay, resp = minimal_parser(bad_script)
    assert not okay
    assert resp == minimal_parser_fail_message + bad_script


def test_should_return_true_for_valid_input():
    good_script = f"{ScriptCommands.Go.value}(10);"
    okay, _ = minimal_parser(good_script)
    assert okay


def test_should_ignore_surrounding_whitespace():
    good_script = f"    {ScriptCommands.Go.value}(10);    "
    okay, _ = minimal_parser(good_script)
    assert okay


def test_should_not_accept_random_trailing_characters():
    bad_script = f"{ScriptCommands.Go.value}(10); If you have any more questions, please let me know"
    okay, _ = minimal_parser(bad_script)
    assert not okay


arguments_test_should_translate_individual_arguments_correctly: list[tuple[str, list[ActionTuple]]] = [
    (f"{ScriptCommands.Go.value}(1);", [action_name_to_action_tuple["FORWARDS"]]),
    (f"{ScriptCommands.Go.value}(-1);", [action_name_to_action_tuple["BACKWARDS"]]),
    (f"{ScriptCommands.Turn.value}({DEGREES_PER_ROTATE});", [action_name_to_action_tuple["RIGHT"]]),
    (f"{ScriptCommands.Turn.value}(-{DEGREES_PER_ROTATE});", [action_name_to_action_tuple["LEFT"]]),
]


@pytest.mark.parametrize("input,expected", arguments_test_should_translate_individual_arguments_correctly)
def test_should_translate_individual_arguments_correctly(input: str, expected: list[ActionTuple]):
    assert minimal_parser(input) == (True, expected)


def test_should_give_empty_list_for_0_length_actions():
    script = f"{ScriptCommands.Go.value}(0);"
    assert minimal_parser(script) == (True, [])


def test_should_decompose_longer_action_sequences_at_semicolons():
    input = f"{ScriptCommands.Go.value}(1);{ScriptCommands.Turn.value}(7);{ScriptCommands.Turn.value}(-7);"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["FORWARDS"],
        action_name_to_action_tuple["RIGHT"],
        action_name_to_action_tuple["LEFT"]
    ])


def test_should_ignore_white_spaces_between_longer_action_sequences():
    input = f"{ScriptCommands.Go.value}(1);  {ScriptCommands.Turn.value}(7);"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["FORWARDS"],
        action_name_to_action_tuple["RIGHT"],
    ])

def test_should_ignore_newline_between_commands():
    input = f"{ScriptCommands.Go.value}(1);\n{ScriptCommands.Turn.value}(7);"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["FORWARDS"],
        action_name_to_action_tuple["RIGHT"],
    ])


def test_for_go_arg_should_produce_arg_forwards():
    input = f"{ScriptCommands.Go.value}(3);"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["FORWARDS"]
    ] * 3)


def test_for_negative_go_arg_should_produce_arg_backwards():
    input = f"{ScriptCommands.Go.value}(-3);"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["BACKWARDS"]
    ] * 3)


# DEGREES_PER_ROTATE
def test_for_turn_arg_should_produce_arg_by_degrees_turns():
    input = f"{ScriptCommands.Turn.value}({DEGREES_PER_ROTATE * 3});"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["RIGHT"]
    ] * 3)


def test_for_negative_turn_arg_should_produce_arg_by_degrees_turns():
    input = f"{ScriptCommands.Turn.value}({DEGREES_PER_ROTATE * -3});"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["LEFT"]
    ] * 3)


def test_should_integer_divide_number_of_degrees():
    input = f"{ScriptCommands.Turn.value}({DEGREES_PER_ROTATE * 3 + DEGREES_PER_ROTATE - 1});"
    assert minimal_parser(input) == (True, [
        action_name_to_action_tuple["RIGHT"]
    ] * 3)

def test_should_reject_script_with_missing_semicolon():
    input = f"{ScriptCommands.Think.value}('I then go forwards'){ScriptCommands.Turn.value}(-90);{ScriptCommands.Go.value}(10);"
    ok, err = minimal_parser(input)
    assert not ok
    assert err == f"Input does not match language spec: Think('Ithengoforwards')Turn(-90);Go(10);"


# >>> Think command tests
def test_think_command_is_accepted_and_not_translated_into_an_aai_action():
    input = f"{ScriptCommands.Think.value}(This is non-script text.);{ScriptCommands.Go.value}(1);"
    assert minimal_parser(input) == (True, [action_name_to_action_tuple["FORWARDS"]])

def test_think_command_should_accept_any_character():
    # Excluding ')' as that is needed to unambiguously end the command
    input = f"{ScriptCommands.Think.value}(abc123(\"'%%#~\\);{ScriptCommands.Go.value}(1);"
    assert minimal_parser(input) == (True, [action_name_to_action_tuple["FORWARDS"]])


def test_think_command_with_spaces_and_one_command():
    input = f"{ScriptCommands.Think.value}(This is non-script text.   );  {ScriptCommands.Go.value}(1);  "
    assert minimal_parser(input) == (True, [action_name_to_action_tuple["FORWARDS"]])


def test_think_command_with_no_spaces_and_multiple_commands():
    input = (f"{ScriptCommands.Think.value}(This is non-script text.);{ScriptCommands.Go.value}(1);"
             f"{ScriptCommands.Turn.value}({DEGREES_PER_ROTATE * 2});")
    assert minimal_parser(input) == (True, [action_name_to_action_tuple["FORWARDS"],
                                            action_name_to_action_tuple["RIGHT"],
                                            action_name_to_action_tuple["RIGHT"]])


def test_think_command_with_spaces_and_multiple_commands():
    input = (f"{ScriptCommands.Think.value}(This is non-script text.);   {ScriptCommands.Go.value}(1);"
             f" {ScriptCommands.Turn.value}({DEGREES_PER_ROTATE * 2});   ")
    assert minimal_parser(input) == (True, [action_name_to_action_tuple["FORWARDS"],
                                            action_name_to_action_tuple["RIGHT"],
                                            action_name_to_action_tuple["RIGHT"]])

def test_think_with_semicolon_should_be_valid():
    good_script = f"{ScriptCommands.Think.value}(I am thinking; I have decided to do x);Go(5);Turn(15);"
    okay, resp = minimal_parser(good_script)
    print(resp)
    assert okay
# <<< Think command tests
