""" Tests written for pytest """
from src.llm_scripting.minimal_parser import minimal_parser, ScriptCommands
from src.definitions.constants import DEGREES_PER_ROTATE
from src.definitions.cardinal_directions import action_name_to_action_tuple
from src.llms.claude import AnthropicAPI
from src.llms.llm import PromptElement
from user_settings import CLAUDE_API_KEY

# Note: this is just made up data, not from an AAI run
prompt = \
f"""
You are playing a game in which you control a character using a basic scripting language.
You are given an observation and you must write the script that will allow you to reach the
goal described in the observation.

The scripting language consists of a string of commands of the form <COMMAND>(<ARG>);

Commands are:
- {ScriptCommands.Think.value}: Reason about what move to make next (does not affect the environment)
- {ScriptCommands.Go.value}: Move the character for the provided number of timesteps (positive values are forwards and negative values are backwards)
- {ScriptCommands.Turn.value}: Turn the character by the provided number of degrees (positive values are right and negative values are left)

For example, a level might proceed like this:
ENVIRONMENT: To your left there is a GOODGOAL at distance 4.0
PLAYER: {ScriptCommands.Turn.value}(-90);{ScriptCommands.Go.value}(2);
ENVIRONMENT: Ahead there is a GOODGOAL at distance 2.0
PLAYER: {ScriptCommands.Think.value}('I only need to go forwards to get the goal now');{ScriptCommands.Go.value}(2);

Now it is your turn to play.
ENVIRONMENT: The reward is 5 units away 45 degrees to your right
"""

response_prefix = "Player:"

expected_number_of_rotates = 45 // DEGREES_PER_ROTATE

def test_llm_should_succeed_with_basic_example():
    session = AnthropicAPI(CLAUDE_API_KEY, "claude-3-sonnet-20240229").start_session()
    response = session.prompt(
        prompt_contents=[
            (PromptElement.Text, prompt)
        ],
        resp_prefix=response_prefix
    )
    ok, aai_commands = minimal_parser(response)
    assert ok
    assert isinstance(aai_commands, list)
    assert all(command == action_name_to_action_tuple["RIGHT"] for command in aai_commands[0:expected_number_of_rotates])
    # Don't make any assumptions on how many forwards the LLM chooses
    assert len(aai_commands[expected_number_of_rotates:]) > 0 \
        and all(command == action_name_to_action_tuple["FORWARDS"] for command in aai_commands[expected_number_of_rotates:])