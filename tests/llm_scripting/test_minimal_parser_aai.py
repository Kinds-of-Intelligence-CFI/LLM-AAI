""" Tests written for pytest """
from animalai.environment import AnimalAIEnvironment
import os

from src.llm_scripting.minimal_parser import minimal_parser, ScriptCommands, YIELD_OBS
from user_settings import ENV_PATH, LOG_FOLDER
from src.definitions.constants import FRAMES_BETWEEN_OBS

configuration_file = "data/arena_configs/unclassified/test_bad_goal_good_goal.yaml"
expected_yield_positions = [0,1,FRAMES_BETWEEN_OBS+2]

def test_a_script_works_with_aai():
    env = AnimalAIEnvironment(
    file_name=ENV_PATH,
        arenas_configurations=configuration_file,
        seed=5555,
        play=False,
        useCamera=False,
        inference=True,
        log_folder=LOG_FOLDER
    )
    try:
        action_script = f"{ScriptCommands.Think.value}('I think I need to turn to the left and then go forwards');{ScriptCommands.Turn.value}(-90);{ScriptCommands.Go.value}(10);"
        ok, actions = minimal_parser(action_script)
        assert ok and isinstance(actions, list)
        behavior = list(env.behavior_specs.keys())[0]  # by default should be AnimalAI?team=0
        env.step()  # Need to make a first step in order to get an observation.
        total_reward = 0
        dec, term = env.get_steps(behavior)
        done = len(term.reward) > 0
        total_reward += dec.reward if len(dec.reward) > 0 else 0 + term.reward if len(term.reward) > 0 else 0
        while not done and len(actions) > 0:
            action = actions.pop(0)
            if action == YIELD_OBS():
                print("Yield obs")
                continue
            env.set_actions(behavior, action)
            env.step()
            dec, term = env.get_steps(behavior)
            done = len(term.reward) > 0
            total_reward += dec.reward if len(dec.reward) > 0 else 0 + term.reward if len(term.reward) > 0 else 0
        assert total_reward == 0.707
    finally:
        env.close()
