import os
import pickle
from typing import Dict, List

import yaml

def load_options(options_path: str) -> Dict:
    with open(options_path, "r") as file:
        options = yaml.safe_load(file)
    check_options(options=options)
    return options


def check_options(options: Dict) -> bool:
    assert isinstance(options["aai_config_path"], str)
    assert os.path.exists(options["aai_config_path"])

    assert isinstance(options["output_folder_path"], str)
    assert os.path.exists(options["output_folder_path"])

    # Note > 1 is because the command prompt mentions "scripts"
    # If using 1 script, update the command prompt to handle this
    if isinstance(options["max_conversation_turns"], list):
        assert all(
            isinstance(max_turns, int)
            for max_turns in options["max_conversation_turns"]
        )
        assert all(
            max_turns > 1 or max_turns == -1
            for max_turns in options["max_conversation_turns"]
        )
    else:
        assert isinstance(options["max_conversation_turns"], int)
        assert (
            options["max_conversation_turns"] > 1
            or options["max_conversation_turns"] == -1
        )

    assert isinstance(options["manually_prompt_llm"], bool)
    assert isinstance(options["save_observations"], bool)
    assert isinstance(options["show_observations"], bool)
    assert isinstance(options["play"], bool)

    assert isinstance(options["watch_agent_interact"], bool)
    assert isinstance(options["verbose"], bool)
    assert isinstance(options["experiment_name"], str)

    assert isinstance(options["llm_family"], str)
    assert isinstance(options["llm_model"], str)
    assert isinstance(options["llm_family_switch"], str) or options["llm_family_switch"] is None
    assert isinstance(options["llm_model_switch"], str) or options["llm_model_switch"] is None
    if isinstance(options["llm_family_switch"], str):
        # Switch can only be used with RecordingSession
        assert options["llm_family"] == "recording"
        assert options["llm_model"]  == "recording"
        assert isinstance(options["llm_model_switch"], str)
        assert options["learn_across_arenas"], "If not learning across arenas, only use switch for the arena that was in flight when the run failed"

    if isinstance(options["resolution"], list):
        assert all(isinstance(resolution, int) for resolution in options["resolution"])
    else:
        assert isinstance(options["resolution"], int)

    if isinstance(options["num_arena_loops"], list):
        assert all(
            isinstance(num_arena_loops, int)
            for num_arena_loops in options["num_arena_loops"]
        )
    else:
        assert isinstance(options["num_arena_loops"], int)

    assert isinstance(options["n_shot_examples_path"], str) or options["n_shot_examples_path"] is None
    if isinstance(options["n_shot_examples_path"], str):
        assert os.path.exists(options["n_shot_examples_path"])
        if not options["n_shot_examples_path"].endswith(".pkl"):
            # If a directory is passed, it should only contain .pkl files of saved histories
            for file in os.listdir(options["n_shot_examples_path"]):
                assert file.endswith(".pkl")
                with open(os.path.join(options["n_shot_examples_path"],file), "rb") as f:
                    loaded_history = pickle.load(f)
                    assert isinstance(loaded_history, list)

    return True


# TODO: add a check that IF the specified path is a folder (i.e. not a single arena
#  config) that all the files that the folder contains are YAMLs.

# TODO: find a more elegant way to check those params that could be passed as either a list or another type.
#  or should it be that the iterable params should always be passed as a list? Discuss with team.

# TODO: could add a check to make sure that the experiment_name is a key of ExperimentFactory's registry
#  in fact, could do this with all the factories (though, maybe not, because utils would depend on a lot of src)

# TODO: Use YAML schema instead of type checking here.

# TODO: consider writing a class to house the options (dot notation + IDE auto-complete)
