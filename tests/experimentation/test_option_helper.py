from src.experimentation.options_helper import check_options, load_options

VALID_OPTIONS = {
    "aai_config_path": "data/arena_configs/unclassified",
    "aai_seeds": [21, 42],
    "output_folder_path": "outputs/",
    "max_conversation_turns": [5],
    "manually_prompt_llm": False,
    "save_observations": True,
    "show_observations": False,
    "play": False,
    "watch_agent_interact": True,
    "verbose": True,
    "llm_family": "claude",
    "llm_model": "claude-3-5-sonnet-20240620",
    "experiment_name": "experiment1",
    "resolution": [150, 600],
    "num_arena_loops": 1,
    "llm_family_switch": None,
    "llm_model_switch": None,
    "n_shot_examples_path": None
}


def test_check_options_should_return_true_when_valid_options_passed():
    assert check_options(VALID_OPTIONS)


def test_load_options_should_return_dict():
    assert (
        load_options(options_path="tests/experimentation/example_options.yaml")
        == VALID_OPTIONS
    )
