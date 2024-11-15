import itertools
import time
import traceback
from os import listdir
from os.path import isfile, join
from typing import List, Dict
from datetime import datetime

import yaml

from src.experimentation.experiments.experiment_factory import ExperimentFactory
from src.utilities.utils import try_mkdir


class ExperimentSuite:
    def __init__(self, options: Dict):
        self.options = options
        self.timestamped_folder_path = self._create_output_directory()
        self.arena_config_paths = self._generate_list_of_arena_config_paths()
        self._save_options_to_output_directory(output_dir=self.timestamped_folder_path)

    def run(self, sleep_duration: float = 3):
        """Runs the experiment suite by iterating over the iterable params and running one experiment per set of args.

        Note:
            - Iterating over the arena configurations is part of the core experiment and is thus not in this method.
        """
        # Extract the list options that are meant to be iterated over
        iterable_options = {k: v for k, v in self.options.items() if isinstance(v, list)}
        non_iterable_options = {k: v for k, v in self.options.items() if not isinstance(v, list)}
        keys, values = zip(*iterable_options.items())

        # Run one experiment per set of iterable params within the cartesian product of the options with itself
        # But, must update src/experimentation/options_helper.py check_options method to allow for new iterable params
        for combination in itertools.product(*values):
            experiment_options = dict(zip(keys, combination))
            experiment_options.update(non_iterable_options)

            experiment_folder_name = "_".join(f"{k}_{v}" for k, v in experiment_options.items() if k in iterable_options)
            experiment_folder_path = join(self.timestamped_folder_path, experiment_folder_name)
            experiment_options["output_folder_path"] = experiment_folder_path

            try:
                experiment = ExperimentFactory().create_experiment(name=experiment_options["experiment_name"],
                                                                   options=experiment_options)
                print(f"Running experiment with options{experiment_options}")
                experiment.run()
            except Exception as e:
                print(f"Experiment failed with error: {e}")
                print(traceback.format_exc())
            finally:
                time.sleep(sleep_duration)  # Required for process not to crash

    def _create_output_directory(self) -> str:
        try_mkdir(self.options["output_folder_path"])
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamped_folder_path = join(self.options["output_folder_path"], now)
        try_mkdir(timestamped_folder_path)
        return timestamped_folder_path

    def _save_options_to_output_directory(self, output_dir: str) -> None:
        with open(f'{output_dir}/options.yaml', 'w') as outfile:
            yaml.dump(self.options, outfile, default_flow_style=False)

    def _generate_list_of_arena_config_paths(self) -> List[str]:
        if ".yaml" in self.options["aai_config_path"] or ".yml" in self.options["aai_config_path"]:
            # Enclose single-config path into a list to treat like a directory of arena configs
            config_paths = [self.options["aai_config_path"]]
        else:
            # Assume the user provided a folder containing all the arena configs to iterate over
            config_paths = []
            for element in listdir(self.options["aai_config_path"]):
                if isfile(join(self.options["aai_config_path"], element)) and (".yaml" in element or ".yml" in element):
                    config_paths += [join(self.options["aai_config_path"], element)]
        return config_paths

    def _get_initialised_experiment_folder_path(self) -> str:
        return join(self.timestamped_folder_path, "")
