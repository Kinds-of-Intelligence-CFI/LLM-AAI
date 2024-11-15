from src.experimentation.options_helper import load_options
from src.experimentation.suite import ExperimentSuite

options_path = "./options.yaml"
options = load_options(options_path=options_path)

experiment_suite = ExperimentSuite(options=options)
experiment_suite.run()
