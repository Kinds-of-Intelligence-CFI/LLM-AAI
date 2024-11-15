from src.experimentation.experiments.experiment import Experiment
from src.experimentation.experiments.experiment1 import Experiment1


class ExperimentFactory:
    registry = {
        "experiment1": Experiment1,
    }

    @classmethod
    def create_experiment(cls, name: str, **kwargs) -> Experiment:
        return cls.registry[name](**kwargs)
