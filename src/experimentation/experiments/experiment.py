from abc import abstractmethod, ABC
from typing import Dict


class Experiment(ABC):
    def __init__(self, options: Dict) -> None:
        self.options = options

    @abstractmethod
    def run(self) -> None:
        pass
