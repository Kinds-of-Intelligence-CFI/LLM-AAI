from abc import ABC, abstractmethod
from typing import Union, Literal, List, Tuple, Optional
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.base_env import DecisionStep, DecisionSteps, TerminalSteps

AAIVisualObservation = Tuple[str, Optional[str]]

class VisionSystem(ABC):
    """
    A General interface for a getting LLM readable observations from Animal AI
    """

    @property
    @abstractmethod
    def observation_prompt(self) -> str:
        """ Text for the preamble to explain observations to the LLM """
        pass

    @abstractmethod
    def get_observation(self,
                        env: AnimalAIEnvironment,
                        decision_step: DecisionSteps,
                        save_observation: bool = False,
                        save_path: str = ".") -> AAIVisualObservation:
        pass

    def add_vision_obs_to_message(self,
                                  message_text: str,
                                  vision_obs: Union[str, List]) -> Union[str, List]:
        pass
