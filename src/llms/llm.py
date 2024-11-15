from os.path import join
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Union, Literal, TypedDict, List
from enum import Enum
import pickle

import numpy as np
from numpy.typing import NDArray

BASE64_STRING = str



class PromptElement(Enum):
    Text = 1
    Image = 2


PROMPT_CONTENTS = list[
    Union[tuple[Literal[PromptElement.Text], str], tuple[Literal[PromptElement.Image], BASE64_STRING]]]


class LLMMessageParam(TypedDict):
    role: Literal["user", "assistant"]
    content: PROMPT_CONTENTS


class LLMSession(ABC):
    """
    General interface for a single session with an LLM
    """

    def __init__(self):
        self.input_costs: NDArray[int] = np.array([])
        self.output_costs: NDArray[int] = np.array([])
        self._history = None

    @abstractmethod
    def prompt(
            self,
            prompt_contents: PROMPT_CONTENTS,
            resp_prefix: Optional[str] = None,
    ) -> str:
        pass

    @abstractmethod
    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        """ Add an artificial prompt-response to the conversation history"""
        pass

    @property
    @abstractmethod
    def history(self) -> list[str]:
        pass

    @history.setter
    def history(self, value):
        self._history = value

    def write_to_file(self,
                      file_name: str = "llm_session_history_",
                      path: str = "./",
                      write_from_index: int = 0) -> None:
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"Saving conversation history to {path + file_name + time}.txt")
        with open(f"{path + file_name + time}.txt", "a") as f:
            try:
                f.write("\n".join([str(message) for message in self.history[write_from_index:]]))
            finally:
                f.close()
        with open(f"{path + file_name + time}.pkl", "wb") as f:
            try:
                pickle.dump(self.history, f)
            finally:
                f.close()

    def save_cost_arrays(self, cost_folder_path: str = "./"):
        np.save(join(cost_folder_path, "costs_input.npy"), self.input_costs)
        np.save(join(cost_folder_path, "costs_output.npy"), self.output_costs)

    @abstractmethod
    def load_from_history_file(self,
                               file: str) -> None:
        pass
    @staticmethod
    @abstractmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        pass


class LLMAPI(ABC):
    """
    General interface for an LLM
    """

    @abstractmethod
    # TODO: Get rid of this wrapper
    def start_session(self) -> LLMSession:
        pass
