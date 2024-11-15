import pickle
from typing import List, Literal, Optional, TypedDict, Union

from src.llms.llm import (
    LLMAPI,
    PROMPT_CONTENTS,
    LLMSession,
    PromptElement,
    LLMMessageParam
)


# Use this to play AAI as a user.
class HumanSession(LLMSession):
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        print(f"Human LLM ignoring key and model {api_key}, {model}")
        self._history: List[LLMMessageParam] = []
        self.input_costs = []
        self.output_costs = []

    def prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        resp_prefix: Optional[str] = None,
    ) -> str:
        prompt = prompt_contents
        self._history.append(
            LLMMessageParam(role="user", content=prompt_contents)
        )
        print(f"Prefix: {resp_prefix}")
        print(f"Prompt: {prompt}")
        response = input("Response: ")
        human_prompt_contents = [(PromptElement.Text, response)]
        self._history.append(
            LLMMessageParam(role="assistant", content=human_prompt_contents)
        )
        return response

    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        self._history.append(
            LLMMessageParam(role="user", content=prompt_contents)
        )
        if len(response_contents) > 1:
            raise NotImplementedError()
        self._history.append(
            LLMMessageParam(role="assistant", content=response_contents)
        )

    @property
    def history(self):
        return self._history

    def load_from_history_file(self, file: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        with open(path_to_pkl, "rb") as file:
            history: List[LLMMessageParam] = pickle.load(file)
        return [element["content"][0][1] for element in history if element["role"] == "assistant"]

class HumanAPI(LLMAPI):
    """
    A dummy API to test accessing LLM APIs without actually doing it
    """

    def start_session(self):
        return HumanSession(api_key="placeholder-key", model="placeholder-model")


if __name__ == "__main__":
    session = HumanAPI().start_session()
    print(session.prompt([(PromptElement.Text, "Hello")]))
    print(session.prompt([(PromptElement.Text, "That's interesting")]))
    print(session.history)
