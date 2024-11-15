import pickle
import warnings
from time import sleep

import google.api_core.exceptions

import user_settings
from typing import List, Literal, Optional, Union

import google.generativeai as genai
from google.generativeai.types.content_types import (
    BlobDict,
    ContentDict,
    PartType,
)
import numpy as np

from src.llms.llm import LLMAPI, PROMPT_CONTENTS, LLMSession, PromptElement

SupportedGeminiModels = Literal["gemini-1.5-flash", "gemini-1.5-pro"]

# Hard-coded because not specified by google.generativeai library
# Can be found as private constants in the following modules:
# google/generativeai/generative_models.py and venv/lib/python3.9/site-packages/google/generativeai/caching.py
# And specified in online docs: https://ai.google.dev/api/caching#Content
SupportedGeminiRoles = Union[Literal["user"], Literal["model"]]

# Typing to indicate that the roles passed into the below ContentDict params are SupportedGeminiRoles.
user_role: SupportedGeminiRoles = "user"
assistant_role: SupportedGeminiRoles = "model"


class GeminiSession(LLMSession):
    stop_sequences = ["<EOS>"]

    def __init__(self, api_key: str, model: SupportedGeminiModels) -> None:
        super().__init__()
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)
        self._history: list[ContentDict] = []

    def prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        resp_prefix: Optional[str] = None,
    ) -> str:
        # Confirm we have at least one text element
        assert (
            len([True for contents_type, _ in prompt_contents if contents_type.value == PromptElement.Text.value]) > 0
        ), "Must have at least 1 text element"
        prompt = self._prompt_contents_to_prompt(prompt_contents)
        self._history.append(ContentDict(role=user_role, parts=prompt))
        if resp_prefix is not None:
            self._history.append(ContentDict(role=assistant_role, parts=[resp_prefix]))

        sleep_time_between_prompt_tries = 20
        try:
            message = self._client.generate_content(
                contents=self._history,
                generation_config=genai.types.GenerationConfig(
                    stop_sequences=GeminiSession.stop_sequences,
                    candidate_count=1,
                    max_output_tokens=1024,
                    temperature=0.0,
                )
            )
        except google.api_core.exceptions.InternalServerError as e:
            # Retry once on internal errors
            print(str(e))
            print(f"wait {sleep_time_between_prompt_tries} seconds and retry")
            sleep(sleep_time_between_prompt_tries)
            message = self._client.generate_content(
                contents=self._history,
                generation_config=genai.types.GenerationConfig(
                    stop_sequences=GeminiSession.stop_sequences,
                    candidate_count=1,
                    max_output_tokens=1024,
                    temperature=0.0,
                )
            )
        except Exception as e:
            if "Unknown field for Candidate" in str(e):
                print(str(e))
                print(f"wait {sleep_time_between_prompt_tries} seconds and retry")
                sleep(sleep_time_between_prompt_tries)
                message = self._client.generate_content(
                    contents=self._history,
                    generation_config=genai.types.GenerationConfig(
                        stop_sequences=GeminiSession.stop_sequences,
                        candidate_count=1,
                        max_output_tokens=1024,
                        temperature=0.0,
                    ),
                )
            else:
                raise e

        self.input_costs = np.append(self.input_costs, message.usage_metadata.prompt_token_count)
        self.output_costs = np.append(self.output_costs, message.usage_metadata.candidates_token_count)

        response_content = message.text

        if len(message.parts) > 1:
            raise ValueError(f"Unexpected multiple returns: {message.parts}")
        if resp_prefix is not None:
            self._history.pop()
            self._history.append(ContentDict(role=assistant_role, parts=[resp_prefix + ''.join(
                [block.text for block in response_content]
            )]))
        else:
            self._history.append(ContentDict(role=assistant_role, parts=[response_content]))
        return response_content


    @property
    def history(self) -> list[ContentDict]:
        return [message for message in self._history]

    def load_from_history_file(self, file: str) -> None:
        assert file.endswith(".pkl"), "File must be a pickle (.pkl)"
        with open(file, "rb") as f:
            try:
                self._history = pickle.load(f)
            finally:
                f.close()

    @staticmethod
    def _prompt_contents_to_prompt(prompt_contents: PROMPT_CONTENTS) -> List[PartType]:
        return [
            contents
            if type.value == PromptElement.Text.value
            else BlobDict(mime_type="image/jpeg", data=contents)
            for type, contents in prompt_contents
        ]

    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        self._history.append(ContentDict(role=user_role, parts=self._prompt_contents_to_prompt(prompt_contents)))
        self._history.append(ContentDict(role=assistant_role, parts=self._prompt_contents_to_prompt(response_contents)))


    @staticmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        with open(path_to_pkl, "rb") as file:
            history: List[ContentDict] = pickle.load(file)
        return [element["parts"][0] for element in history if element["role"] == assistant_role]


class GeminiAPI(LLMAPI):
    def __init__(self, api_key: str, model: SupportedGeminiModels) -> None:
        self._api_key = api_key
        self._model = model

    def start_session(self) -> LLMSession:
        return GeminiSession(self._api_key, self._model)


if __name__ == "__main__":
    import base64
    import httpx

    # Testing standard usage
    session = GeminiAPI(api_key=user_settings.GEMINI_API_KEY,
                            model="gemini-1.5-flash").start_session()

    # -- Check response to text-only prompt --
    print(f"Prompt response: {session.prompt([(PromptElement.Text, 'Hello')], None)}")
    print(f"Session history: {session.history}")

    # -- Check response to image+text prompt --
    ant_picture = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    bee_picture = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    media_type = "image/jpeg"
    ant_data = base64.b64encode(httpx.get(ant_picture).content).decode("utf-8")
    bee_data = base64.b64encode(httpx.get(bee_picture).content).decode("utf-8")
    print(session.prompt([
        (PromptElement.Image, ant_data),
        (PromptElement.Image, bee_data),
        (PromptElement.Text, "What is the difference between these two images?")
    ]))

    # -- Check that costing is being tracked expectedly --
    print(f"Input costs: {session.input_costs}")
    print(f"Output costs: {session.output_costs}")