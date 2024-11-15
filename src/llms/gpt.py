import warnings
from typing import List, Literal, Optional, Union
from time import sleep

import numpy as np
import pickle
import openai
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from src.llms.llm import LLMAPI, PROMPT_CONTENTS, LLMSession, PromptElement
from user_settings import GPT_API_KEY, GPT_API_ENDPOINT

# https://platform.openai.com/docs/models/gpt-4o
SupportedGPTModels = Literal["gpt-4o-2024-05-13", "gpt-4o-mini-2024-07-18", "gpt-4-turbo-2024-04-09"]


class GPTSession(LLMSession):
    stop_sequences = ["<EOS>"]

    def __init__(self, api_key: str, model: SupportedGPTModels) -> None:
        print(f"Starting new GPT session.")
        super().__init__()
        self._client = openai.AzureOpenAI(
            azure_endpoint=GPT_API_ENDPOINT, # TODO: Fix inconsistency with this vs api key
            api_key=api_key,
            api_version="2024-05-01-preview",
        )
        self._history: List[ChatCompletionMessageParam] = []
        self._model = model

    def prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        resp_prefix: Optional[str] = None,
    ) -> str:
        assert (
            len([True for prompt_type, _ in prompt_contents if prompt_type.value == PromptElement.Text.value]) > 0
        ), "Must have at least 1 " "text element"
        prompt = self._prompt_contents_to_prompt(prompt_contents)

        self._history.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )
        if resp_prefix is not None:
            self._history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=resp_prefix
                )
            )
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                messages=self._history,
                stop=GPTSession.stop_sequences,
            )
        except openai.APIStatusError as e:
            # Retry once on these errors since we see them occasionally
            print(f"---- caught an APIStatusError error: waiting 1 minute and trying again ----\n error: {e}")
            sleep(60)
            completion = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                temperature=0.0,
                messages=self._history,
                stop=GPTSession.stop_sequences,
            )
        self.input_costs = np.append(self.input_costs, completion.usage.prompt_tokens)
        self.output_costs = np.append(
            self.output_costs, completion.usage.completion_tokens
        )

        # Note: For GPT API, the message content is provided as an optional string alone rather
        # than a List[ContentBlock] as it is the case for Claude API. See completion response format:
        # https://platform.openai.com/docs/guides/chat-completions/response-format
        choice = completion.choices[0]
        response_content = choice.message.content
        if response_content is None:
            raise ValueError(f"Unexpected empty response.")
        if completion.choices[0].finish_reason != "stop":
            warnings.warn(f"Non-standard completion finish reason: {choice.finish_reason}")
        if resp_prefix is not None:
            self._history.pop()
            self._history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=resp_prefix + response_content,
                )
            )
        else:
            self._history.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=response_content
                )
            )
        return response_content

    def artificial_prompt(
        self,
        prompt_contents: PROMPT_CONTENTS,
        response_contents: PROMPT_CONTENTS,
    ) -> None:
        if len(response_contents) != 1:
            raise NotImplementedError("Cannot handle responses of length != 1")
        """ Add an artificial prompt-response to the conversation history"""
        self._history.append(
            ChatCompletionUserMessageParam(
                role='user',
                content=self._prompt_contents_to_prompt(prompt_contents)
            )
        )
        self._history.append(
            ChatCompletionAssistantMessageParam(
                role='assistant',
                content=response_contents[0][1]
            )
        )

    @property
    def history(self) -> list[ChatCompletionMessageParam]:
        return [message for message in self._history]

    def load_from_history_file(self,
                               file: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def _prompt_contents_to_prompt(
        prompt_contents: PROMPT_CONTENTS,
    ) -> List[
        Union[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]
    ]:
        return [
            ChatCompletionContentPartImageParam(
                type="image_url",
                image_url=ImageURL(
                    # Need url prefix to interpret base64, see:
                    # https://platform.openai.com/docs/guides/vision
                    url=f"data:image/jpeg;base64,{contents}",
                    # Level of detail of the image, see:
                    # https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding
                    detail="high",
                ),
            )
            if prompt_type.value == PromptElement.Image.value
            else ChatCompletionContentPartTextParam(type="text", text=contents)
            for prompt_type, contents in prompt_contents
        ]

    @staticmethod
    def get_assistant_commands_from_pkl_history(path_to_pkl: str) -> List[str]:
        with open(path_to_pkl, "rb") as file:
            history: List[ChatCompletionMessageParam] = pickle.load(file)
        return [response["content"] for response in history if response["role"] == "assistant"]


class GPTAPI(LLMAPI):
    def __init__(self, api_key: str, model: SupportedGPTModels) -> None:
        self._api_key = api_key
        self._model = model

    def start_session(self):
        return GPTSession(self._api_key, self._model)


if __name__ == "__main__":
    session = GPTAPI(api_key=GPT_API_KEY, model="gpt-4o-mini-2024-07-18").start_session()

    # -- Basic check --
    print(session.prompt([(PromptElement.Text, "Hello")], None))
    print(session.prompt([(PromptElement.Text, "That's interesting")], None))
    print(session.history)

    # -- Image check --
    ant_picture = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    bee_picture = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    media_type = "image/jpeg"
    ant_data = base64.b64encode(httpx.get(ant_picture).content).decode("utf-8")
    bee_data = base64.b64encode(httpx.get(bee_picture).content).decode("utf-8")
    print(
        session.prompt(
            [
                (PromptElement.Image, ant_data),
                (PromptElement.Image, bee_data),
                (
                    PromptElement.Text,
                    "What is the difference between these two images?",
                ),
            ]
        )
    )
    print(session.history)

    # -- Artificial prompt check --
    session.artificial_prompt(
        [(PromptElement.Text, "Hello! How are you!")],
        [(PromptElement.Text, "Huh? Who are you and what do you want?")]
    )
    session.artificial_prompt(
        [(PromptElement.Text, "Wait, aren't you going to be friendly?")],
        [(PromptElement.Text, "To you? Bleurgh!")]
    )
    print(session.prompt([(PromptElement.Text, "Why are you being like this?")]))
    print(session.history)
