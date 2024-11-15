from src.llms.llm import PromptElement
import base64
import httpx
import pytest
from src.llms.claude import AnthropicAPI, SupportedAnthropicModels
from src.llms.gpt import GPTAPI, SupportedGPTModels
from src.llms.gemini import GeminiAPI, SupportedGeminiModels
from user_settings import CLAUDE_API_KEY, GPT_API_KEY, GEMINI_API_KEY
from typing import Tuple, Union

SupportedLLMModels = Union[SupportedAnthropicModels, SupportedGPTModels, SupportedGeminiModels]

def _get_api_key_and_model(cls) -> Tuple[str, SupportedLLMModels]:
    if cls == AnthropicAPI:
        return CLAUDE_API_KEY, "claude-3-5-sonnet-20240620"
    elif cls == GPTAPI:
        return GPT_API_KEY, "gpt-4o-2024-05-13"
    elif cls == GeminiAPI:
        return GEMINI_API_KEY, "gemini-1.5-flash"
    raise NotImplementedError("Class for key and model search not found")

@pytest.mark.parametrize("cls", [AnthropicAPI, GPTAPI, GeminiAPI])
def test_start_session(cls):
    api_key, model = _get_api_key_and_model(cls)
    cls(api_key=api_key, model=model).start_session()

@pytest.mark.parametrize("cls", [AnthropicAPI, GPTAPI, GeminiAPI])
def test_prompt_text(cls):
    api_key, model = _get_api_key_and_model(cls)
    session = cls(api_key=api_key, model=model).start_session()
    response = session.prompt(prompt_contents=[(PromptElement.Text, 'Hello')], resp_prefix=None)
    assert isinstance(response, str)

@pytest.mark.parametrize("cls", [AnthropicAPI, GPTAPI, GeminiAPI])
def test_prompt_image_and_text(cls):
    api_key, model = _get_api_key_and_model(cls)

    ant_picture = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    bee_picture = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    ant_data = base64.b64encode(httpx.get(ant_picture).content).decode("utf-8")
    bee_data = base64.b64encode(httpx.get(bee_picture).content).decode("utf-8")

    session = cls(api_key=api_key, model=model).start_session()
    response = session.prompt([
        (PromptElement.Image, ant_data),
        (PromptElement.Image, bee_data),
        (PromptElement.Text, "What is the difference between these two images?")
    ])
    assert isinstance(response, str)

@pytest.mark.parametrize("cls", [AnthropicAPI, GPTAPI, GeminiAPI])
def test_image_and_text_inputs_are_more_costly_than_text_outputs(cls):
    api_key, model = _get_api_key_and_model(cls)

    ant_picture = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    bee_picture = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    ant_data = base64.b64encode(httpx.get(ant_picture).content).decode("utf-8")
    bee_data = base64.b64encode(httpx.get(bee_picture).content).decode("utf-8")

    session = cls(api_key=api_key, model=model).start_session()
    session.prompt([
        (PromptElement.Image, ant_data),
        (PromptElement.Image, bee_data),
        (PromptElement.Text, "What is the difference between these two images?")
    ])

    total_input_costs = sum(session.input_costs)
    total_output_costs = sum(session.output_costs)

    assert total_input_costs > total_output_costs
