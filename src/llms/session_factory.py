from src.llms.llm import LLMSession
from src.llms.claude import AnthropicSession
from src.llms.human import HumanSession
from src.llms.gpt import GPTSession
from src.llms.gemini import GeminiSession

from src.llms.recording import RecordingSession


class LLMSessionFactory:
    registry = {
        "human": HumanSession,
        "claude": AnthropicSession,
        "gpt": GPTSession,
        "gemini": GeminiSession,
        "recording": RecordingSession
    }

    @classmethod
    def create_llm_session(cls, name: str, **kwargs) -> LLMSession:
        return cls.registry[name](**kwargs)

    @classmethod
    def get_llm_constructor(cls, name: str) -> LLMSession:
        return cls.registry[name]
