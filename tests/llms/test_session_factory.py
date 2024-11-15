from src.llms.session_factory import LLMSessionFactory
from src.llms.human import HumanSession


def test_factory_can_create_basic_llm_session():
    human_session = LLMSessionFactory.create_llm_session(name="human", api_key="",  model="")
    assert isinstance(human_session, HumanSession)
