def create_in_session_reward_text(health: float, remaining_scripts: int) -> str:
    return f"ENVIRONMENT: Your remaining health is {health: .1f}."


IN_SESSION_MSG_TO_LLM = create_in_session_reward_text

YIELD_OBS_MESSAGE = "Environment observation captured:"

PREVIOUS_RESPONSE_IS_INVALID = ("ENVIRONMENT: Your previous response is invalid. Remember to use commands from "
                                "the scripting language only. You won't move until you do but your health will keep "
                                "decreasing! ")