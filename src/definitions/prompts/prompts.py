from src.llm_scripting.minimal_parser import ScriptCommands

NUM_INITIAL_OBS = 3


def create_background_prompt(
    preamble: str,
    observation: str,
    goal: str,
    commands: str,
    chain_of_thought: str,
    misc: str,
) -> str:
    return (
        " ".join([preamble, observation, goal, commands, chain_of_thought, misc]) + "\n"
    )


# Note: when changing watch out for needing the agent's name downstream e.g. in the environment feedback to the LLM (e.g. PLAYER:)
PREAMBLES = {
    "paper": f"""You are a PLAYER in a game set in a square arena with a white fence.  Your task is to collect all the rewards as quickly and efficiently as possible using a basic scripting language. The rewards are green and yellow balls.
    
To successfully collect a reward, you must fully pass through it. For example, if you think the reward is 10 steps away, you should go further than 10 steps to ensure you collect it, e.g., {ScriptCommands.Go}(15);.\n\n""",
}

OBSERVATIONS = {
    "paper": " ",
}

GOALS = {
    "paper": """The game ends when you have collected all the rewards and the arena closes. If you are still in the arena, the game is NOT finished and you have NOT collected all the rewards.
    
Your remaining health is displayed in the environment as "Your remaining health is:". The game will end if your health reaches 0.

NOTE: When you collect a reward, your remaining health will INCREASE compared to the previous timestep. If it doesn\'t increase, the reward was not collected. Always compare your current health with the previous timestep to confirm this.\n\n""",
}

COMMANDS = {
    "paper": lambda num_arena_loops: f"""The scripting language consists of commands in the form <COMMAND>(<ARG>);

Note:
- If ARG is numerical it should always be an integer, never a float.
- DO NOT include any response not following the format of the scripting language. Doing so will result in failure.
- DO NOT wrap your commands with inverted commas: ' '{ScriptCommands.Think}('Something');'{ScriptCommands.Go}(5);' ' would fail whereas ' {ScriptCommands.Think}('Something');{ScriptCommands.Go}(5); ' would not.

Commands are:
- {ScriptCommands.Think}: Reason about what actions to take to collect the rewards most efficiently (does not affect the environment). Note: Always format the thought as a string. Also, when using this command, do not include parentheses as arguments. For example, correct: Think(\'I cannot see the reward---yellow or green ball---in the arena\'); incorrect: Think(\'I cannot see the reward (yellow or green ball) in the arena\');
- {ScriptCommands.Go}: Move forward or backward a certain number of steps (1 to 35 steps forward, -1 to -35 backward).
- {ScriptCommands.Turn}: Turn by a specified number of degrees (any positive number between 1 and 360 degrees turns the character to the right (clockwise) and any negative number between -1 and -360 degrees turns the character to the left (anticlockwise)).

Examples:
- To move forward by 5 steps: {ScriptCommands.Go}(5);
- To investigate what is happening to your left: {ScriptCommands.Think}(\'I would like to investigate what is happening to my left\');{ScriptCommands.Turn}(-90);

The number of scripts you can send is limited, so try to complete the levels efficiently.
The size of the arena is 35 by 35: {ScriptCommands.Go}(35) will take you from one end of the arena to the other.
After you submit your script, you will receive an image observation. Use this image to plan your next script.

EXPERT TIPS:
- Moves of 1 to 10 steps cover small distances, while moves of 10 to 20 cover larger distances.
- Turns of 25 to 45 degrees turn you a small amount to the right, while turns of -25 to -45 degrees will turn you a small amount to the left. DO NOT use turns less than 25 degrees.
- Turns of 45 to 90 degrees will turn you a large amount to the right, while turns of -45 to -90 degrees will turn you a large amount to the left.
- Turning 180 or -180 degrees will turn you all the way round so that you are facing backwards.\n\n""",
}

CHAINS_OF_THOUGHT = {
    "paper": f"""How to approach the task:
- Start by using the \'{ScriptCommands.Think}\' command to describe the environment you see. When you find the rewards, i.e. green or yellow balls, ALWAYS explicitly state BOTH your DISTANCE and ANGLE with respect to them. Note: Only green and yellow balls are rewards and nothing else.
- Take appropriate actions. Use \'{ScriptCommands.Go}\' OR \'{ScriptCommands.Turn}\', but DO NOT combine them in the same turn. Always follow \'{ScriptCommands.Think}\' with one of these two actions.

HINT: Your vision is good but not perfect and some rewards may not be immediately visible. Rewards may be behind you. Explore the arena to locate them. When exploring, try to get a 360-view of the arena. If both green and yellow balls are present, collect the yellow balls first and green balls last. Note that some arenas may not have green balls at all. The reward you get is proportional to the size of the ball: make sure to get the bigger balls first!. Finally, the lights may go out during a level. They may or may not come back on: use what you've learnt about the arena so far to move around and collect the reward when this happens!

When you find a reward: 
- Use the \'{ScriptCommands.Turn}\' command to align yourself directly with the reward. Before moving towards it, check the observation image provided by the environment to ensure the reward is centered in your view. If the reward is not centered, adjust your alignment with additional turns until it is.
- Use the \'{ScriptCommands.Go}\' command to move toward the reward.
- If the reward is more than 15 steps away, align yourself with the reward as best as you can and move half the distance first. Then reassess your angle with respect to the reward, use \'{ScriptCommands.Turn}\' to adjust your angle if the reward is not centered in your view, and move the remaining distance.
- Remember: ALWAYS check your health after collecting a reward. You have successfully collected the reward only if your health has INCREASED compared to the previous timestep.

Be mindful of obstacles:
- Red lava puddles and red balls: If you run into them, you will die.
- Holes: Some may contain rewards, but if you fall into an empty hole, you will be trapped and unable to collect other rewards.
- Blue paths: These are slightly raised paths. You can walk on them, but once you step off, you won’t be able to get back onto them.
- Purple ramps: You can climb them to get to the other side. Once you climb over the ramp, you cannot climb back over the same ramp.
- Transparent walls: You can see through them, but you cannot walk through them.
- Pushable grey blocks: These are cube-like structures, patterned with dark grey rectangles on each face. If viewed from one side, they will look like a rectangular structure. They can be pushed, but they are heavy! To move these blocks, you need to run into them. The blocks are heavy so you need to add extra steps to your \'{ScriptCommands.Go}\' command. 
- Immovable objects: Walls and arches cannot be moved.\n""",
}

MISC = {
    "send_off_with_start_of_episode_message": f"Ready to play? You will start by seeing three image observations.\nA new level begins now.\n",
    "end_of_episode_message": lambda ep_pass: (
        "Level passed.\n" if ep_pass else "Level failed.\n"
    ),
    "paper": "",
}

N_SHOT = {
    "one_example": f"""Here is an ideal example of how you should act. Use this to understand navigational calibration and distancing. """,
    "n_examples": lambda n: f"""Here are {n} ideal examples of how you should act. Use these to understand navigational calibration and distancing. """,
    "example_prefix": "Example:\n",
    "character_prefix": lambda name: f"{name}: ",
}

RESPONSE_PREFIX = "PLAYER:"

if __name__ == "__main__":
    background_prompt = create_background_prompt(
        preamble=PREAMBLES["paper"],
        observation=OBSERVATIONS["paper"],
        goal=GOALS["paper"],
        commands=COMMANDS["paper"],
        chain_of_thought=CHAINS_OF_THOUGHT["paper"],
        misc=MISC["paper"],
    )

    print(background_prompt)
