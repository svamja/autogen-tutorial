import os
from dotenv import load_dotenv
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."

async def get_nutrients(fruit: str) -> str:
    return f"The {fruit} is rich in Vitamin C."

async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=OpenAIChatCompletionClient(
            model=GPT_MODEL,
            api_key=OPENAI_API_KEY,
        ),
        tools=[get_weather],
    )

    # Define an agent
    nutrition_agent = AssistantAgent(
        name="nutrition_agent",
        model_client=OpenAIChatCompletionClient(
            model=GPT_MODEL,
            api_key=OPENAI_API_KEY,
        ),
        tools=[get_nutrients],
    )

    # Define a team with a single agent and maximum auto-gen turns of 1.
    agent_team = RoundRobinGroupChat([weather_agent, nutrition_agent], max_turns=1)

    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        stream = agent_team.run_stream(task=user_input)
        await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
# await main()

asyncio.run(main())


