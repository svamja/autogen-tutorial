import os
from dotenv import load_dotenv
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL")

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
      model=GPT_MODEL,
      api_key=OPENAI_API_KEY,
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")
# Define a termination condition that stops the task after 5 messages.
max_message_termination = MaxMessageTermination(5)
# Combine the termination conditions using the `|`` operator so that the
# task stops when either condition is met.
termination = text_termination | max_message_termination

# Create a team with the primary and critic agents.
reflection_team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=termination)

async def main():
    # Run the team and stream messages to the console.
    # stream = reflection_team.run_stream(task="Write a short poem about fall season.")
    # asyncio.run(Console(stream))
    await Console(
        reflection_team.run_stream(task="Write a short poem about fall season.")
    )  # Stream the messages to the console.

asyncio.run(main())

