import asyncio
import os
import chainlit as cl
import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv

# Disable tracing since we're not connected to a supported tracing provider
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
@cl.on_chat_start
async def on_chat_start():
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
    agent = Agent(
            name="Ponniyin Selvan tutor",
            instructions="You are a Ponniyin Selvan tutor. Assist the user in learning about Ponniyin Selvan Novel using content exclusively from the following links:  ### https://www.projectmadurai.org/pm_etexts/utf8/pmuni0278_01.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni0278_02.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni0278_03.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni0278_04.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni0386.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni0883.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni1001.html; https://www.projectmadurai.org/pm_etexts/utf8/pmuni1013.html. Do not reference or include information from any external sources outside of these links. Ensure responses are focused, accurate, crisp, and directly draw from the provided material.",
            model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):  
    agent = cl.user_session.get("agent")  
    result = await Runner.run(agent, input=message.content)  # Changed from message.text to message.content
    await cl.Message(
        content=result.final_output,
        author="assistant",
        metadata={"source": "agent"},
    ).send()

    print(f"Message received: {message.content}")  # Changed from message.text to message.content
    print(f"Response sent: {result.final_output}")

""" async def main():
    result = await Runner.run(agent, input="hi how are you?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main()) """
