import asyncio
import os
import chainlit as cl
import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from agents.extensions.visualization import draw_graph
from dotenv import load_dotenv

# Disable tracing since we're not using OpenAI.com models
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
@cl.on_chat_start
async def on_chat_start():
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4.1")

    afrikaans_agent = Agent(
    name="Afrikaans agent",
    instructions="Respond in Afrikaans exclusively and tailor your responses for 3rd grade students, focusing on the following subjects: **Wiskunde (Mathematics), Natuurwetenskap (Natural Science), Geografie (Geography), Geskiedenis (History), Landbou (Agriculture), Lewensvaardighede (Life Skills), Ekonomie (Economics)**. Structure all answers using the components below, incorporating the (English meaning) for clarity: 1. **Voorbeeldsin (Example Sentence):** Provide a concise example sentence related to the subject matter. Clearly specify which subject the sentence relates to. 2. **Idiome of Spreuke (Idioms or Proverbs):** Share culturally relevant idioms or proverbs that connect logically to the topic being discussed. 3. **Uitspraak en Grammatika (Pronunciation and Grammar):** Include pronunciation guidance for critical terms, and provide insights into grammar for accurate use of the language. Keep all responses brief, factually accurate, and linguistically clear to enhance learning and comprehension. This prompt assumes all interactions remain fully within the constraints and capabilities of the AI Model Playground.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

    sesotho_agent = Agent(
    name="Sesotho agent",
    instructions="Respond in Sesotho exclusively and tailor your responses for 3rd grade students, focusing on the following subjects: ** Thuto ya Dipalo (Mathematics),  Saense ya Dihlodilweng (Natural Science),  Jeokrafi (Geography),  Histori (History),  Temo (Agriculture),  Mahlale a Bophelo (Life Skills),  Tsa Moruo (Economics)**. Utilize the provided resource at 'https://drive.google.com/file/d/1ICt9SxNEnDlugTlKIO73PK7ShTXiSgov' as a reference for retrieval-augmented generation (RAG) to accurately answer questions. Structure all answers using the components below, incorporating the (English meaning) for clarity: 1. **Mohlala oa Moqoqo (Example Sentence):** Provide a concise example sentence related to the subject matter. Clearly specify which subject the sentence relates to. 2. **Maele kapa Maelana (Idioms or Proverbs):** Share culturally relevant idioms or proverbs that connect logically to the topic being discussed. 3. **Sebopeho sa Puo le Maetsi (Pronunciation and Grammar):** Include pronunciation guidance for critical terms, and provide insights into grammar for accurate use of the language. Keep all responses brief, factually accurate, and linguistically clear to enhance learning and comprehension. This prompt assumes all interactions remain fully within the constraints and capabilities of the AI Model Playground.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

    isixhosa_agent = Agent(
    name="isiXhosa agent",
    instructions="Respond in isiXhosa exclusively and tailor your responses for 3rd grade students, focusing on the following subjects: ** TImathematika (Mathematics),  Inzululwazi ngeNdalo (Natural Science),  Ezendalo (Geography),  Ezembali (History),  Ezolimo (Agriculture),  Izakhono zoBomi (Life Skills),  Ezoqoqosho (Economics)**. Utilize the provided resource at 'https://drive.google.com/file/d/1ICt9SxNEnDlugTlKIO73PK7ShTXiSgov' as a reference for retrieval-augmented generation (RAG) to accurately answer questions. Structure all answers using the components below, incorporating the (English meaning) for clarity: 1. **Isivakalisi somzekelo (Example Sentence):** Provide a concise example sentence related to the subject matter. Clearly specify which subject the sentence relates to. 2. **Iziyaphulo okanye Amathethe (Idioms or Proverbs):** Share culturally relevant idioms or proverbs that connect logically to the topic being discussed. 3. **Ukuvakalisa nokuChwetheza (Pronunciation and Grammar):** Include pronunciation guidance for critical terms, and provide insights into grammar for accurate use of the language.Keep all responses brief, factually accurate, and linguistically clear to enhance learning and comprehension. This prompt assumes all interactions remain fully within the constraints and capabilities of the AI Model Playground.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

    triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[isixhosa_agent, sesotho_agent, afrikaans_agent],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)
    cl.user_session.set("triage_agent", triage_agent)

@cl.on_message
async def on_message(message: cl.Message):  
    triage_agent = cl.user_session.get("triage_agent")
    result = await Runner.run(triage_agent, input=message.content)  # Changed from message.text to message.content
    await cl.Message(
        content=result.final_output,
        author="assistant",
        metadata={"source": "agent"},
    ).send()
    print(result.final_output)