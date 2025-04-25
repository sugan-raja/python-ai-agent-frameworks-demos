import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a language tutor who is well versed in Afrikaans, isiXhosa, and Sesotha." \
        "Please provide spelling rules for given word in english and provide explanation using idioms and proverbs."),
        UserMessage(content="What is the staple food in south africa?"),
    ],
    model=os.getenv("GITHUB_MODEL", "gpt-4.1"),
)
print(response.choices[0].message.content)
