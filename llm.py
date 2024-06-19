from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def prompt_llm_with_stream(messages, model):
    stream = client.chat.completions.create(
    model = model,
    messages = messages,
    stream=True,
)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
                yield chunk.choices[0].delta.content


def prompt_llm(messages, model):
    response = ""

    for chunk in prompt_llm(messages, model):
        response += chunk
        print(chunk, end="")
    
    
    br = '\n' + '-' * 20 + '\n'
    with open('chat.txt', 'a') as f:
        f.write(f"===PROMPT===\n{json.dumps(messages, indent=2)}{br}===RESPONSE ({model})===:\n{response}{br}")
    return response


def wrap_message(role, content):
    return {
        'role': role,
        'content': content
    }