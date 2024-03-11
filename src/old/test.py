import os
from dotenv import load_dotenv
from httpx import BasicAuth
from typing import Final

from ollama import Client, AsyncClient

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')

AUTH_USERNAME: Final[str] = os.getenv("BASIC_AUTH_USERNAME")
AUTH_PASSWORD: Final[str] = os.getenv("BASIC_AUTH_PASSWORD")
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST_URL")

ollama = Client(host=OLLAMA_HOST, auth=BasicAuth(AUTH_USERNAME, AUTH_PASSWORD), verify=False)

# for part in ollama.generate(model='phi', prompt='Why is the sky blue?', stream=True):
#   print(part['response'], end='', flush=True)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in ollama.chat('openhermes:latest', messages=messages, stream=True):
    print(part['message']['content'], end='', flush=True)