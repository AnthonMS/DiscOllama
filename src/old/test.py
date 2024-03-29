import os
from dotenv import load_dotenv
from httpx import BasicAuth
from typing import Final

from ollama import Client, AsyncClient

import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')



whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium')

text = whisper('audio/audio_benteb3nt_0.wav')

print(text)


# speech_model_id = "openai/whisper-medium"
# # self.speech_model_id = "openai/whisper-large-v3"
# speech_device = "cuda:0" if torch.cuda.is_available() else "cpu"
# speech_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
#   	speech_model_id, torch_dtype=speech_torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# speech_model.to(speech_device)
# speech_processor = AutoProcessor.from_pretrained(speech_model_id)





# AUTH_USERNAME: Final[str] = os.getenv("BASIC_AUTH_USERNAME")
# AUTH_PASSWORD: Final[str] = os.getenv("BASIC_AUTH_PASSWORD")
# OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST_URL")

# ollama = Client(host=OLLAMA_HOST, auth=BasicAuth(AUTH_USERNAME, AUTH_PASSWORD), verify=False)

# # for part in ollama.generate(model='phi', prompt='Why is the sky blue?', stream=True):
# #   print(part['response'], end='', flush=True)

# messages = [
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ]

# for part in ollama.chat('openhermes:latest', messages=messages, stream=True):
#     print(part['message']['content'], end='', flush=True)