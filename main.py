import logging
import os
import redis
import ollama
import discord
from dotenv import load_dotenv
from src.DiscOllama import DiscOllama
import torch
import pyttsx3
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

load_dotenv(f'{THIS_PATH}\\.env')
logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True
    
    # speech_model_id = "openai/whisper-medium"
    # # self.speech_model_id = "openai/whisper-large-v3"
    # speech_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # speech_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     speech_model_id, torch_dtype=speech_torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    # )
    # speech_model.to(speech_device)
    # speech_processor = AutoProcessor.from_pretrained(speech_model_id)
    speech_processor = pipeline('automatic-speech-recognition', model='openai/whisper-medium')
    # text = speech_processor('audio/audio_benteb3nt_0.wav')
    # logging.info(text['text'])
        
    DiscOllama(
        str(os.getenv("OLLAMA_MODEL", "phi")),
        ollama.AsyncClient(host=os.getenv("OLLAMA_HOST_URL"), auth=(os.getenv("BASIC_AUTH_USERNAME"), os.getenv("BASIC_AUTH_PASSWORD")), verify=False),
        discord.Client(intents=intents),
        redis.Redis(host=str(os.getenv("REDIS_HOST")), port=int(os.getenv("REDIS_PORT")), db=0, decode_responses=True),
        speech_processor=speech_processor,
        tts=pyttsx3.init()
    ).run(os.getenv("DISCORD_TOKEN"))