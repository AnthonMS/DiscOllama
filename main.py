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
    tts = pyttsx3.init()
    # text = speech_processor('audio/audio_benteb3nt_0.wav')
    # logging.info(text['text'])
    
    disc = discord.Client(intents=intents)
    redis_host = str(os.getenv("REDIS_HOST"))
    redis_port = os.getenv("REDIS_PORT")
    redis_client = False
    if not redis_host == "" and redis_port:
        redis_client = redis.Redis(host=redis_host, port=int(redis_port), db=0, decode_responses=True)
    
    # Ollama initialization
    host = str(os.getenv("OLLAMA_HOST_URL"))
    model = str(os.getenv("OLLAMA_MODEL", "phi"))
    auth_name = os.getenv("BASIC_AUTH_USERNAME")
    auth_pass = os.getenv("BASIC_AUTH_PASSWORD")
    verify_ssl = os.getenv("VERIFY_SSL", "True")
    if verify_ssl.isdigit():
        verify_ssl = bool(int(verify_ssl))
    else:
        verify_ssl = verify_ssl.lower() == "true" # converts to bool
    
    
    llama = False
    if auth_name and auth_pass:
        llama = ollama.AsyncClient(host, auth=(auth_name, auth_pass), verify=verify_ssl)
    else:
        llama = ollama.AsyncClient(host, verify=verify_ssl)
        
    DiscOllama(
        model,
        llama,
        disc,
        redis_client,
        speech_processor,
        tts
    ).run(os.getenv("DISCORD_TOKEN"))