import logging
import os
import redis
import ollama
import discord
from dotenv import load_dotenv
from src.DiscOllama import DiscOllama
import torch
import pyttsx3
from transformers import pipeline
# from faster_whisper import WhisperModel # pip install faster-whisper / python3 -m pip install -U faster-whisper


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

load_dotenv(f'{THIS_PATH}\\.env')
logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

os.environ["OMP_NUM_THREADS"] = "4"

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
    
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if processing_device == "cuda" else "float32"
    # stt = WhisperModel("small", device=processing_device, compute_type=compute_type)
    # model = WhisperModel("large-v3", device="cpu", compute_type="float32")
    stt = pipeline('automatic-speech-recognition', model='openai/whisper-small') # whisper-medium
    tts = pipeline("text-to-speech", "microsoft/speecht5_tts")
    
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
        stt,
        tts
    ).run(os.getenv("DISCORD_TOKEN"))