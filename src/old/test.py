import os
import time
from dotenv import load_dotenv
from httpx import BasicAuth
from typing import Final
import wave
import numpy as np
from ollama import Client, AsyncClient
# from IPython.display import Audio # pip install ipythons

## For CUDA TO WORK: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
# py -m pip install --upgrade setuptools pip wheel
# py -m pip install nvidia-pyindex
# py -m pip install nvidia-cuda-runtime-cu12
# 

import torch
from transformers import pipeline, AutoModel, AutoProcessor, BarkModel, set_seed

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')

def float32_array_to_bytes(audio_float32):
    # De-normalize from float32 range -1.0 to 1.0 to int16 range
    audio_int16 = (audio_float32 * 32768).astype(np.int16)
    # Convert int16 numpy array to bytes
    audio_bytes = audio_int16.tobytes()
    return audio_bytes


def save_to_wav(filename, audio_data, channels=2, sampwidth=2, framerate=48000):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)  # stereo
        wav_file.setsampwidth(sampwidth)  # 2 bytes = 16 bits
        wav_file.setframerate(framerate)  # sample rate
        wav_file.writeframes(audio_data)
        
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small").to("device")
# model =  model.to_bettertransformer() # pip install optimum
# model.enable_cpu_offload()
processor = AutoProcessor.from_pretrained("suno/bark-small", torch_dtype=torch.float16)
text_prompt = "Let's try generating speech, with Bark, a text-to-speech model."
start_time = time.time()
inputs = processor(text_prompt, voice_preset="v2/en_speaker_6")
output = model.generate(**inputs)
# output = model.generate(**inputs, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f}")
audio_data = output.detach().cpu().numpy()
audio_data_int16 = np.int16(audio_data * 32767)
sampling_rate = model.generation_config.sample_rate
save_to_wav("audio/output.wav", audio_data_int16, channels=1, framerate=sampling_rate)
  
# # suno/bark-small
# # fishaudio/fish-speech-1
# # microsoft/speecht5_tts (pip install sentencepiece)
# tts = pipeline("text-to-speech", model="suno/bark-small")
# # tts = AutoModel.from_pretrained("myshell-ai/MeloTTS-English")
# text = "This is a test... and I just took a long pause."
# output = tts(text)
# for key in output.keys():
#     print(f"{key}")

# audio_data = float32_array_to_bytes(output['audio'])
# if audio_data is not None:
#     save_to_wav("audio/test.wav", audio_data, framerate=output["sampling_rate"], channels=2)
#     print(f"Output: {output}")
# else:
#     print("No audio data received?")
#     print(f"Output: {output}")
    
    
    
    
    
    
    
# whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium')
# text = whisper('audio/audio_benteb3nt_0.wav')
# print(text)


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