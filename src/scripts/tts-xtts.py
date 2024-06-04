import os
import time
from dotenv import load_dotenv
import wave
import numpy as np

## For CUDA TO WORK: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
# py -m pip install --upgrade setuptools pip wheel
# py -m pip install nvidia-pyindex
# py -m pip install nvidia-cuda-runtime-cu12

import torch
from transformers import AutoProcessor, BarkModel

from TTS.api import TTS # pip install coqui-tts

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')

device = "cuda" if torch.cuda.is_available() else "cpu"

# # List available 🐸TTS models
# print(TTS().list_models())

print("Initializing TTS...")
# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device) # Simply just freezes and also asks for terms acceptance?
print("TTS initialized!")

# # Run TTS
# # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# # Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="audio/cloner-anthon.wav", language="en")
# # Text to speech to a file
# tts.tts_to_file(text="Hello world!", speaker_wav="audio/cloner-anthon.wav", language="en", file_path="audio/output-anthon.wav")

