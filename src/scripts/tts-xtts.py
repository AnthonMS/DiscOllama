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
import sys
sys.path.append(r'C:\Users\Antho\Projects\hugging-face\XTTS-v2')

from TTS.tts.configs.xtts_config import XttsConfig

# THIS_PATH = os.path.dirname(os.path.realpath(__file__))
# load_dotenv(f'{THIS_PATH}\\.env')

