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

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')

def float32_array_to_bytes(audio_float32):
    audio_int16 = (audio_float32 * 32768).astype(np.int16) # De-normalize from float32 range -1.0 to 1.0 to int16 range
    audio_bytes = audio_int16.tobytes() # Convert int16 numpy array to bytes
    return audio_bytes


def save_to_wav(filename, audio_data, channels=2, sampwidth=2, framerate=48000):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)  # stereo
        wav_file.setsampwidth(sampwidth)  # 2 bytes = 16 bits
        wav_file.setframerate(framerate)  # sample rate
        wav_file.writeframes(audio_data)
        
        
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained("suno/bark-small").to(device)
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
  