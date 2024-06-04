import os
from dotenv import load_dotenv
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')
import time

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
# pip install git+https://github.com/huggingface/parler-tts.git

text_prompt = "Let's try generating speech with parler, a text-to-speech model."
description = "Thomas speaks at a normal speed in a happy tone with emphasis and high quality audio."

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)

# model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
# tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)

start_time = time.time()
prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
set_seed(42)
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("audio/out-parler-expresso.wav", audio_arr, model.config.sampling_rate)
# prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
# generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("audio/out-parler.wav", audio_arr, model.config.sampling_rate)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f}")



# Specify the name of a male speaker (Jerry, Thomas) or female speaker (Talia, Elisabeth) for consistent voices
# The model can generate in a range of emotions, including: "happy", "confused", "default" (meaning no particular emotion conveyed), "laughing", "sad", "whisper", "emphasis"
# Include the term "high quality audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
# Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech
# To emphasise particular words, wrap them in asterisk (e.g. *you* in the example above) and include "emphasis" in the prompt
