import os
from dotenv import load_dotenv
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')
import time
import wave
import numpy as np
import torch


# pip install --upgrade transformers sentencepiece datasets[audio]
from transformers import pipeline
from datasets import load_dataset


def save_to_wav(filename, audio_data, channels=2, sampwidth=2, framerate=48000):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)  # stereo
        wav_file.setsampwidth(sampwidth)  # 2 bytes = 16 bits
        wav_file.setframerate(framerate)  # sample rate
        wav_file.writeframes(audio_data)

text_prompt = "Let's try generating speech with Microsoft T five, a text to speech model."

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# print(f"Count: {len(embeddings_dataset)}") # Count: 7931
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) # You can replace this embedding with your own as well. But where to find or create these embeddings?
start_time = time.time()
speech = synthesiser(text_prompt, forward_params={"speaker_embeddings": speaker_embedding})

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f}")

audio_data_int16 = np.int16(speech["audio"] * 32767)
sampling_rate = speech["sampling_rate"]
save_to_wav("audio/out-t5-2.wav", audio_data_int16, channels=1, framerate=sampling_rate)




# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# start_time = time.time()
# inputs = processor(text=text_prompt, return_tensors="pt")
# # load xvector containing speaker's voice characteristics from a dataset
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f}")
# sf.write("audio/speech2.wav", speech.numpy(), samplerate=16000)
