import os
from dotenv import load_dotenv
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')
import time

import torchaudio
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN
# pip install speechbrain

text_prompt = "Let's try generating speech with speechbrain, a text to speech model."


# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="pretrained_models/tts-fastspeech2-ljspeech")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")



start_time = time.time()
mel_output, durations, pitch, energy = fastspeech2.encode_text(
    [text_prompt],
    pace=1.0,        # scale up/down the speed
    pitch_rate=1.5,  # scale up/down the pitch
    energy_rate=1.5, # scale up/down the energy
)
# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)
# Save the waverform
torchaudio.save('audio/out-speechbrain.wav', waveforms.squeeze(1), 22050)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f}")


