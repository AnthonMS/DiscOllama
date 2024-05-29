import asyncio
import io
import re
import wave
from datetime import datetime, timedelta
import logging
import torch
import torchaudio
import numpy as np
from discord import FFmpegPCMAudio
from discord import SpeakingState
from discord.ext import voice_recv

       
       
## This will listen to user audio in connected voice channel and save it to a file when user stops speaking for 2 seconds
class VoiceChat:
    def __init__(self, voice_channel, discord) -> None:
        self.vc = voice_channel
        self.discord = discord
        
        self.test_buffer = bytearray()
        
        self.user_audio = {}
        self.user_texts = {}
        
        self.vc.listen(voice_recv.BasicSink(self.listen))
        
        
    def listen(self, user, data: voice_recv.VoiceData):
        audio_data = self.convert_audio_data(data.pcm)
        self.test_buffer.extend(audio_data)
        
        if user.id not in self.user_audio:
            self.user_audio[user.id] = {'audio': bytearray(), 'last_spoke': datetime.now()}

        self.user_audio[user.id]['audio'].extend(audio_data)
        # # voice power level, how loud the user is speaking
        # ext_data = data.extension_data.get(voice_recv.ExtensionID.audio_power)
        # value = int.from_bytes(ext_data, 'big')
        # power = 127-(value & 127)
        # power = int(power * (79/128))
        # print('#' * power)
    
    async def process_audio(self):
        while True:
            await asyncio.sleep(0.5)
            keys_to_remove = []
            for user_id, userdata in self.user_audio.items():
                if datetime.now() - userdata['last_spoke'] > timedelta(seconds=1):
                    self.save_to_wav(f"audio/{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav", userdata['audio'], channels=1, framerate=16000)
    
            for key in keys_to_remove:
                self.user_audio.pop(key, None)
                
                
    def test(self):
        logging.info("Test from VoiceChat")
        self.save_to_wav("audio/test2.wav", self.test_buffer, channels=1, framerate=16000)
        # self.save_to_wav("audio/test.wav", self.test_buffer, channels=2, framerate=48000)
        
        
    def save_to_wav(self, filename, audio_data, channels=2, sampwidth=2, framerate=48000):
        # Write audio data to a WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(channels)  # stereo
            wav_file.setsampwidth(sampwidth)  # 2 bytes = 16 bits
            wav_file.setframerate(framerate)  # sample rate
            wav_file.writeframes(audio_data)


    def convert_audio_data(self, audio_data) -> bytes:
        speech_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Convert the audio data bytes to a numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        # Convert the stereo audio data to mono by averaging the two channels
        audio_data_mono = audio_data.reshape(-1, 2).mean(axis=1)
        # Convert the numpy array to a PyTorch tensor
        waveform = torch.from_numpy(audio_data_mono).float().to(speech_device)
        # Resample the audio data to 16kHz
        resampler = torchaudio.transforms.Resample(48000, 16000)
        waveform_resampled = resampler(waveform)
        # Convert the resampled waveform to 16-bit integers
        waveform_resampled_int16 = waveform_resampled.numpy().astype(np.int16)
        # Convert the 16-bit integers to bytes
        waveform_resampled_bytes = waveform_resampled_int16.tobytes()
        return waveform_resampled_bytes