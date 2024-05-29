# import asyncio
import io
# import re
# import wave
# from datetime import datetime, timedelta
# import logging
# import torch
# import torchaudio
import numpy as np
from discord import FFmpegPCMAudio
from discord.ext import voice_recv

class Response:
    def __init__(self, message):
        self.message = message
        self.channel = message.channel
        self.author = message.author

        self.r = None
        self.sb = io.StringIO()

    async def write(self, s, end=''):
        if self.sb.seek(0, io.SEEK_END) + len(s) + len(end) > 2000:
            self.r = None
            self.sb.seek(0, io.SEEK_SET)
            self.sb.truncate()

        self.sb.write(s)

        value = self.sb.getvalue().strip()
        if not value:
            return
            
        if self.r:
            await self.r.edit(content=value + end)
            return

        if self.channel:
            self.r = await self.channel.send(value)
        elif self.author:
            self.r = await self.author.send(value)
       
       
       
       
       
       
# ## This will listen to user audio in connected voice channel and save it to a file when user stops speaking for 2 seconds
# class VoiceResponse:
#     def __init__(self, voice_channel, discOllama) -> None:
#         self.voice_channel = voice_channel
#         self.discOllama = discOllama
#         self.speech_device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         # self.speech_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#         self.audio_buffers = {} # Each userkey will have: status, audio_buffer, last_written
#         self.filename_counter = 0
#         self._task_incoming_audio = asyncio.create_task(self._task_incoming())
#         self._task_outgoing_audio = asyncio.create_task(self._task_outgoing())
#         self.sb = io.StringIO()
#         self.user_speaking = False
#         self.responding = False
#         self.speaking_task = None
#         self.responding_task = None
#         self.no_of_responses = 0
#         self.responses = []
        
#         self.voice_channel.listen(voice_recv.BasicSink(self.user_speak))
        
        
#     def user_speak(self, user, data: voice_recv.VoiceData):
#         self.user_speaking = True
#         if self.voice_channel.is_playing():
#             self.voice_channel.pause()
#             # self.responses = []
            
#         try:
#             self.responding_task.cancel()
#         except Exception as e:
#             pass
#         try:
#             self.speaking_task.cancel()
#         except Exception as e:
#             pass
        
#         # # voice power level, how loud the user is speaking
#         # ext_data = data.extension_data.get(voice_recv.ExtensionID.audio_power)
#         # value = int.from_bytes(ext_data, 'big')
#         # power = 127-(value & 127)
#         # power = int(power * (79/128))
#         # print('#' * power)
        
#         if user not in self.audio_buffers:
#             self.audio_buffers[user] = [{'status': 'incoming', 'audio_buffer': bytearray(), 'last_written': datetime.now()}]

#         try:
#             # Find the last audio buffer with status 'incoming', or create a new one if none exists
#             audio_buffer = next((buffer for buffer in reversed(self.audio_buffers[user]) if buffer['status'] == 'incoming'), None)
#             if audio_buffer is None:
#                 audio_buffer = {'status': 'incoming', 'audio_buffer': bytearray(), 'last_written': datetime.now()}
#                 self.audio_buffers[user].append(audio_buffer)

#             audio_buffer['audio_buffer'].extend(data.pcm)
#             audio_buffer['last_written'] = datetime.now()
#         except Exception as e:
#             logging.error(f"Error in user_speak")
#             logging.error(e)   
          
            
#     async def _task_incoming(self):
#         while True:
#             await asyncio.sleep(0.5)  # Check every 100ms
#             self.process_incoming_audio()
#             self.create_response()
            
#     async def _task_outgoing(self):
#         while True:
#             await asyncio.sleep(0.5)
#             if self.voice_channel.is_paused() and not self.user_speaking and len(self.responses) > 0:
#                 self.voice_channel.resume()
#             elif not self.voice_channel.is_playing() and not self.user_speaking and len(self.responses) > 0:
#                 self.speak()
           
    
#     def speak(self):
#         try:
#             # response = self.responses.pop(0)
#             response = self.responses[0]
#             source = FFmpegPCMAudio(response)
#             self.voice_channel.play(source, after=self.speak_after)
#         except asyncio.CancelledError:
#             pass
    
#     def speak_after(self, error):
#         if len(self.responses) > 0:
#             self.responses.pop(0)
#         if not self.user_speaking and len(self.responses) > 0:
#             self.speak()
    
#     def create_response(self):
#         if not self.responding and not self.user_speaking:
#             # No one is speaking, so it's time to create our response
#             processed_audio_buffers = []
#             for user in list(self.audio_buffers.keys()):
#                 # Extract all audio_buffer dictionaries from under any user that has the status "processed"
#                 processed_buffers = [audio_buffer_dict for audio_buffer_dict in self.audio_buffers[user] if audio_buffer_dict['status'] == 'processed']
#                 processed_audio_buffers.extend(processed_buffers)
#                 # Remove all audio_buffer dictionaries from under any user that has the status "processed" or "noise"
#                 self.audio_buffers[user] = [audio_buffer_dict for audio_buffer_dict in self.audio_buffers[user] if audio_buffer_dict['status'] not in ['processed', 'noise']]
            
#             if len(processed_audio_buffers) > 0:
#                 try:
#                     self.responding_task.cancel()
#                 except Exception as e:
#                     pass
#                 # Create a new responding task
#                 self.responding_task = asyncio.create_task(self.respond_new())
    
#     def process_incoming_audio(self):
#         for user in list(self.audio_buffers.keys()):
#             for audio_buffer_dict in self.audio_buffers[user]:
#                 if audio_buffer_dict['status'] == 'incoming' and datetime.now() - audio_buffer_dict['last_written'] > timedelta(seconds=1):
#                     self.user_speaking = False
#                     audio_buffer_dict['status'] = 'processing'
#                     filename = f"audio/audio_{user}_{self.filename_counter}.wav"
#                     audio_buffer_dict['filename'] = filename
#                     self.filename_counter += 1
#                     audio_process = asyncio.create_task(self.process_audio(audio_buffer_dict, user))
#                     audio_buffer_dict['task'] = audio_process
#                     break  # Only process one audio buffer at a time for each user
         
#     async def process_audio(self, audio_buffer_dict, user):
#         loop = asyncio.get_event_loop()
#         audio_buffer = audio_buffer_dict['audio_buffer']
        
#         # Calculate the duration of the audio clip
#         num_samples = len(audio_buffer) // 2  # Each sample is 2 bytes
#         duration = num_samples / (48000 * 2)  # Sample rate is 48000 Hz and 2 channels
#         if duration < 0.5:
#             return
        
#         audio_data = self.convert_audio_data(audio_buffer)
#         await self.save_to_wav(audio_buffer_dict['filename'], audio_data, channels=1, sampwidth=2, framerate=16000)
        
#         # Convert bytearray to numpy array
#         # audio_buffer_np = np.frombuffer(self.audio_buffers[user], dtype=np.int16)
        
#         future = loop.run_in_executor(None, self.discOllama.speech_processor, audio_buffer_dict['filename'])
#         text = await future
#         textResult = text['text'].strip()
#         if len(text['text'].strip().split()) > 0:
#             tmpStr = re.sub(r'[^\w\s]', '', textResult.lower()) # Keep alphanumeric characters and whitespace
#             if (len(tmpStr.split()) < 10):
#                 stop_percentage = self.stop_command_percentage(tmpStr)
#                 if stop_percentage >= 75 and self.voice_channel.is_playing():
#                     self.responses = []
#                     self.voice_channel.stop()
#                     audio_buffer_dict['status'] = 'noise'
#                     return
#                 # else:
#             self.discOllama.save_voice_message(self.voice_channel.channel.id, text['text'], user)
#             audio_buffer_dict['status'] = 'processed'
#         else:
#             audio_buffer_dict['status'] = 'noise'
    
    
#     async def respond_new(self):
#         self.responding = True
#         full_response = ""
#         partial_response = ""
#         try:
#             chat_history = await self.discOllama.get_voice_messages(self.voice_channel.channel.id)
#             async for part in self.discOllama.chat(chat_history, milliseconds=None, model=self.discOllama.model_voice):
#                 # print(part['message']['content'])
#                 part_content = part['message']['content']
#                 full_response += part_content
#                 partial_response += part_content
                
#                 # Check if partial_response contains an end sentence indicator
#                 sentences = re.split('(?<=[.!?]) +', partial_response)
#                 for i in range(len(sentences) - 1):
#                     sentence = sentences[i]
#                     self.no_of_responses += 1
#                     self.text_to_speech(sentence)  # Call text_to_speech for each sentence
#                 partial_response = sentences[-1]  # Keep the last part (it doesn't end with an end sentence indicator)

#             self.no_of_responses += 1
#             self.text_to_speech(partial_response)  # Call text_to_speech for each sentence
                    
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             logging.error(f"Error generating response")
#             logging.error(e)
#             pass
#         finally:
#             full_response = full_response.strip() 
#             if full_response:  # Check if full_response is not empty
#                 self.discOllama.save_voice_response(self.voice_channel.channel.id, full_response)
#             # self.discOllama.save_voice_response(self.voice_channel.channel.id, full_response)
#             self.responding = False
    
#     def text_to_speech(self, text):
#         filename = f"audio/response_{self.no_of_responses}.wav"
#         self.responses.append(filename)
#         self.discOllama.tts.save_to_file(text, filename)
#         self.discOllama.tts.runAndWait()
    
        
#     async def save_to_wav(self, filename, audio_data, channels=2, sampwidth=2, framerate=48000):
#         # Write audio data to a WAV file
#         with wave.open(filename, 'wb') as wav_file:
#             wav_file.setnchannels(channels)  # stereo
#             wav_file.setsampwidth(sampwidth)  # 2 bytes = 16 bits
#             wav_file.setframerate(framerate)  # sample rate
#             wav_file.writeframes(audio_data)


#     def convert_audio_data(self, audio_buffer) -> bytes:
#         # Convert the audio data bytes to a numpy array
#         audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
#         # Convert the stereo audio data to mono by averaging the two channels
#         audio_data_mono = audio_data.reshape(-1, 2).mean(axis=1)
#         # Convert the numpy array to a PyTorch tensor
#         waveform = torch.from_numpy(audio_data_mono).float().to(self.speech_device)
#         # Resample the audio data to 16kHz
#         resampler = torchaudio.transforms.Resample(48000, 16000)
#         waveform_resampled = resampler(waveform)
#         # Convert the resampled waveform to 16-bit integers
#         waveform_resampled_int16 = waveform_resampled.numpy().astype(np.int16)
#         # Convert the 16-bit integers to bytes
#         waveform_resampled_bytes = waveform_resampled_int16.tobytes()
#         return waveform_resampled_bytes

        
#     def stop_command_percentage(self, text):
#         stop_commands = ["stop", "quiet", "silence", "talking", "please", "now", "shut", "up", "fuck"]
#         words = text.split()
#         matching_words = [word for word in words if word in stop_commands]
#         if len(words) == 0:
#             return 0
#         else:
#             return len(matching_words) / len(words) * 100
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# # Convert the audio data bytes to a numpy array
# audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
# print("Extract speech as text from audio buffer 2")

# # Convert the numpy array to a PyTorch tensor
# waveform = torch.from_numpy(audio_data).float().to(self.speech_device)
# print("Extract speech as text from audio buffer 3")

# # Use the speech processor to convert the audio data to IDs
# inputs = self.speech_processor(waveform, sampling_rate=16000, return_tensors="pt")
# print("Extract speech as text from audio buffer 4")

# # Use the speech processor to convert the IDs to text
# text = self.speech_processor.decode(inputs["input_values"][0])
# print("Extract speech as text from audio buffer 5")

# return text
# print("Extract speech as text from audio buffer 1")

# # Convert the audio data bytes to a numpy array
# audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
# print("Extract speech as text from audio buffer 2")

# # Convert the stereo audio data to mono by averaging the two channels
# audio_data_mono = audio_data.reshape(-1, 2).mean(axis=1)
# print("Extract speech as text from audio buffer 3")

# # Convert the numpy array to a PyTorch tensor
# waveform = torch.from_numpy(audio_data_mono).float().to(self.speech_device)
# print("Extract speech as text from audio buffer 4")

# # Resample the audio data to 16kHz
# resampler = torchaudio.transforms.Resample(48000, 16000)
# waveform_resampled = resampler(waveform)
# print("Extract speech as text from audio buffer 5")

# # Use the speech processor to convert the audio data to IDs
# inputs = self.speech_processor(waveform_resampled, sampling_rate=16000, return_tensors="pt")
# print("Extract speech as text from audio buffer 6")

# # Use the speech processor to convert the IDs to text
# text = self.speech_processor.decode(inputs["input_values"][0])
# print("Extract speech as text from audio buffer 7")


# return text
# # Convert stereo audio to mono by averaging the channels
# audio_mono = np.mean(np.reshape(audio_buffer, (-1, 2)), axis=1)
# print("Extract speech as text from audio buffer 2")

# # Convert the audio data to a tensor
# audio_tensor = torch.tensor(audio_mono, dtype=self.speech_torch_dtype).to(self.speech_device)
# print("Extract speech as text from audio buffer 3")

# # Use the speech processor to convert the audio data to IDs
# inputs = self.speech_processor(audio_tensor, sampling_rate=48000, return_tensors="pt")
# print("Extract speech as text from audio buffer 4")

# # Use the speech processor to convert the IDs to text
# text = self.speech_processor.decode(inputs["input_values"][0])
# print("Extract speech as text from audio buffer 5")

# return text

## Single User Version
# class VoiceResponse:
#     def __init__(self) -> None:
#         self.audio_buffer = bytearray()
#         self.new_audio = True
#         self.last_written = datetime.now()
#         self.filename_counter = 0
#         self._task = asyncio.create_task(self._background_task())
    
#     async def _background_task(self):
#         while True:
#             await asyncio.sleep(0.1)  # Check every 100ms
#             if datetime.now() - self.last_written > timedelta(seconds=2) and self.new_audio:
#                 # 2 seconds have passed since the last write, stop accumulating
#                 filename = f"audio_{self.filename_counter}.wav"
#                 await self.save_to_wav(filename)
#                 self.filename_counter += 1
#                 self.audio_buffer.clear()
#                 self.new_audio = False
                
                
#     def write(self, pcm_audio: bytes):
#         self.audio_buffer.extend(pcm_audio)
#         self.last_written = datetime.now()
#         self.new_audio = True
        
#     async def save_to_wav(self, filename):
#         # Convert the audio buffer to bytes
#         audio_data = bytes(self.audio_buffer)

#         # Write the PCM data to a WAV file
#         with wave.open(filename, 'wb') as wav_file:
#             wav_file.setnchannels(2)  # stereo
#             wav_file.setsampwidth(2)  # 2 bytes = 16 bits
#             wav_file.setframerate(48000)  # sample rate
#             wav_file.writeframes(audio_data)

#         # Clear the audio buffer
#         self.audio_buffer.clear()