import asyncio
import io
import wave
from datetime import datetime, timedelta

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
       
       
## This will listen to user audio in connected voice channel and save it to a file when user stops speaking for 2 seconds
class VoiceResponse:
    def __init__(self) -> None:
        self.audio_buffers = {}
        self.last_written = {}
        self.filename_counter = 0
        self.new_audio = {}
        self._task = asyncio.create_task(self._background_task())
    
    async def _background_task(self):
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms
            for username in list(self.audio_buffers.keys()):
                if datetime.now() - self.last_written[username] > timedelta(seconds=2) and self.new_audio[username]:
                    # 2 seconds have passed since the last write, stop accumulating
                    filename = f"audio_{username}_{self.filename_counter}.wav"
                    await self.save_to_wav(filename, username)
                    self.filename_counter += 1
                    self.audio_buffers[username].clear()
                    self.new_audio[username] = False
                
    def write(self, username, pcm_audio: bytes):
        if username not in self.audio_buffers:
            self.audio_buffers[username] = bytearray()
            # self._tasks[username] = asyncio.create_task(self._background_task(username))

        self.audio_buffers[username].extend(pcm_audio)
        self.last_written[username] = datetime.now()
        self.new_audio[username] = True
        
    async def save_to_wav(self, filename, username):
        # Convert the audio buffer to bytes
        audio_data = bytes(self.audio_buffers[username])

        # Write the PCM data to a WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)  # stereo
            wav_file.setsampwidth(2)  # 2 bytes = 16 bits
            wav_file.setframerate(48000)  # sample rate
            wav_file.writeframes(audio_data)

        # Clear the audio buffer
        self.audio_buffers[username].clear()



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