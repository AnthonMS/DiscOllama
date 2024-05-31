import asyncio
import os
import threading
from collections import deque
import time
import json
import wave
from datetime import datetime, timedelta
import logging
import numpy as np
import torch
import torchaudio
from discord import FFmpegPCMAudio
from discord import SpeakingState
from discord.ext import voice_recv
from .misc import *




loop_handler = AsyncLoopThread()
response_handler = AsyncLoopThread()


## This will listen to user audio in connected voice channel and save it to a file when user stops speaking for 2 seconds
class VoiceChat:
    def __init__(self, voice_client, discOllama) -> None:
        self.vc = voice_client
        self.discOllama = discOllama
        self.discord = discOllama.discord
        self.ollama = discOllama.ollama
        self.redis = discOllama.redis
        self.asr = discOllama.asr
        
        self.last_active = datetime.now()
        
        self.new_messages = False
        self.silence_detected = False
        self.user_audio = {}
        self.active_transcriptions = 0
        self.res_thread = None
        
        # self.res_task = asyncio.create_task(self.responding())
        
        self.vc.listen(voice_recv.BasicSink(self.listen))
        
    def listen(self, user, data: voice_recv.VoiceData):
        if user is None:
            return
        
        # loop_handler.stop_loop()
        self.last_active = datetime.now()
        self.silence_detected = False
        
        audio_data = self.convert_audio_data(data.pcm)
        if user.id not in self.user_audio:
            self.user_audio[user.id] = {
                'username': user.name,
                'last_spoke': datetime.now(), 
                'started_speaking': datetime.now(),
                'audio': bytearray(), 
                'processed_audio': bytearray(), 
                'text': []
            }
            
        if self.user_audio[user.id]['started_speaking'] == None:
            self.user_audio[user.id]['started_speaking'] = datetime.now()

        self.user_audio[user.id]['audio'].extend(audio_data)
        self.user_audio[user.id]['last_spoke'] = datetime.now()
        time_speaking = self.user_audio[user.id]['last_spoke'] - self.user_audio[user.id]['started_speaking']
        
        if time_speaking >= timedelta(milliseconds=500):
            # keys_to_process.append(user_id)
            audio_data = self.user_audio[user.id]['audio']
            processed_data = self.user_audio[user.id]['processed_audio']
            new_audio_data = audio_data[len(processed_data):]
            if len(new_audio_data) < 6400:
                return # Audio less than 200ms, wait for more audio data
            
            if is_silence(new_audio_data[-6400:], 5): # Past 200ms is very silent
                if is_silence(new_audio_data, 10): # Whole audio is silent
                    return
                
                filename = f"audio/{user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                self.user_audio[user.id]['text'].append(filename) # Fill spot with filename so we can put the text in this position later
                self.save_to_wav(filename, new_audio_data, channels=1, framerate=16000) # This is just while testing.
                self.user_audio[user.id]['processed_audio'].extend(new_audio_data)
                self.user_audio[user.id]['started_speaking'] = None
                
                ## Maybe instead of trying to start a transcription task or do it from here. We add it to a queue and then have another task checking for queued audio_data?
                # loop_handler.run_coroutine(self.transcribe_user_audio(new_audio_data, filename, user.id)) # This runs it as a loop. I just need to run it once. AI Made it lol.
                # self.transcribe_user_audio(new_audio_data, filename, user.id) # This blocks it from listening until it's done transcribing
                
                # # self.test_task = asyncio.create_task(self.transcribe_user_audio(new_audio_data, filename, user.id)) ## Error: No running event loop
            
    async def transcribe_user_audio(self, audio_data, filename, user_id):
        self.active_transcriptions += 1
        textResult = None
        try:
            asr_data = bytes_to_float32_array(audio_data)
            start_time = datetime.now()
            # loop = asyncio.get_event_loop()
            # future = loop.run_in_executor(None, self.asr, asr_data)
            # text = await future
            text = self.asr(asr_data)
            textResult = text['text']
            end_time = datetime.now()
            elapsed_time = end_time - start_time
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("Error transcribing audio: ")
            logging.error(e)
        finally:
            self.active_transcriptions -= 1
            # self.user_audio[user_id]['text']
            if (textResult == None):
                return

            # self.user_audio[user_id]['processed_audio'].extend(audio_data)
            # self.user_audio[user_id]['started_speaking'] = None
            self.new_messages = True
            logging.info(f"Result: {textResult}, Time taken: {elapsed_time.total_seconds():.2f} seconds")
            # Replace filename string with text result
            for i, str in enumerate(self.user_audio[user_id]['text']):
                if str == filename:
                    self.user_audio[user_id]['text'][i] = textResult.strip()
                    break
                
            # if self.active_transcriptions == 0:
            #     if datetime.now() - self.last_active > timedelta(seconds=1):
            #         self.on_silence() # Call the function when all transcriptions are done and channel is silent for 1 second
 
    def on_silence(self):
        text_items = []
        new_messages = []
        for user_id, userdata in self.user_audio.items():
            if len(userdata['text']) > 0:
                text = ' '.join(userdata['text'])
                text_item = {
                    'user_id': user_id,
                    'username': userdata['username'],
                    'text': text,
                    'time': userdata['last_spoke']
                }
                text_items.append(text_item)
                new_messages.append({
                    "role": "assistant" if user_id == self.discord.user.id else "user",
                    "content": text,
                })
                userdata['text'] = []
                userdata['started_speaking'] = None
                userdata['processed_audio'] = bytearray()
                userdata['audio'] = bytearray()
        
        if (len(text_items) > 0):
            text_items.sort(key=lambda x: x['time'])
            for item in text_items:
                logging.info(f"{item['time']}: User {item['user_id']}: {item['text']}")
                # Save in redis!
                self.save_message(item['text'], item['user_id'], item['username'])
            
            # Respond to all newly saved messages
            # await self.respond(new_messages)
            if self.active_transcriptions == 0:
                # await self.respond(new_messages)
                loop_handler.run_coroutine(self.respond(new_messages))
                # response_handler.run_coroutine(self.respond(new_messages))
                # new_loop = asyncio.new_event_loop()
                # thread = threading.Thread(target=start_async_loop, args=(self.respond(new_messages), new_loop))
                # thread.start()
                # thread.daemon = True
    
    async def responding(self):
        while True:
            if self.active_transcriptions == 0 and self.new_messages:
                if datetime.now() - self.last_active > timedelta(seconds=1):
                    self.new_messages = False
                    logging.info("RESPONDING....")
                    self.on_silence() # Call the function when all transcriptions are done and channel is silent for 1 second
            await asyncio.sleep(1)
    
    async def respond(self, messages=[]):
        try:
            saved_messages = self.get_messages()
            if (len(saved_messages) == 0):
                saved_messages = messages
            if (len(saved_messages) == 0):
                logging.info("No messages to respond to...")
                return
            logging.info("Responding to messages: " + str(saved_messages))
            
            current_model = self.get_current_model()
            full_response = ""        # ollama, model, messages
            # async for part in self.ollama.chat(current_model, messages=saved_messages, stream=True):
            # chat_iterator = await self.ollama.chat(current_model, messages=saved_messages, stream=True)
            # async for part in chat_iterator:
            async for part in self.discOllama.chat(saved_messages, milliseconds=None, model=current_model):
                part_content = part['message']['content']
                # logging.info(f"Part: {part_content}")
                full_response += part_content
                
            logging.info("Full response: " + full_response)
                    
            # await response.write('')
        except asyncio.CancelledError:
            logging.info("Responding cancelled")
        except Exception as e:
            logging.error("Error answering")
            logging.error(e)
        finally:
            # logging.info("FINALLY RESPOND: " + full_response)
            pass
        

    async def test(self):
        logging.info("Test from VoiceChat")
        # logging.info(f"TESTER: {str(self.user_audio)}")
        saved_messages = self.get_messages()
        if (len(saved_messages) == 0):
            logging.info("No messages to respond to...")
            return
        logging.info("Responding to messages: " + str(saved_messages))
        
        current_model = self.get_current_model()
        full_response = ""
        async for part in self.discOllama.chat(saved_messages, milliseconds=None, model=current_model):
            part_content = part['message']['content']
            logging.info(f"Part: {part_content}")
            full_response += part_content
            
        # response = await self.ollama.chat(current_model, messages=saved_messages, stream=False)
        # full_response = response['message']['content']
        
        # async for part in self.discOllama.chat(saved_messages, milliseconds=None, model=current_model):
        # async for part in self.ollama.chat(current_model, messages=saved_messages, stream=True):
        # chat_iterator = await self.ollama.chat(current_model, messages=saved_messages, stream=True)
        # async for part in chat_iterator:
        #     part_content = part['message']['content']
        #     logging.info(f"Part: {part_content}")
        #     full_response += part_content
        logging.info("Full response: " + full_response)
        
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
    
    
    def get_current_model(self):
        try:
            model_string = self.redis.get(f"model:{self.vc.channel.id}")
            return model_string if model_string else str(os.getenv("OLLAMA_MODEL", "phi"))
        except Exception as e:
            logging.error(f"Error getting current model in voice channel: {e}")
            return []
        
    
    def get_messages(self):
        try:
            messages = self.redis.lrange(f"messages:{self.vc.channel.id}", 0, -1)
            # Convert the messages from JSON format to Python dictionaries
            messages = [json.loads(msg) for msg in messages]
            messages = [
                    {"role": "assistant" if msg["author"] == self.discord.user.id else "user","content": msg["content"]}
                for msg in messages.copy()
            ]
            return messages
        except Exception as e:
            logging.error(f"Error getting voice messages: {e}")
            return []
            
            
        
    
    def save_message(self, message, user_id, username):
        try:
            if user_id is not self.discord.user.id:
                message = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message + "\n\nSent by: " + username
                
            self.redis.rpush(f"messages:{self.vc.channel.id}", json.dumps({
                "author": user_id,
                "content": message,
            }))
        except Exception as e:
            logging.error(f"Error saving voice message: {e}")