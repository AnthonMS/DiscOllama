import asyncio
import functools
import os
import threading
from collections import deque
import time
import json
import uuid
import wave
from datetime import datetime, timedelta
import logging
import numpy as np
import torch
import torchaudio
from discord import FFmpegPCMAudio
from discord import SpeakingState
from discord.ext import voice_recv
from discord import FFmpegPCMAudio
from .misc import *
from datasets import load_dataset


## This will listen to user audio in connected voice channel and save it to a file when user stops speaking for 2 seconds
class VoiceChat:
    def __init__(self, voice_client, discOllama) -> None:
        self.vc = voice_client
        self.discOllama = discOllama
        self.discord = discOllama.discord
        self.ollama = discOllama.ollama
        self.redis = discOllama.redis
        self.stt = discOllama.stt
        self.tts = discOllama.tts
        self.user_audio = {}
        self.last_active = datetime.now()
        self.new_messages = False
        self.new_responses = False
        self.active_transcriptions = 0
        self.transcribe_users = []
        self.responses = []
        
        tts_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embedding = torch.tensor(tts_embeddings_dataset[7306]["xvector"]).unsqueeze(0) # You can replace this embedding with your own as well. But where to find or create these embeddings?
        
        self.transcribe_task = asyncio.create_task(self.transcribing())
        self.respond_task = asyncio.create_task(self.responding())
        self.speaking_task = asyncio.create_task(self.speaking())
        self.ollama_task = None
        self.vc.listen(voice_recv.BasicSink(self.listen))
        

    def listen(self, user, data: voice_recv.VoiceData):
        if user is None:
            return
        
        if self.ollama_task is not None:
            self.ollama_task.cancel()
            
        self.last_active = datetime.now()
        
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
                self.save_to_wav(filename, new_audio_data, channels=1, framerate=16000) # This is just while testing.
                if user.id not in self.transcribe_users: # Add user_id to list so it can be processed
                    self.transcribe_users.append(user.id)
                self.user_audio[user.id]['text'].append(new_audio_data) # Fill spot with audio bytearray so we can transcribe it and switch it out with text later
                # self.user_audio[user.id]['text'].append(filename) # Fill spot with filename so we can put the text in this position later
                self.user_audio[user.id]['processed_audio'].extend(new_audio_data)
                self.user_audio[user.id]['started_speaking'] = None
                
    
    async def transcribing(self):
        while True:
            # Check if there are audio to transcribe and under which user
            if len(self.transcribe_users) > 0:
                self.active_transcriptions += 1
                for i, user_id in enumerate(self.transcribe_users.copy()):
                    # self.user_audio[user_id]
                    # Loop through self.user_audio[user_id]['text] and find the one that are bytearray instead of string and call transcribe audo on that bytearray and switch it out with the result
                    for i, item in enumerate(self.user_audio[user_id]['text']):
                        if isinstance(item, bytearray):
                            start_time = time.time()
                            transcribed_text = await self.transcribe_audio(item)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            logging.info(f"Transcribed text in {elapsed_time:.2f}: {transcribed_text}")
                            
                            if transcribed_text is not None:
                                self.user_audio[user_id]['text'][i] = transcribed_text
                                self.new_messages = True
                            else:
                                self.user_audio[user_id]['text'][i] = ""

                    self.transcribe_users.remove(user_id)
                self.active_transcriptions -= 1
            else:
                await asyncio.sleep(1)
          
           
    async def transcribe_audio(self, audio_data):
        textResult = None
        try:
            asr_data = bytes_to_float32_array(audio_data)
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self.stt, asr_data)
            textResult = text['text']
        except Exception as e:
            logging.error("Error transcribing audio")
            logging.error(e)
        finally:
            if (textResult == None):
                return None
            return textResult
            
            
            
    async def responding(self):
        while True:
            if self.active_transcriptions == 0 and self.new_messages:
                if datetime.now() - self.last_active > timedelta(seconds=1):
                    self.new_messages = False
                    self.save_new_messages() # Call the function when all transcriptions are done and channel is silent for 1 second
            await asyncio.sleep(1)
            
                
    def save_new_messages(self):
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
                self.save_message(item['text'], item['user_id'], item['username'])
            
            # Respond to all newly saved messages or new_messages if no redis
            if self.active_transcriptions == 0:
                self.ollama_task = asyncio.create_task(self.respond(new_messages))
    
    
    async def respond(self, messages=[]):
        try:
            saved_messages = self.get_messages()
            if (len(saved_messages) == 0):
                saved_messages = messages
            if (len(saved_messages) == 0):
                logging.info("No messages to respond to...")
                return
            saved_messages.append({"role": "system", "content": f"You are friendly ai friend chatting in voice chat. Timestamps in messages from users should be ignored unless relevant in conversation. DO NOT FORMAT RESPONSES WITH TIMESTAMPS OR SENT BY!"})
            
            current_model = self.get_current_model()
            full_response = ""
            sentance = ""
            async for part in self.discOllama.chat(saved_messages, milliseconds=None, model=current_model):
                part_content = part['message']['content']
                full_response += part_content
                sentance += part_content
                ## TODO: Check for voice activity, if there is, we should cancel response generation and just skip to the Finally clause
                if (check_end_of_sentance(sentance)):
                    logging.info(f"New sentance: {sentance}")
                    ## TODO: Run in executor to not block heartbeat, but we need to pass multiple parameters
                    loop = asyncio.get_running_loop()
                    tts_partial = functools.partial(self.tts, sentance, forward_params={"speaker_embeddings": self.speaker_embedding})
                    speech = await loop.run_in_executor(None, tts_partial)
                    # speech = self.tts(sentance, forward_params={"speaker_embeddings": self.speaker_embedding})
                    
                    audio_data = np.int16(speech["audio"] * 32767)
                    audio_data = audio_data.tobytes()
                    
                    random_filename = f"audio/responses/{uuid.uuid4()}.wav"
                    self.save_to_wav(random_filename, audio_data, channels=1, framerate=speech["sampling_rate"])
                    self.responses.append({
                        'text': sentance.strip(),
                        'audio': audio_data,
                        'sampling_rate': speech["sampling_rate"],
                        'status': 'ready',
                        'path': random_filename
                    })
                    self.new_responses = True
                    sentance = ""
                
        except asyncio.CancelledError:
            logging.info("Responding cancelled")
        except Exception as e:
            logging.error("Error answering")
            logging.error(e)
        finally:
            logging.info("Full Response: " + full_response)
            self.save_message(full_response, self.discord.user.id)
            self.ollama_task = None
        

    async def speaking(self):
        while True:
            # logging.info(f"Checking for text to make into audio... {self.new_responses} : {str(self.responses)}")
            # # Check if there are responses in self.responses
            # # If there are, then we should start making them into audio that can be played.
            # # self.response is array of dicts that contain text, audio, state: "None","Speaking","Paused"
            # # audio is None unless the text has been made into TTS
            if self.new_responses:
                if not self.vc.is_playing():
                    next_response = self.responses.pop(0)
                    # self.vc.send_audio_packet(next_response['audio'])
                    source = FFmpegPCMAudio(next_response['path'])
                    # source = FFmpegPCMAudio(next_response['audio'])
                    self.vc.play(source)
                
                if len(self.responses) == 0:
                    self.new_responses = False
                
            await asyncio.sleep(0.1)
            

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
    
    def save_message(self, message, user_id, username=""):
        try:
            if user_id is not self.discord.user.id:
                message = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message + "\n\nSent by: " + username
            # else:
            #     message = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n' + message
                
            self.redis.rpush(f"messages:{self.vc.channel.id}", json.dumps({
                "author": user_id,
                "content": message,
            }))
        except Exception as e:
            logging.error(f"Error saving voice message: {e}")
 
            
       
    # async def transcribe_audio_faster(self, audio_data): # Using faster-whisper, this was in fact not faster?
    #     textResult = None
    #     try:
    #         processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    #         compute_type = "float16" if processing_device == "cuda" else "float32"
    #         # Convert the audio data bytes to a numpy array and then to the appropriate type
    #         audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
    #         audio_data = audio_data.astype(compute_type)  # Convert to the appropriate type
            
    #         # Assuming audio_data is a numpy array of the appropriate type
    #         loop = asyncio.get_running_loop()
    #         # segments, info = await loop.run_in_executor(None, self.stt.transcribe, audio_data)
    #         # segments, info = await loop.run_in_executor(None, self.stt.transcribe, audio_data, beam_size=1)
    #         segments, info = await loop.run_in_executor(None, 
    #             lambda: self.stt.transcribe(audio_data, beam_size=1)
    #         )

    #         # Iterate over the segments to get the transcribed text
    #         textResult = " ".join([segment.text for segment in segments])

    #         logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
    #     except Exception as e:
    #         logging.error("Error transcribing audio")
    #         logging.error(e)
    #     finally:
    #         if (textResult == None):
    #             return None
    #         return textResult
        
         
    # async def transcribe_user_audio(self, audio_data, filename, user_id):
    #     self.active_transcriptions += 1
    #     textResult = None
    #     try:
    #         asr_data = bytes_to_float32_array(audio_data)
    #         start_time = datetime.now()
    #         # loop = asyncio.get_event_loop()
    #         # future = loop.run_in_executor(None, self.asr, asr_data)
    #         # text = await future
    #         text = self.asr(asr_data)
    #         textResult = text['text']
    #         end_time = datetime.now()
    #         elapsed_time = end_time - start_time
    #     except asyncio.CancelledError:
    #         pass
    #     except Exception as e:
    #         logging.error("Error transcribing user audio: ")
    #         logging.error(e)
    #     finally:
    #         self.active_transcriptions -= 1
    #         # self.user_audio[user_id]['text']
    #         if (textResult == None):
    #             return

    #         # self.user_audio[user_id]['processed_audio'].extend(audio_data)
    #         # self.user_audio[user_id]['started_speaking'] = None
    #         self.new_messages = True
    #         logging.info(f"Result: {textResult}, Time taken: {elapsed_time.total_seconds():.2f} seconds")
    #         # Replace filename string with text result
    #         for i, str in enumerate(self.user_audio[user_id]['text']):
    #             if str == filename:
    #                 self.user_audio[user_id]['text'][i] = textResult.strip()
    #                 break
                
    #         # if self.active_transcriptions == 0:
    #         #     if datetime.now() - self.last_active > timedelta(seconds=1):
    #         #         self.on_silence() # Call the function when all transcriptions are done and channel is silent for 1 second