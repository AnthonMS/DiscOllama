import asyncio
from collections import deque
import logging
import threading
import time
import numpy as np
import torchaudio
import torch
import io
from datetime import datetime, timedelta


def check_end_of_sentance(str):
    end_of_sentence_chars = ['.', '!', '?']
    if str and str[-1] in end_of_sentence_chars:
        return True
    return False


def start_async_loop(coro, loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()


def float32_array_to_bytes(audio_float32):
    # De-normalize from float32 range -1.0 to 1.0 to int16 range
    audio_int16 = (audio_float32 * 32768).astype(np.int16)
    # Convert int16 numpy array to bytes
    audio_bytes = audio_int16.tobytes()
    return audio_bytes

def bytes_to_float32_array(audio_bytes):
    # Convert bytes to int16 numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    # Normalize to float32 range -1.0 to 1.0
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32

def preprocess_audio(audio_data):
    # Assuming audio_data is a bytes object containing 16-bit PCM audio
    waveform = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0  # Normalize to [-1.0, 1.0]
    waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)  # Adjust if original rate differs
    waveform = resampler(waveform)
    return waveform.numpy()  # Convert to NumPy array for compatibility with the ASR model

async def react_thinking(message, user=False):
    if not user:
        await message.add_reaction('🤔')
    else:
        await message.remove_reaction('🤔', user)

async def react_poop(message, user=False):
    if not user:
        await message.add_reaction('💩')
    else:
        await message.remove_reaction('💩', user)
        
async def react_no(message, user=False):
    if not user:
        await message.add_reaction('❌')
    else:
        await message.remove_reaction('❌', user)
        
async def react_ok(message, user=False):
    if not user:
        await message.add_reaction('👌')
    else:
        await message.remove_reaction('👌', user)
        
async def react_test(message, user=False):
    if not user:
        await message.add_reaction('🚫')
    else:
        await message.remove_reaction('🚫', user)
        
# 🙅‍♂️



async def get_model_list(ollama):
    model_list = await ollama.list()
    available_models = []
    for model in model_list['models']:
        available_models.append(f"{model['name']}")
    return available_models


def is_silence(audio_data, threshold=20):
    """Determine if the given audio segment is silence based on volume threshold."""
    if audio_data is None or len(audio_data) == 0:
        logging.error("Audio data is None or empty.")
        return True  # Assuming silence if no data is provided

    if not isinstance(audio_data, (bytes, bytearray)):
        logging.error(f"Unexpected data type for audio_data: {type(audio_data)}")
        return False  # Assuming silence if data type is incorrect

    # Convert bytes to int16 numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Check if the array is empty after conversion
    if audio_array.size == 0:
        logging.error("Converted audio array is empty.")
        return False 

    # Calculate Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.square(audio_array)))

    # Check for NaN values which could arise from invalid operations
    if np.isnan(rms):
        logging.info("RMS calculation resulted in NaN.")
        return False  

    return rms < threshold

# def is_silence(audio_data, threshold=20):
#     """Determine if the given audio segment is silence based on volume threshold."""
#     if audio_data is None:
#         logging.error("Audio data is None.")
#         return 0

#     if not isinstance(audio_data, (bytes, bytearray)):
#         logging.error(f"Unexpected data type for audio_data: {type(audio_data)}")
#         return 0
    
#     # Root Mean Square
#     rms = np.sqrt(np.mean(np.square(np.frombuffer(audio_data, dtype=np.int16))))
#     return rms < threshold

def get_audio_duration(audio_data, sample_rate=16000):
    """
    Calculate the duration of the audio data in milliseconds.

    :param audio_data: The bytearray containing the audio data.
    :param sample_rate: The sample rate of the audio data in Hz (default is 16000 Hz).
    :return: Duration of the audio in milliseconds.
    """
    if audio_data is None:
        logging.error("Audio data is None.")
        return 0

    if not isinstance(audio_data, (bytes, bytearray)):
        logging.error(f"Unexpected data type for audio_data: {type(audio_data)}")
        return 0
    
    try:
        num_samples = len(audio_data) // 2  # 2 bytes per sample for 16-bit audio
        duration_seconds = num_samples / sample_rate
        duration_milliseconds = duration_seconds * 1000
        return duration_milliseconds
    except Exception as e:
        logging.error(f"Error calculating audio duration: {e}")
        return 0
    
    


## For sure working
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = None
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.coroutine_queue = deque()
#         self.last_execution_time = time.time()
#         self.group_delay = 1  # seconds to wait before grouping and executing coroutines
#         self.thread.start()

#     def start_loop(self):
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.create_task(self.manage_coroutines())
#         self.loop.run_forever()

#     def stop_loop(self):
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()

#     def run_coroutine(self, coro):
#         future = asyncio.run_coroutine_threadsafe(self.enqueue_coroutine(coro), self.loop)
#         return future

#     async def enqueue_coroutine(self, coro):
#         self.coroutine_queue.append(coro)
#         return "Coroutine enqueued"

#     async def manage_coroutines(self):
#         while True:
#             await asyncio.sleep(0.1)  # Check every 100ms
#             if self.coroutine_queue and (time.time() - self.last_execution_time >= self.group_delay):
#                 self.last_execution_time = time.time()
#                 await self.execute_coroutines()

#     async def execute_coroutines(self):
#         coroutines = []
#         while self.coroutine_queue:
#             coroutines.append(self.coroutine_queue.popleft())
#         if coroutines:
#             grouped = asyncio.gather(*coroutines)
#             results = await grouped
#             print("Executed grouped coroutines:", results)




class AsyncLoopThread:
    def __init__(self):
        self.loop = None
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.coroutine_queue = deque()
        self.tasks = {}  # Dictionary to track tasks
        self.last_execution_time = time.time()
        self.group_delay = 1  # seconds to wait before grouping and executing coroutines
        self.thread.start()

    def start_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.manage_coroutines())
        self.loop.run_forever()

    def stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    def run_coroutine(self, coro):
        """Enqueue coroutine to be run in the asyncio loop."""
        future = asyncio.run_coroutine_threadsafe(self.enqueue_coroutine(coro), self.loop)
        return future

    async def enqueue_coroutine(self, coro):
        """Add coroutine to the queue and return a future representing the task."""
        task = self.loop.create_task(coro)
        self.tasks[task] = coro
        return task

    async def manage_coroutines(self):
        while True:
            await asyncio.sleep(0.1)  # Check every 100ms
            if self.coroutine_queue and (time.time() - self.last_execution_time >= self.group_delay):
                self.last_execution_time = time.time()
                await self.execute_coroutines()

    async def execute_coroutines(self):
        coroutines = []
        while self.coroutine_queue:
            coro = self.coroutine_queue.popleft()
            coroutines.append(coro)
        if coroutines:
            tasks = [self.loop.create_task(coro) for coro in coroutines]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("Executed grouped coroutines:", results)

    def cancel_task(self, task):
        """Cancel a specific task."""
        if task in self.tasks:
            task.cancel()
            del self.tasks[task]

    def cancel_all_tasks(self):
        """Cancel all tasks in the event loop."""
        for task in list(self.tasks):
            task.cancel()
            del self.tasks[task]
            
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = None
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.coroutine_queue = deque()
#         self.last_execution_time = time.time()
#         self.group_delay = 1  # seconds to wait before grouping and executing coroutines
#         self.thread.start()

#     def start_loop(self):
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.create_task(self.manage_coroutines())
#         self.loop.run_forever()

#     def stop_loop(self):
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()

#     def run_coroutine(self, coro):
#         """Enqueue coroutine to be run in the asyncio loop."""
#         asyncio.run_coroutine_threadsafe(self.enqueue_coroutine(coro), self.loop)

#     async def enqueue_coroutine(self, coro):
#         """Add coroutine to the queue."""
#         self.coroutine_queue.append(coro)

#     async def manage_coroutines(self):
#         while True:
#             await asyncio.sleep(0.1)  # Check every 100ms
#             if self.coroutine_queue and (time.time() - self.last_execution_time >= self.group_delay):
#                 self.last_execution_time = time.time()
#                 await self.execute_coroutines()

#     async def execute_coroutines(self):
#         coroutines = []
#         while self.coroutine_queue:
#             coro = self.coroutine_queue.popleft()
#             coroutines.append(coro)
#         if coroutines:
#             # Create asyncio tasks for each coroutine within the loop
#             tasks = [self.loop.create_task(coro) for coro in coroutines]
#             results = await asyncio.gather(*tasks, return_exceptions=True)
#             print("Executed grouped coroutines:", results)

#     def cancel_all_tasks(self):
#         """Cancel all tasks in the event loop."""
#         for task in asyncio.all_tasks(self.loop):
#             task.cancel()
            
#     def cancel_task(self, future):
#         """Cancel a specific future."""
#         if future in self.future_queue:
#             future.cancel()
#             self.future_queue.remove(future)








        
        
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = asyncio.new_event_loop()
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.coroutine_queue = deque()
#         self.thread.start()

#     def start_loop(self):
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def stop_loop(self):
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()

#     def run_coroutine(self, coro):
#         # Ensure that 'coro' is a coroutine function, not a called coroutine
#         if asyncio.iscoroutinefunction(coro):
#             # Enqueue the coroutine function to be called later
#             self.coroutine_queue.append(coro)
#         else:
#             raise TypeError("Expected a coroutine function")

#     async def manage_coroutines(self):
#         while True:
#             await asyncio.sleep(0.1)  # Manage frequency as needed
#             if self.coroutine_queue:
#                 await self.execute_coroutines()

#     async def execute_coroutines(self):
#         coroutines = [coro() for coro in self.coroutine_queue]  # Call each coroutine function
#         self.coroutine_queue.clear()
#         if coroutines:
#             grouped = asyncio.gather(*coroutines)
#             results = await grouped
#             print("Executed grouped coroutines:", results)      


## When using this AsyncLoopThread class, it will transcribe all the audio files, even if more is coming while transcribing.
# This is the correct behavior.
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = None
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.thread.start()

#     def start_loop(self):
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def stop_loop(self):
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()

#     def run_coroutine(self, coro):
#         asyncio.run_coroutine_threadsafe(coro, self.loop)
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = None
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.thread.start()

#     def start_loop(self):
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def run_coroutine(self, coro):
#         asyncio.run_coroutine_threadsafe(coro, self.loop)

#     def stop_loop(self):
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()
## When I use this class instead, it will stop the transcription task when it starts a new trasncription task.
# class AsyncLoopThread:
#     def __init__(self):
#         self.loop = None
#         self.thread = threading.Thread(target=self.start_loop, daemon=True)
#         self.thread.start()
#         self.current_task = None

#     def start_loop(self):
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_forever()

#     def stop_loop(self):
#         if self.current_task:
#             self.current_task.cancel()
#         self.loop.call_soon_threadsafe(self.loop.stop)
#         self.thread.join()

#     def run_coroutine(self, coro):
#         if self.current_task and not self.current_task.done():
#             self.current_task.cancel()  # Cancel the current task if it's still running
#         self.current_task = asyncio.run_coroutine_threadsafe(coro, self.loop)

# I made these changes because if I try to create both 
# transcription_handler = AsyncLoopThread()
# response_handler = AsyncLoopThread()
# then the response will be buggy and it will not wait for the response from the chat
# I need the transcription handler to not cancel any of the old tasks, it needs to do them in order like the original class did by default.