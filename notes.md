pip install -U transformers torch
pip install -U accelerate
pip install -U numpy
pip install -U soundfile
pip install -U torchaudio
pip install -U pyttsx3









To create custom model based on another model.

$: ollama show phi --modelfile
$: ollama show phi --modelfile > discord-modelfile
$: ollama create discord-phi --file discord-modelfile

$: ollama create openhermes-discord --file openhermes-discord-modelfile
$: ollama create openhermes-voice --file openhermes-voice.modelfile

$: docker pull redis
$: docker run -d --name my-redis-container -p 6379:6379 redis
$: docker start my-redis-container
$: docker exec -it 939e5bf8961723016fd200b51f21edafa02331716f70f33e5d4e508c40e18895 redis-cli FLUSHALL



docker exec -it 939e5bf8961723016fd200b51f21edafa02331716f70f33e5d4e508c40e18895 redis-cli FLUSHALL && python3 main.py







Getting voice to work was a hassle. So here are some notes for later so I remember how I got around the import errors.

First of all, discord library HAS to be installed from github discord.py and it HAS to be discord.py. Having another discord library in the same environment will mess with things. Do NOT install `discord`, always install `discord.py`
```
$ git clone https://github.com/Rapptz/discord.py
$ cd discord.py
$ python3 -m pip install -U .[voice]
```

After that we can install the extension:
```
pip install -U discord-ext-voice-recv
```

When installed this way, there should be no problems.