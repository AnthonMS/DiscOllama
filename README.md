# DiscOllama - Your Personal Discord AI Chatbot

DiscOllama is a Discord bot that uses Ollama as the backend LLM provider. It's designed to provide an interactive and engaging experience for Discord users.

## Features

- Ollama LLM server integration: The bot uses Ollama to process and generate responses based on user input and past interactions.
- Huggingface pipelines for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech)
- Nginx for secure reverse proxy to Ollama


## Developer's Notes

This is just a stupid idea that I had to try after playing around with Ollama and different LLMs for a day or two. I quickly googled if it had an API to interact with Ollama and so many ideas started running through my mind. I can now host a local AI for my smart home as well. Hmmm... My mind is racing with ideas.

So to learn the API and start some kind of project, learning to create a discord bot was something that was on the TODO list as well anyway, I decided this was probably a fun little project. So lets see where this takes me.

Later edit: I have stumpled upon the official discollama discord bot used in the Ollama discord server. This is a very good starting point, so I will be borrowing the code to expand on it instead of reinventing the wheel.

## Getting Started (No requirements.txt yet. This is going to change a lot.)

1. Clone the repository.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Create a `.env` file in the root directory and fill it with your configuration values. Use the `example.env` file as a template.
4. Run the bot by executing `python main.py`.


## Contributing

Contributions are welcome! Please feel free to submit a pull request.

If you know how to do async shit well, hit me up. As you can see from this repo, I need help. lol.
If you know how to make the huggingface pipelines faster, hit me up, I would love to learn more about optimizing LLM usage and in general maybe get a better understanding of what is going on.


Everything in this project is a learning experience for me. So don't judge me too negatively. Teach me instead.