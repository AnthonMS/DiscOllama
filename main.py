import logging
import os
import redis
import ollama
import discord
from dotenv import load_dotenv
# from DiscOllama import DiscOllama
from src.DiscOllama import DiscOllama

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

load_dotenv(f'{THIS_PATH}\\.env')
logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')


if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True
    
    DiscOllama(
        str(os.getenv("OLLAMA_MODEL", "phi")),
        ollama.AsyncClient(host=os.getenv("OLLAMA_HOST_URL"), auth=(os.getenv("BASIC_AUTH_USERNAME"), os.getenv("BASIC_AUTH_PASSWORD")), verify=False),
        discord.Client(intents=intents),
        redis.Redis(host=str(os.getenv("REDIS_HOST")), port=int(os.getenv("REDIS_PORT")), db=0, decode_responses=True),
    ).run(os.getenv("DISCORD_TOKEN"))