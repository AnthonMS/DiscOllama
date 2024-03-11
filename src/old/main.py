import asyncio
import os
from dotenv import load_dotenv
import logging
from typing import Final
from discord import Intents, Client, Message
from messageHandler import MessageHandler
from ollama import Client as OllamaClient, AsyncClient


THIS_PATH = os.path.dirname(os.path.realpath(__file__))
load_dotenv(f'{THIS_PATH}\\.env')
logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')

TOKEN: Final[str] = os.getenv("DISCORD_TOKEN")
SERVER_ID: Final[int] = os.getenv("DISCORD_SERVER")
CHANNEL_ID: Final[int] = os.getenv("DISCORD_CHANNEL")

intents: Intents = Intents.default()
intents.message_content = True
client: Client = Client(intents=intents)

ollama = OllamaClient(host=os.getenv("OLLAMA_HOST_URL"), auth=(os.getenv("BASIC_AUTH_USERNAME"), os.getenv("BASIC_AUTH_PASSWORD")), verify=False)
message_handlers = []

logging.getLogger('httpx').setLevel(logging.ERROR)

# Handle Startup of the Bot
@client.event
async def on_ready() -> None:
    print(f'{client.user} has connected to Discord!')
    logging.info(f'{client.user} has connected to Discord!')
    client.loop.create_task(cleanup_message_handlers())


# Handle Incoming Messages
@client.event
async def on_message(message: Message) -> None:
    # Ignore messages from the bot itself
    if message.author == client.user:
        return
    
    if (message.content.lower().startswith('!stop')):
        cancel_message_handlers_by_author(message)
        return
    
    message_handler = MessageHandler(logging, ollama, message)
    message_handlers.append(message_handler)
    
    # Check if the message is a DM
    if message.guild is None:
        # message_handler = MessageHandler(logging, ollama, message)
        # message_handlers.append(message_handler)
        await message_handler.direct_message(message)
        return
    
    # Check if message is on wrong server
    if int(message.guild.id) != int(SERVER_ID):
        try:
            await message.channel.send(f"I do not respond to traitors. This is not the server you are looking for.")
        except Exception as e:
            logging.error("Error sending message")
            logging.error(e)
        return
    
    # Check if message is on main channel or if bot was mentioned
    if int(message.channel.id) == int(CHANNEL_ID) or client.user in message.mentions:
        # message_handler = MessageHandler(logging, ollama, message)
        # message_handlers.append(message_handler)
        await message_handler.channel_message()
        return
    
    # Well shit, we shouldn't be here
    logging.error("Error handling message")
    logging.error(f"Message from {message.author} in guild={message.guild} channel={message.channel}: {message.content}")




def cancel_message_handlers_by_author(message: Message) -> None:
    global message_handlers
    for handler in message_handlers[:]:  # Create a copy for iteration
        if handler.message.author == message.author:
            remove_message_handler(handler)


def remove_message_handler(target_handler: MessageHandler) -> None:
    global message_handlers
    # if not target_handler.ai_task is None and not target_handler.ai_task.done():
    #     logging.info("Cancelling AI Task")
    #     target_handler.ai_task.cancel()
    # message_handlers = [handler for handler in message_handlers if handler != target_handler]

async def cleanup_message_handlers():
    while True:
        global message_handlers
        for handler in message_handlers[:]:  # Create a copy for iteration
            if handler.finished:
                remove_message_handler(handler)
                
        await asyncio.sleep(10)

def main() -> None:
    client.run(token=TOKEN)


if __name__ == '__main__':
    main()