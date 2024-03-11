import os
import io
import json
import redis
import ollama
import discord
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv


# THIS_PATH = os.path.dirname(os.path.realpath(__file__))

# load_dotenv(f'{THIS_PATH}\\.env')
# logging.basicConfig(filename=f'{THIS_PATH}\\bot.log', level=logging.INFO, format='%(asctime)s %(message)s')


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


class DiscOllama:
    def __init__(self, model, ollama, discord, redis):
        self.model = model
        self.ollama = ollama
        self.discord = discord
        self.redis = redis
        
        self.answering_tasks = {}
        
        try:
            with open('dm.whitelist', 'r') as f:
                self.dm_whitelist = json.load(f)
        except FileNotFoundError:
            self.dm_whitelist = []
            
        try:
            with open('admin.whitelist', 'r') as f:
                self.admin_whitelist = json.load(f)
        except FileNotFoundError:
            self.admin_whitelist = []

        # register event handlers
        self.discord.event(self.on_ready)
        self.discord.event(self.on_message)
        
        
    def run(self, token):
        try:
            self.discord.run(token)
        except Exception:
            logging.exception('Discord client encountered an error')


    async def on_ready(self):
        activity = discord.Activity(name='bentebot', state='Being Developed...', type=discord.ActivityType.custom)
        await self.discord.change_presence(activity=activity)

        logging.info(
            'Ready! Invite URL: %s',
            discord.utils.oauth_url(
            self.discord.application_id,
            permissions=discord.Permissions(
                read_messages=True,
                send_messages=True,
                create_public_threads=True,
            ),
            scopes=['bot'],
            ),
        )

    async def on_message(self, message):
        if self.discord.user == message.author:
            # don't respond to ourselves
            return
        
        # Todo: Create handle_command function that takes in message and calls the appropriate function
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        
        if (content.lower().startswith('!test')):
            # await message.channel.send("Testing...")
            await message.add_reaction('👌')
            testing = await self.get_messages(message)
            logging.info(testing)
            return
        if (content.lower().startswith('!stop')):
            # User requested to stop their tasks
            self.stop_authors_tasks(message)
            return
        if (content.lower().startswith('!wipe')):
            # User requested to stop their tasks
            await self.wipe_messages(message)
            return
        
        
        # Save all messages in all channels and DMs
        await self.save_message(message)
        
        passed = await self.check_message_conditions(message)
        if not passed:
            return
        
        
        r = Response(message)
        answering = asyncio.create_task(self.answering(r))
        self.answering_tasks[message.id] = (r, answering)
        
        # async for part in self.chat([{"role": "user", "content": content}]):
        #     thinking.cancel()
        #     # print(part['message']['content'], end='', flush=True)
        #     await r.write(part['message']['content'], end='...')
        # await r.write('')
    
    
    async def check_message_conditions(self, message):
        if message.guild is not None and (os.getenv("DISCORD_SERVER") and int(message.guild.id) != int(os.getenv("DISCORD_SERVER"))):
            # Dont respond in other servers than the one specified in .env
            try:
                await message.channel.send(f"I only serve my master!")
            except Exception as e:
                logging.error("Error sending message")
                logging.error(e)
            finally:
                return False
        
        if message.author.id not in self.dm_whitelist:
            # Don't respond to DMs from non-whitelisted users
            logging.info(f"Ignoring DM from non-whitelisted user {message.author.id} != {self.dm_whitelist}")
            await message.author.send("Who do you think you are?")
            return False
        
        
        if message.guild is not None and not self.discord.user.mentioned_in(message):
            # don't respond to messages in channels, that don't mention us
            return False

        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        if not content:
            # Don't respond to empty messages
            return False
        
        return True


        
        
    async def thinking(self, message, timeout=999):
        try:
            await message.add_reaction('🤔')
            # async with message.channel.typing():
            #     await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            # await message.add_reaction('❌')
            pass
        except Exception as e:
            logging.error("Error thinking")
            logging.error(e)
            await message.add_reaction('💩')
            pass
        finally:
            await message.remove_reaction('🤔', self.discord.user)
        
    async def answering(self, response):
        full_response = ""
        try:
            thinking = asyncio.create_task(self.thinking(response.message))
            chat_history = await self.get_messages(response.message)
            messages = [{
                    "role": "assistant" if msg["author"] == self.discord.user.id else "user",
                    "content": msg["content"],}
                for msg in chat_history
            ]
            # messages.append({"role": "user", "content": content})
            
            async for part in self.chat(messages):
                if thinking is not None and not thinking.done():
                    thinking.cancel()
                # print(part['message']['content'], end='', flush=True)
                part_content = part['message']['content']
                full_response += part_content
                await response.write(part_content, end='...')
                    
            await response.write('')
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("Error answering")
            logging.error(e)
            pass
        finally:
            if thinking is not None and not thinking.done():
                thinking.cancel()
            await response.message.remove_reaction('🤔', self.discord.user) # Make sure we remove thinking reaction when done answering
            del self.answering_tasks[response.message.id]  # Remove the task from the dictionary
            await self.save_message(response.message, full_response)
       

    async def chat(self, messages):
        sb = io.StringIO() # create new StringIO object that can write and read from a string buffer
        t = datetime.now()
        try:
            generator = await self.ollama.chat(self.model, messages=messages, stream=True)
            async for part in generator:
                sb.write(part['message']['content']) # write content to StringIO buffer
                # print(part['message']['content'], end='', flush=True)
            
                if part['done'] or datetime.now() - t > timedelta(seconds=1):
                    part['message']['content'] = sb.getvalue()
                    yield part
                    t = datetime.now()
                    sb.seek(0, io.SEEK_SET) # change current position in StringIO buffer (io.SEEK_SET = position is relative to beginning of buffer)
                    sb.truncate() # resizes StringIO buffer to current position. Since current position was just set to 0, this clears the buffer
                
        except Exception as e:
            logging.error("Error getting AI chat response")
            logging.error(e)
        
    async def generate(self, content):
        sb = io.StringIO()
        t = datetime.now()
        try:
            generator = await self.ollama.generate(model=self.model, prompt=content, keep_alive=-1, stream=True)
            async for part in generator:
                sb.write(part['response'])

                if part['done'] or datetime.now() - t > timedelta(seconds=1):
                    part['response'] = sb.getvalue()
                    yield part
                    t = datetime.now()
                    sb.seek(0, io.SEEK_SET)
                    sb.truncate()

        except Exception as e:
            logging.error("Error getting AI generate response")
            logging.error(e)


    async def save_message(self, message, response:str=None):
        # Take the message and save it under either user's id or channel's id
        redis_path = f"discord:channel:{message.channel.id}"
        if message.guild is None: # Save under user's id
            redis_path = f"discord:user:{message.author.id}"
        
        content = message.content
        if response is not None:
            content = response
        else:
            content = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + content + "\n\nSent by: " + str(message.author.name)
            # content = content + "\nTimestamp: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.redis.rpush(redis_path, json.dumps({
            "author": message.author.id if response is None else self.discord.user.id,
            "content": content,
            "id": message.id,
            "attachments": [attachment.url for attachment in message.attachments] if response is None else [],
        }))


    async def wipe_messages(self, message):
        if message.guild is None: # DM
            if message.author.id in self.dm_whitelist: # Whitelisted user
                logging.info(f"Wiping direct message history for {message.author.id} {message.author.name}")
                redis_path = f"discord:user:{message.author.id}"
                self.redis.delete(redis_path)
                await message.add_reaction('👌')
                return True
        elif message.author.id in self.admin_whitelist: # Channel admin
            logging.info(f"Wiping message history in ({message.guild.id}) {message.guild.name} ({message.channel.id}) {message.channel.name}. Requested by {message.author.id} {message.author.name}")
            redis_path = f"discord:channel:{message.channel.id}"
            self.redis.delete(redis_path)
            await message.add_reaction('👌')
            return True
            
        # User not allowed to wipe, react accordingly.
        await message.add_reaction('🖕')
        return False
        
    async def stop_authors_tasks(self, message):
        for message_id, (response, answer_task) in self.answering_tasks.items():
            if response.message.author.id == message.author.id:
                answer_task.cancel()
                response.message.add_reaction('❌')
                
    async def get_messages(self, message):
        if message.guild is None:
            # Retrieve from user's id
            messages = self.redis.lrange(f"discord:user:{message.author.id}", 0, -1)
        else:
            # Retrieve from channel's id
            messages = self.redis.lrange(f"discord:channel:{message.channel.id}", 0, -1)

        # Convert the messages from JSON format to Python dictionaries
        messages = [json.loads(msg) for msg in messages]

        return messages

def main():
    intents = discord.Intents.default()
    intents.message_content = True
    
    DiscOllama(
        str(os.getenv("OLLAMA_MODEL", "phi")),
        ollama.AsyncClient(host=os.getenv("OLLAMA_HOST_URL"), auth=(os.getenv("BASIC_AUTH_USERNAME"), os.getenv("BASIC_AUTH_PASSWORD")), verify=False),
        discord.Client(intents=intents),
        redis.Redis(host=str(os.getenv("REDIS_HOST")), port=int(os.getenv("REDIS_PORT")), db=0, decode_responses=True),
    ).run(os.getenv("DISCORD_TOKEN"))


if __name__ == '__main__':
    main()