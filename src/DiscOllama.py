import os
import io
import json
import discord # python3 -m pip install -U "discord.py[voice]"
from discord import SpeakingState
from discord.ext import voice_recv # python -m pip install -U discord-ext-voice-recv
import logging
import asyncio
from datetime import datetime, timedelta
from .Response import Response
from .VoiceChat import VoiceChat

class DiscOllama:
    def __init__(self, model, ollama, discord, redis, speech_processor, tts):
        self.model = model
        self.model_voice = "openhermes-voice:latest"
        self.ollama = ollama
        self.discord = discord
        self.redis = redis
        self.speech_processor = speech_processor
        self.tts = tts
        self.voice_chats = []
        
        self.writing_tasks = {}
        
        self.pull_tasks = {}
        

        # register event handlers
        self.discord.event(self.on_ready)
        self.discord.event(self.on_message)
        
    
    def load_whitelist(self, whitelistPath):
        try:
            with open(whitelistPath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        
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
                administrator=True,
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
        commands = ['stop', 'join', 'leave', 'model', 'models', 'wipe', 'test', 'admin', 'dm']
        if any(content.startswith(f'!{command}') for command in commands):
            await self.handle_command(message)
            return
        
        # Save every single message the bot has access to, but not bot commands
        await self.save_message(message)
        
        passed = await self.check_message_conditions(message)
        if not passed:
            return
        
        
        r = Response(message)
        writing = asyncio.create_task(self.writing(r))
        self.writing_tasks[message.id] = (r, writing)
    
    
    async def handle_command(self, message):
        admin_commands = ['model', 'models', 'wipe', 'admin', 'dm', 'test']
        dm_commands = ['model', 'models', 'wipe']
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        
        # If server, check if commander is admin either under guildID in redis or super admin
        if message.guild is not None:
            if any(content.startswith(f'!{command}') for command in admin_commands):
                if (not self.is_admin(message)):
                    logging.info(f"{message.author.id} tried to use admin command '{content}' in channel '{message.channel.id}' while not being admin...")
                    await message.add_reaction('🚫')
                    return
        
        # if DM, check that commander is whitelisted in redis or is super admin
        if message.guild is None:
            if any(content.startswith(f'!{command}') for command in dm_commands):
                dm_allowed = await self.is_user_allowed_dm(message)
                if not dm_allowed:
                    logging.info(f"{message.author.id} tried to use admin command '{content}' in DM while not being whitelisted to DM...")
                    return
            
        ## These can be called by anyone
        if (content.lower().startswith('!stop')):
            await self.handle_command_stop(message)
        elif (content.lower().startswith('!join')):
            await self.handle_command_join(message)
        elif (content.lower().startswith('!leave')):
            await self.handle_command_leave(message)
            
        ## These can be called by admins
        elif (content.lower().startswith('!models')):
            await self.handle_command_models(message)
        elif (content.lower().startswith('!model')):
            await self.handle_command_model(message)        
        elif (content.lower().startswith('!wipe')):
            await self.handle_command_wipe(message)                
        elif (content.lower().startswith('!admin')):
            await self.handle_command_admin(message)                
        elif (content.lower().startswith('!dm')):
            await self.handle_command_dm(message)    
        elif (content.lower().startswith('!test')):
            await message.add_reaction('👌')
            # asyncio.create_task(self.test(message))
            await self.test(message)
        
        
    
    async def check_message_conditions(self, message):
        allowed_server_ids = os.getenv("DISCORD_SERVER")
        if message.guild is not None and allowed_server_ids:
            allowed_server_ids = [int(server_id.strip()) for server_id in allowed_server_ids.split(",")]
            if int(message.guild.id) not in allowed_server_ids:
                # Dont respond in other servers than the one specified in .env
                try:
                    await message.channel.send(f"I only serve my masters in another channel!")
                except Exception as e:
                    logging.error("Error sending message")
                    logging.error(e)
                finally:
                    return False
        
        dm_allowed = await self.is_user_allowed_dm(message)
        if not dm_allowed:
            # Don't respond to DMs from non-whitelisted users
            logging.info(f"Ignoring DM from non-whitelisted user {message.author.id}")
            await message.add_reaction('🙅‍♂️')
            messages = await self.get_messages(message)
            messages.append({"role": "system", "content": f"Generate an answer where you refuse you to respond to the user who has the name '{message.author.name}'. You are encouraged to be a little mean to {message.author.name} in you response. You have access to past messages from this user if there is a history, you can choose to use this history to get your point across further. {message.author.name}'s original message to you is: {message.content}"})
            current_model = await self.get_current_model(message.channel.id)
            response = await self.ollama.chat("openchat:7b-v3.5-1210-q5_K_M", messages=messages, stream=False)
            await message.author.send(response['message']['content'])
            await self.save_message(message, response['message']['content'])
            return False
        
        
        if message.guild is not None and not self.discord.user.mentioned_in(message):
            # don't respond to messages in channels, that don't mention us
            return False

        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        if not content:
            # Don't respond to empty messages
            return False
        
        return True

    async def test(self, message):
        logging.info("!test begin")
        
        is_admin = self.is_admin_in_channel(message.guild.id, message.author.id)
        logging.info(f"IS ADMIN IN CHANNEL: {is_admin}")
        # self.voice_chats[0].test()
        # self.voice_chat.test()
        # voice_channel = message.author.voice.channel.id
        # test = await self.get_voice_messages(voice_channel)
        # logging.info(test)
        # logging.info(f"User speaking: {self.voice_response.user_speaking}")
        # logging.info(f"Bot speaking: {self.voice_response.responding}")
        # logging.info(f"Audio Buffers: {self.voice_response.audio_buffers}")
        logging.info("!test end")
        # loop = asyncio.get_event_loop()
        # future = loop.run_in_executor(None, self.speech_processor, 'src/old/audio_benteb3nt_0.wav')
        # text = await future
        # logging.info(text['text'])
        
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
        
    async def writing(self, response):
        full_response = ""
        try:
            thinking = asyncio.create_task(self.thinking(response.message))
            messages = await self.get_messages(response.message)
            current_model = await self.get_current_model(response.message.channel.id)
            
            async for part in self.chat(messages, current_model):
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
            del self.writing_tasks[response.message.id]  # Remove the task from the dictionary
            await self.save_message(response.message, full_response)
       
     
    async def chat(self, messages, model=None, milliseconds=1000):
        if model is None:
            model = self.model
        sb = io.StringIO() # create new StringIO object that can write and read from a string buffer
        t = datetime.now()
        try:
            generator = await self.ollama.chat(model, messages=messages, stream=True)
            async for part in generator:
                sb.write(part['message']['content']) # write content to StringIO buffer
                # print(part['message']['content'], end='', flush=True)
            
                if milliseconds is None:
                    # If milliseconds is None, yield every time we get a return from the stream
                    part['message']['content'] = sb.getvalue()
                    yield part
                    sb.seek(0, io.SEEK_SET)
                    sb.truncate()
                elif part['done'] or datetime.now() - t > timedelta(milliseconds=milliseconds):
                    part['message']['content'] = sb.getvalue()
                    yield part
                    t = datetime.now()
                    sb.seek(0, io.SEEK_SET) # change current position in StringIO buffer (io.SEEK_SET = position is relative to beginning of buffer)
                    sb.truncate() # resizes StringIO buffer to current position. Since current position was just set to 0, this clears the buffer
                
        except Exception as e:
            logging.error("Error getting AI chat response")
            logging.error(e)
        
    async def generate(self, content, model=None):
        if model is None:
            model = self.model
        sb = io.StringIO()
        t = datetime.now()
        try:
            generator = await self.ollama.generate(model=model, prompt=content, keep_alive=-1, stream=True)
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


    async def get_messages(self, message):
        if not self.redis:
            return [{"role": "assistant" if message.author.id == self.discord.user.id else "user", "content": message.content}]
        
        messages = self.redis.lrange(f"messages:{message.channel.id}", 0, -1)
        # Convert the messages from JSON format to Python dictionaries
        messages = [json.loads(msg) for msg in messages]
        messages = [
                {"role": "assistant" if msg["author"] == self.discord.user.id else "user","content": msg["content"]}
            for msg in messages.copy()
        ]
        return messages

    async def save_message(self, message, response:str=None):
        if not self.redis:
            return False
        
        content = message.content
        if response is not None:
            content = response
        else:
            content = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + content + "\n\nSent by: " + str(message.author.name)
            # content = content + "\nTimestamp: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.redis.rpush(f"messages:{message.channel.id}", json.dumps({
            "author": message.author.id if response is None else self.discord.user.id,
            "content": content,
            "id": message.id,
            "attachments": [attachment.url for attachment in message.attachments] if response is None else [],
        }))
    
    async def get_chat_model(self, message):
        logging.info(f"Getting the model for chat {message.channel.id}")
        
    # Save voice messages as text for bot
    def save_voice_response(self, channel, text):
        if not self.redis:
            return False
        logging.info(f"Saving voice response in channel {channel}")
        self.redis.rpush(f"discord:voice:{channel}", json.dumps({
            "author": self.discord.user.id,
            "content": text,
        }))
    
    ## Save voice messages as text for users
    def save_voice_message(self, channel, text, user):
        if not self.redis:
            return False
        logging.info(f"Saving voice message from {user.name} in channel {channel}")
        if (user.id == self.discord.user.id):
            content = text
        else:
            content = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(text) + "\n\nSaid by: " + str(user.name)
        
        self.redis.rpush(f"discord:voice:{channel}", json.dumps({
            "author": user.id,
            "content": content,
        }))


    async def get_voice_messages(self, channel):
        if not self.redis:
            # return [{"role": "assistant" if message.author.id == self.discord.user.id else "user", "content": message.content}]
            return []
        
        messages = self.redis.lrange(f"discord:voice:{channel}", 0, -1)

        # Convert the messages from JSON format to Python dictionaries
        messages = [json.loads(msg) for msg in messages]

        messages = [{
                "role": "assistant" if msg["author"] == self.discord.user.id else "user",
                "content": msg["content"],}
            for msg in messages.copy()
        ]
        return messages

    async def wipe_messages(self, message):
        if not self.redis:
            return False
        
        logging.info(f"Wiping message history in {message.channel.id}. Requested by {message.author.id} {message.author.name}")
        redis_path = f"messages:{message.channel.id}"
        self.redis.delete(redis_path)
        await message.add_reaction('👌')
        return True
            
        
    async def stop_authors_tasks(self, message):
        for message_id, (response, answer_task) in self.writing_tasks.items():
            if response.message.author.id == message.author.id:
                answer_task.cancel()
                response.message.add_reaction('❌')
                
    async def join_vc(self, message):
        logging.info(f"Joining Voicechat with user {message.author.name}")
        if not message.channel:
            message.author.send("I cannot join a voice channel in a DM!")
            return
        if message.author.voice is None:
            await message.channel.send("You're not connected to a voice channel!")
            return
        ## TODO: Change so it checks if we are already in a voice channel in the same guild
        if self.discord.voice_clients:
            # vc = await self.discord.voice_clients[0].move_to(voice_channel)
            await message.channel.send("I'm already speaking to someone else. Go away!")
            return
        
        voice_channel = message.author.voice.channel
        vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        await message.channel.send(f"Joined {voice_channel.name}!")
        
        chat = VoiceChat(vc, self.discord)
        self.voice_chats.append(chat)
        self.voice_chat = chat
        ## TODO: Add voice response to self.connected_vcs list so we can listen to multiple voice channels in multiple guilds
        # self.voice_response = VoiceResponse(vc, self)
        # def callback(user, data: voice_recv.VoiceData):
        #     voice_response.user_speak(user, data)
        # vc.listen(voice_recv.BasicSink(callback))
                
    async def leave_vc(self, message):
        logging.info(f"Leaving Voicechat with user {message.author.name}")
        if not message.channel:
            message.author.send("I cannot leave a voice channel in a DM!")
            return
        if message.author.voice is None:
            await message.channel.send("You're not connected to a voice channel!")
            return
        
        voice_channel = message.author.voice.channel
        if self.discord.voice_clients:
            if self.discord.voice_clients[0].channel == voice_channel:
                await self.discord.voice_clients[0].disconnect()
                await message.channel.send(f"Left {voice_channel.name}!")
            else:
                await message.channel.send("We're not in the same voice channel!")
                
    # Check if either super admin or local guild admin
    def is_admin(self, message):
        super_admins = os.getenv("SUPER_ADMIN")
        if super_admins:
            super_admins = [int(id.strip()) for id in super_admins.split(",")]
        if (message.author.id in super_admins):
            return True
        
        # author is not a super admin, check if they are admin in guild
        if self.is_admin_in_channel(message.guild.id, message.author.id):
            return True
        
        return False
    
    def is_admin_in_channel(self, guildID, userID):
        if not self.redis:
            return False

        if self.redis.sismember(f"admins:{guildID}", str(userID)):
            return True
        else:
            return False
        

    ### ---------- Command Functions ----------
    async def handle_command_stop(self,message):
        # User requested to stop their tasks
        self.stop_authors_tasks(message)
            
    async def handle_command_join(self,message):
        # Admin request to join voice chat
        await self.join_vc(message)
        # asyncio.create_task(self.join_vc(message))
            
    async def handle_command_leave(self,message):
        # Admin request to leave voice chat
        await self.leave_vc(message)
            
    async def handle_command_models(self,message):
        await message.add_reaction('🤔')
        model_list = await self.ollama.list()
        model_string = ""
        for model in model_list['models']:
            model_string += f"{model['name']}\n"
        
        await message.remove_reaction('🤔', self.discord.user)
        
        ## TODO: Send as Thread? I think not.
        # msg_thread = await message.channel.create_thread(name='List of available models', message=message, auto_archive_duration=60)
        # await msg_thread.send(f"**Models:**\n{model_string}")
        await message.author.send(f"**Models:**\n{model_string}")
        
    async def handle_command_model(self,message):
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        # If there are no second word, then assume user wants to see current model used in this chat
        if len(content.split()) > 1:
            second_word = content.split()[1].lower()
            if second_word == "set":
                await self.handle_set_model(message)
            elif second_word == "pull":
                pull_task = asyncio.create_task(self.handle_pull_model(message))
                self.pull_tasks[message.id] = pull_task
                # await self.handle_pull_model(message)
            elif second_word == "delete":
                await self.handle_delete_model(message)
            elif second_word == "create":
                await self.handle_create_model(message)
                                                                                                                                        
                                                                                                                    
        else:
            current_model = await self.get_current_model(message.channel.id)
            await message.reply(f"**Current model:** {current_model}")
            
    async def handle_command_wipe(self,message):
        await self.wipe_messages(message)
            
    async def handle_command_admin(self,message):
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        if len(content.split()) > 1:
            second_word = content.split()[1].lower()
            if second_word == "add":
                await self.add_admin(message)
            if second_word == "remove":
                await self.remove_admin(message)
                
    async def handle_command_dm(self,message):
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        # The first word is "!admin" check if the second word is "add" or "remove"
        if len(content.split()) > 1:
            second_word = content.split()[1].lower()
            if second_word == "add":
                await self.add_user_to_dm(message)
            if second_word == "remove":
                await self.remove_user_from_dm(message)

        
    async def handle_set_model(self, message):
        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        
        if len(content.split()) > 2:
            model_string = content.split()[2]
            model_list = await self.ollama.list()
            available_models = []
            for model in model_list['models']:
                available_models.append(f"{model['name']}")
            if model_string in available_models:
                await self.set_current_model(message.channel.id, model_string)
                await message.add_reaction('👌')
            else:
                logging.info(f"{model_string} is not an available model.")
                await message.add_reaction('💩')
        else:
            await message.reply(f"**Missing Model...**")
    
    async def handle_pull_model(self,message):
        try:
            content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
            if len(content.split()) > 2:
                model_string = content.split()[2]
                logging.info(f"Pulling model {model_string}!")
                await message.add_reaction('🤔')  # Add a reaction to indicate loading
                progress_message = await message.reply("Status: ...")
                
                current_digest = ''
                generator = await self.ollama.pull(model_string, stream=True)
                async for progress in generator:
                    digest = progress.get('digest', '')
                    if digest != current_digest:
                        # progress_bar += '.'
                        current_digest = digest
                    if not digest:
                        status = progress.get('status')
                        if status:
                            await progress_message.edit(content=f"Status: {status}")
                            logging.info(f"PULL STATUS: {status}")
                        continue
            
            
                # test = await self.ollama.pull(model_string)
                logging.info(f"Pulling model {model_string} complete!")
                await message.remove_reaction('🤔', self.discord.user)
                # await message.add_reaction('✅')
                # await progress_message.delete()
            else:
                await message.reply(f"**Missing Model...**")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error("Error pulling model...")
            logging.error(e)
            pass
        finally:
            del self.pull_tasks[message.id]  # Remove the task from the dictionary
        
        
        
    async def handle_delete_model(self,message):
        pass
    async def handle_create_model(self,message):
        pass
                
    async def add_admin(self, message):
        if not self.redis:
            await message.add_reaction('💩')
            return
        
        mentions = message.mentions
        users_to_add = [member for member in mentions if member.id != self.discord.user.id]
        
        if (len(users_to_add) > 0):
            for member in users_to_add:
                logging.info(f"adding {member.id} to admins:{message.guild.id}")
                self.redis.sadd(f"admins:{message.guild.id}", member.id)
        
        await message.add_reaction('👌')
                
    async def remove_admin(self, message):
        if not self.redis:
            await message.add_reaction('💩')
            return
        
        mentions = message.mentions
        users_to_remove = [member for member in mentions if member.id != self.discord.user.id]

        if len(users_to_remove) > 0:
            for member in users_to_remove:
                admin_set_name = f"admins:{message.guild.id}"
                if self.redis.sismember(admin_set_name, str(member.id)):
                    logging.info(f"Removing {member.id} from {admin_set_name}")
                    self.redis.srem(admin_set_name, member.id)
                else:
                    logging.info(f"{member.id} is not an admin in {admin_set_name}")
                    
        await message.add_reaction('👌')
        
    async def add_user_to_dm(self, message):
        if not self.redis:
            await message.add_reaction('💩')
            return
        
        mentions = message.mentions
        users_to_add = [member for member in mentions if member.id != self.discord.user.id]
        
        if (len(users_to_add) > 0):
            for member in users_to_add:
                logging.info(f"adding {member.id} to dm_whitelist")
                self.redis.sadd(f"dm_whitelist", member.id)
        
        await message.add_reaction('👌')
        
    async def remove_user_from_dm(self, message):
        if not self.redis:
            await message.add_reaction('💩')
            return
        
        mentions = message.mentions
        users_to_remove = [member for member in mentions if member.id != self.discord.user.id]

        if len(users_to_remove) > 0:
            for member in users_to_remove:
                if self.redis.sismember("dm_whitelist", str(member.id)):
                    logging.info(f"Removing {member.id} from dm_whitelist")
                    self.redis.srem("dm_whitelist", member.id)
                else:
                    logging.info(f"{member.id} is not in dm_whitelist")
                    
        await message.add_reaction('👌')
        
    async def is_user_allowed_dm(self, message):
        super_admins = os.getenv("SUPER_ADMIN")
        if super_admins:
            super_admins = [int(id.strip()) for id in super_admins.split(",")]
        if (message.author.id in super_admins):
            return True
        
        if not self.redis:
            return False

        if self.redis.sismember("dm_whitelist", str(message.author.id)):
            return True
        else:
            return False
        
    async def get_current_model(self, channelID):
        if not self.redis:
            return self.model
        
        model_string = self.redis.get(f"model:{channelID}")
        return model_string if model_string else self.model
    
    async def set_current_model(self, channelID, model_string):
        if not self.redis:
            return
        
        self.redis.set(f"model:{channelID}", model_string)