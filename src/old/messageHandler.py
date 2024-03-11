import os
import json
from discord import Message
import asyncio
from typing import List, Dict, Union
from historyHandler import HistoryHandler
from helpers import image_url_to_base64


## dolphin-mixtral:8x7b-v2.5-q5_K_M

class MessageHandler:
    def __init__(self, logger, ollama, message:Message):
        self.logger = logger
        self.ollama = ollama
        self.message = message
        self.history = HistoryHandler(message)
        self.response = []
        self.sent_messages = []
        self.finished = False
        self.ai_task = None
        self.send_task = None
        self.queue = asyncio.Queue()
        
        self.ai_model = "openhermes:latest"
        
        try:
            with open('dm.whitelist', 'r') as f:
                self.whitelist = json.load(f)
        except FileNotFoundError:
            self.whitelist = []
            
    
            
        
    async def direct_message(self) -> None:
        """
        Handles direct messages sent to the bot.
        """
        if self.message.author.id not in self.whitelist:
            self.logger.info(f"Ignoring DM from non-whitelisted user {self.message.author.id} != {self.whitelist}")
            await self.message.author.send("Pay your developers!")
            return
        
        
        try:
            image_urls = await self.get_image_urls(self.message)
            # self.history.add_message("user", original_message.content, image_urls)
            # chat_history = await self.get_chat_history()
            
            
            # msg:str = f'Thinking...'
            # sent_message = await original_message.channel.send(msg)
            # ai_res = await self.get_ai_response(chat_history)
            # self.history.add_message("assistant", ai_res)
            # if (len(ai_res) > 0):
            #     ai_res_chunks = await self.split_response_into_chunks(ai_res)
            #     await self.send_chunks(original_message, sent_message, ai_res_chunks)
            
            # await original_message.author.send("Done!")
            # await original_message.author.send("You are whitelisted, congrats bro! But I still don't respond to DMs yet...")
        except Exception as e:
            self.logger.error("Error sending direct message")
            self.logger.error(e)
          
    
    
    async def channel_message(self) -> None:
        """
        Handles messages sent to the bot in a channel.
        """
        try:
            # Add message from user to history
            image_urls = await self.get_image_urls(self.message)
            self.history.add_message("user", self.message.content, image_urls)
            chat_history = await self.get_chat_history()
            
            # Send message indicating we got the response.
            sent_message = await self.send_message('Thinking...')
            self.sent_messages.append(sent_message)
            
            # self.get_thread = threading.Thread(target=self.get_ai_response_background, args=(chat_history,))
            # self.get_thread.start()
            
            self.ai_task = asyncio.create_task(self.get_ai_response())
            self.send_task = asyncio.create_task(self.send_ai_response())
            
        except Exception as e:
            self.logger.error("Error handling message in channel")
            self.logger.error(e)
    
    
    
    
    async def get_ai_response(self) -> None:
        try:
            chat_history = await self.get_chat_history()
            for part in self.ollama.chat(self.ai_model, messages=chat_history, stream=True):
                print(part['message']['content'], end='', flush=True)
                # self.logger.info(part['message']['content'])
                if len(self.response) == 0 or len(self.response[-1]) + len(part) > 1900:
                    self.response.append(part['message']['content'])
                else:
                    self.response[-1] += part['message']['content']
            print("\n")
            
            self.history.add_message("assistant", ''.join(self.response))
            # await self.send_message("Done!")
        except Exception as e:
            self.logger.error("Error getting AI response")
            self.logger.error(e)
            self.finished = True
            self.history.add_message("assistant", ''.join(self.response))
        
        
    async def send_ai_response(self):
        try:
            while not self.finished:
                self.logger.info("Sending AI response...")
                    
                if len(self.sent_messages) > 0:
                    last_sent_message = self.sent_messages[-1]
                    last_sent_response = self.response[len(self.sent_messages) - 1]
                    if last_sent_message.content != last_sent_response:
                        new_last_sent_message = await self.edit_message(last_sent_message, last_sent_response)
                        self.sent_messages[-1] = new_last_sent_message
                        
                # Check if there is a new response that hasnt been sent yet.
                if len(self.response) > len(self.sent_messages):
                    last_response = self.response[-1]
                    new_message = await self.send_message(last_response)
                    self.sent_messages.append(new_message)
                
                if self.ai_task.done():
                    self.finished = True
                    break
                
                await asyncio.sleep(2)
        except Exception as e:
            self.logger.error("Error sending ai response")
            self.logger.error(e)
            self.finished = True
        
    
    async def send_message(self, message:str):
        if (message == None or len(message) == 0):
            self.logger.error("Error sending message: message cannot empty")
            return
        try:
            if self.message.guild is None:
                return await self.message.author.send(message)
            else:
                return await self.message.channel.send(message)
        except Exception as e:
            self.logger.error("Error sending message")
            self.logger.error(e)
            
    
    async def edit_message(self, message:Message, given_edit:str):
        if (given_edit == None or len(given_edit) == 0):
            self.logger.error("Error editing message: message cannot empty")
            return
        try:
            return await message.edit(content=given_edit)
        except Exception as e:
            self.logger.error("Error editing message")
            self.logger.error(e)
            return message
        
        
    async def get_image_urls(self, original_message: Message) -> List[str]:
        return [attachment.url for attachment in original_message.attachments if attachment.content_type.startswith('image/')]


    async def get_chat_history(self) -> List[Dict[str, Union[str, List[str]]]]:
        chat_history = [
            {key: message[key] for key in ("role", "content", "images") if key in message}
            for message in self.history.get_messages()
        ]
        # Convert image_urls to base64 strings
        for message in chat_history:
            if "images" in message:
                message["images"] = [image_url_to_base64(url) for url in message["images"]]
        return chat_history
    