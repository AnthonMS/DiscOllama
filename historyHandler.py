import os
import json
import threading
from discord import Message

class HistoryHandler:
    _lock = threading.Lock()
    
    def __init__(self, message:Message):
        self.message = message
        self.file_path = 'conversations.json'
        self.conversation_id = str(message.author.id) if message.guild is None else str(message.channel.id)
        
        self.conversation = self.getConversationFromJSONFile()
        if self.conversation is None:
            self.createNewConversation()
               
    def getConversationFromJSONFile(self):
        with self._lock:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    all_conversations = json.load(f)
                    return all_conversations.get(self.conversation_id, None)
        return None
      
    def createNewConversation(self):
        if self.message.guild is None:
            self.conversation = {
                "user": self.message.author.name,
                "history": []
            }
        else:
            self.conversation = {
                "channel": self.message.channel.name,
                "guild": self.message.guild.name,
                "guild_id": self.message.guild.id,
                "history": []
            }
            
        self.writeConversationToJSON()
    
    def writeConversationToJSON(self):
        with self._lock:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    all_conversations = json.load(f)
            else:
                all_conversations = {}
            all_conversations[self.conversation_id] = self.conversation
            with open(self.file_path, 'w') as f:
                json.dump(all_conversations, f, indent=4)
                
                
                
                
    def add_message(self, role, content, images = []):
        message = {
            "role": role,
            "content": content,
        }
        if role == "user" and self.message.guild is not None:
            # message["content"] = str(self.message.author.id) + ' ' + self.message.author.name + ' : ' + content
            # message["content"] = self.message.author.name + ' : ' + content
            # message["user_id"] = self.message.author.id
            message["username"] = self.message.author.name
        
        if (len(images) > 0):
            message["images"] = images
            
        self.conversation["history"].append(message)
        self.writeConversationToJSON()
        
    
    def get_messages(self):
        return self.conversation["history"]