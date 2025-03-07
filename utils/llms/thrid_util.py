"""
Copyright (c) 2025 by yuanzhenhui. All right reserved.
FilePath: /brain-mix/utils/llms/thrid_util.py
Author: yuanzhenhui
Date: 2025-01-10 23:13:52
LastEditTime: 2025-01-11 00:37:50
"""
import time
import random
from transformers import AutoTokenizer
from openai import OpenAI

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))
from yaml_util import YamlConfig
from logging_util import LoggingUtil 
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))
    
class SiliconUtil:

    _instance = None
    _initialized = False 

    def __init__(self):
        """
        Initializes the SiliconUtil instance with the configurations.

        This method reads the configurations from the YAML files and sets the
        configurations to the instance variables. It also initializes the
        chat model's tokenizer.

        """
        if not SiliconUtil._initialized:
            thrid_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config', 'llms', 'thrid_cnf.yml')
            )
            self.api_keys = thrid_config.get_value('silicon.api_key')
            self.models = thrid_config.get_value('silicon.models')
            self.url = thrid_config.get_value('silicon.url')
            
            pytorch_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config', 'llms', 'pytorch_cnf.yml')
            )
            self.chat_model_path = pytorch_config.get_value('pytorch.chat_path')
            self.chat_tokenizer = self._init_tokenizer()
            
            base_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config','llms', 'base_cnf.yml')
                )
            self.temperature = float(base_config.get_value('llm.system.temperature'))
            self.top_p = float(base_config.get_value('llm.system.top_p'))
            self.max_token = int(base_config.get_value('llm.system.max_token'))
            self.stream_flag = base_config.get_value('llm.system.stream_flag')
            
            SiliconUtil._initialized = True
            
    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of SiliconUtil is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one and
        assigns it to the `_instance` class variable. If an instance already
        exists, it simply returns the existing instance.

        Returns:
            SiliconUtil: The instance of the class.
        """
        if cls._instance is None:
            # Create a new instance
            cls._instance = super().__new__(cls)
        # Return the instance
        return cls._instance
    
    def silicon_qwen(self, message):
        """
        Uses the Silicon API to generate a response to the given message.

        Args:
            message (str): The message to generate a response for.

        Yields:
            dict: A dictionary containing the generated text, the number of
                  tokens in the generated text, the total number of tokens
                  generated so far, and the token rate (i.e. the number of
                  tokens generated per second).
        """
        # Select a random API key and model
        api_key = random.choice(self.api_keys)
        qwen_model = random.choice(self.models)
        client = OpenAI(base_url=self.url, api_key=api_key)
        # Get the start time
        start_time = time.time()
        token_count = 0
        try:
            # Send the request
            response = client.chat.completions.create(
                model=qwen_model,
                messages=message,
                stream=self.stream_flag,  
                max_tokens=self.max_token,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            # Get the response text
            for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                chunk_token_count = len(self.chat_tokenizer.encode(chunk_message))
                token_count += chunk_token_count
                yield {
                    'text': chunk_message,
                    'token_count': chunk_token_count,
                    'total_token_count': token_count,
                    'token_rate': token_count / (time.time() - start_time)
                }
        except Exception as e:
            # If an error occurs, yield an error message
            yield f"Error: {str(e)}"
            
    def _init_tokenizer(self):
        """
        Initializes the chat model's tokenizer from the pre-trained model path.

        This method loads the tokenizer using the specified model path and
        enables trusting remote code for loading the tokenizer.

        Returns:
            AutoTokenizer: The initialized tokenizer for the chat model.
        """
        # Load the tokenizer from the pre-trained model path
        return AutoTokenizer.from_pretrained(self.chat_model_path, trust_remote_code=True)
