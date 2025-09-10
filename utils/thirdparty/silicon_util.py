"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/thirdparty/silicon_util.py
Author: yuanzhenhui
Date: 2025-08-04 16:51:58
LastEditTime: 2025-09-10 16:05:44
"""

import requests
import json
import time
import json
import random
import tiktoken
from openai import OpenAI

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import const_util as CU

from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

CALL_CONTENT_TYPE = "application/json"
STREAM_TIMEOUT = 60

class SiliconUtil:

    def __init__(self):
        """
        Initialize the API utility.

        This class provides methods for making requests to the Silicon API.
        """

        # Load the API configuration from the YAML file
        self.utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.api_keys = YamlUtil(self.utils_cnf).get_value('silicon.api_key')
        self.base_url = YamlUtil(self.utils_cnf).get_value('silicon.url')
        self.max_retries = int(YamlUtil(self.utils_cnf).get_value('silicon.max_retries'))

        # Initialize the TikToken encoder
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Create a dictionary mapping letter tokens to a score of -100
        # This is used to penalize the model for generating letters outside of the
        # top 1000 tokens
        self.letter_tokens = {self.enc.encode(c)[0]: -100 for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    def _create_circular_reader(self, arr):
        """
        Creates a circular reader for the given array.

        This function returns a reader function that, when called, returns
        the current element from the array and advances the index in a
        circular manner. When the index reaches the end of the array, it
        wraps around to the beginning.

        Parameters:
            arr (list): The array to read from in a circular manner.

        Returns:
            function: A reader function that returns the next element from
                the array, wrapping around to the beginning when it reaches
                the end.
        """

        index = 0

        def reader():
            """
            Returns the next element from the array, wrapping around to the
            beginning when it reaches the end.
            """
            nonlocal index
            result = arr[index]
            index = (index + 1) % len(arr)
            return result
        return reader

    def chat_with_sync(self, params, prompt_str):
        """
        Make a synchronous chat request to the Silicon API.

        This function takes a prompt string and a dictionary of parameters as input and
        makes a synchronous chat request to the Silicon API. The request is sent using
        the POST method and the response is expected to be a JSON object with a single
        key-value pair, where the key is "choices" and the value is a list of objects
        with a single key-value pair, where the key is "message" and the value is an
        object with two key-value pairs, where the keys are "role" and "content" and the
        values are strings. The function returns the value of the "content" key in the
        first element of the list.

        Parameters:
            params (dict): A dictionary containing the parameters for the request,
                including the model name, max tokens, temperature, top p, frequency penalty,
                and enable thinking.
            prompt_str (str): The prompt string to send to the API.

        Returns:
            str: The response content from the API, or None if all retries fail.
        """

        url = self.base_url+"/chat/completions"
        max_retries = self.max_retries
        for attempt in range(max_retries):
            try:
                api_key = self._create_circular_reader(self.api_keys)
                headers = {
                    "Authorization": f"Bearer {api_key()}",
                    "Content-Type": CALL_CONTENT_TYPE
                }
                payload = {
                    "model": params["model"],
                    "stream": False,
                    "response_format": {"type": "text"},
                    "max_tokens": params["options"]["max_tokens"],
                    "temperature": params["options"]["temperature"],
                    "top_p": params["options"]["top_p"],
                    "frequency_penalty": params["options"]["frequency_penalty"],
                    "messages": [
                        {
                            "role": "system",
                            "content": params["prompt"]
                        },
                        {
                            "role": "user",
                            "content": prompt_str
                        }
                    ]
                }
                
                # If enable thinking is specified, add it to the payload
                if "enable_thinking" in params["options"]:
                    payload["enable_thinking"] = params["options"]["enable_thinking"]
                
                # Send the chat request
                response = requests.request("POST", url, json=payload, headers=headers)
                
                # Check if the request is successful
                if response.status_code == 200:
                    
                    # Return the response content
                    return json.loads(response.text)["choices"][0]["message"]["content"]
            except Exception as e:
                
                # If there is an exception, sleep for a random time between 1 and 5 seconds
                if attempt == max_retries - 1:
                    raise
                time.sleep(random.randint(1, 5))
                
        # If all retries fail, return None
        return None

    def chat_with_stream(self, params, prompt_array):
        """
        Call Silicon API to get the stream of chat completion.

        This function will call the Silicon API to get the stream of chat completion.
        The Silicon API will return a stream of chat completion that is sorted by the relevance score.
        The function will return the stream of chat completion.

        Args:
            params (dict): The parameters to pass to the Silicon API.
            prompt_array (list): The list of prompts to pass to the Silicon API.

        Yields:
            dict: A dictionary containing the content, reasoning_content, token_count, total_token_count and token_rate.
        """
        total_array = [{"role": "system","content": params["prompt"]}]
        total_array.extend(prompt_array)
        try:
            api_key = self._create_circular_reader(self.api_keys)
            client = OpenAI(base_url=self.base_url, api_key=api_key())
            token_count = 0
            start_time = time.time()
            with client.chat.completions.create(
                    model=params["model"],
                    messages=total_array,
                    stream=True,
                    max_tokens=params["options"]["max_tokens"],
                    max_completion_tokens=params["options"]["max_tokens"],
                    temperature=params["options"]["temperature"],
                    top_p=params["options"]["top_p"],
                    timeout=STREAM_TIMEOUT,
                    frequency_penalty=params["options"]["frequency_penalty"],
                    logit_bias=self.letter_tokens,
                    seed=random.randint(1, 999999)
                ) as response:
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    reasoning_content = chunk.choices[0].delta.model_extra.get("reasoning_content")  or ""
                    token_count += len(self.enc.encode(content))
                    yield {
                            "content": content,
                            "reasoning_content": reasoning_content,
                            "token_count": len(self.enc.encode(content)),
                            "total_token_count": token_count,
                            "token_rate": token_count / (time.time() - start_time)
                        }
        except Exception as e:
            logger.error(f"Chat with stream failed: {str(e)}")

    def embedding_with_sync(self, params, content_array):
        """
        Call Silicon API to get the embeddings of a given list of content.

        This function will call the Silicon API to get the embeddings of a given list of content.
        The Silicon API will return a list of embeddings that is sorted by the relevance score.
        The function will return the list of embeddings.

        Args:
            params (dict): A dictionary that contains the model name and the options dictionary.
            content_array (list): A list of content to get the embeddings for.

        Returns:
            list: A list of embeddings.
        """
        url = self.base_url+"/embeddings"
        embedding_array = []
        for content in content_array:
            max_retries = self.max_retries
            for attempt in range(max_retries):
                try:
                    
                    # Get the API key from the circular reader
                    api_key = self._create_circular_reader(self.api_keys)
                    
                    # Create the headers and payload for the request
                    headers = {
                        "Authorization": f"Bearer {api_key()}",
                        "Content-Type": CALL_CONTENT_TYPE
                    }
                    payload = {
                        "model": params,
                        "input": content,
                        "encoding_format": "float"
                    }
                    
                    # Send the request and check if it is successful
                    response = requests.request("POST", url, json=payload, headers=headers)
                    if response.status_code == 200:
                        
                        # If the request is successful, get the embedding from the response
                        embedding_array.append(json.loads(response.text)["data"][0]["embedding"])
                        break
                except Exception as e:
                    
                    # If there is an exception, sleep for a random time between 1 and 5 seconds
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        return embedding_array
    
    def rerank_with_sync(self, params, lastest_ask, search_knowledges):
        """
        Call Silicon API to rerank the search results.

        This function will call the Silicon API to rerank the search results. The Silicon API
        will return a list of documents that is sorted by the relevance score. The function will
        return the top N documents where N is the value of the top_n parameter in the options
        dictionary of the params.

        Args:
            params (dict): A dictionary that contains the model name and the options dictionary.
            lastest_ask (str): The latest question asked by the user.
            search_knowledges (list): A list of search results.

        Returns:
            list: A list of top N documents where N is the value of the top_n parameter in the options
            dictionary of the params.
        """
        url = self.base_url+"/rerank"
        max_retries = self.max_retries
        for attempt in range(max_retries):
            try:
                
                # Get the API key from the circular reader
                api_key = self._create_circular_reader(self.api_keys)
                
                # Create the headers and payload for the request
                headers = {
                    "Authorization": f"Bearer {api_key()}",
                    "Content-Type": CALL_CONTENT_TYPE
                }
                payload = {
                    "model": params["model"],
                    "query": lastest_ask,
                    "documents": search_knowledges,
                    "top_n": params["options"]["top_n"],
                    "return_documents": params["options"]["return_documents"]
                }
                
                # Send the request and check if it is successful
                response = requests.request("POST", url, json=payload, headers=headers)
                if response.status_code == 200:
                    
                    # If the request is successful, get the sorted results from the response
                    results = json.loads(response.text)["results"]
                    sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
                    return [result["document"]["text"] for result in sorted_results[:int(params["options"]["top_n"])]]
            except Exception as e:
                
                # If there is an exception, sleep for a random time between 1 and 5 seconds
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        return None
