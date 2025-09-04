"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/api_util.py
Author: yuanzhenhui
Date: 2025-02-25 18:03:21
LastEditTime: 2025-09-04 13:39:31
"""
import requests
import json
import time
import json
import random
import tiktoken
from openai import OpenAI

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import const_util as CU

from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

CALL_CONTENT_TYPE = "application/json"
STREAM_TIMEOUT = 60


class ApiUtil:

    def __init__(self):

        # 硅基流动配置
        self.utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.api_keys = YamlUtil(self.utils_cnf).get_value('silicon.api_key')
        self.base_url = YamlUtil(self.utils_cnf).get_value('silicon.url')
        self.max_retries = int(YamlUtil(self.utils_cnf).get_value('silicon.max_retries'))
        
        # # ollama配置
        # self.ollama_url = YamlUtil(self.utils_cnf).get_value('ollama.url')
        # self.ollama_model = YamlUtil(self.utils_cnf).get_value('ollama.model')
        # self.ollama_max_retries = int(YamlUtil(self.utils_cnf).get_value('ollama.max_retries'))
        
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.letter_tokens = {self.enc.encode(c)[0]: -100 for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    def _create_circular_reader(self, arr):
        
        """
        为给定数组创建圆形读取器。

        此函数返回一个读取器函数，当调用该函数时，返回
        从数组中提取当前元素，并以循环方式推进索引。

        Parameters:
            arr（list）：以循环方式读取的数组。

        Returns:
            函数：返回数组中下一个元素的读取器函数
            每次调用时，都会在到达结束后循环回到开始。
        """

        index = 0

        def reader():
            nonlocal index
            result = arr[index]
            index = (index + 1) % len(arr)
            return result
        return reader

    def chat_with_sync(self, params, prompt_str):
        """
        该函数使用同步方式来进行聊天。

        该函数将 prompt_str 作为用户输入，prompt_str 作为系统提示，并将其作为
        聊天API的输入参数，使用同步方式来获取聊天结果，并将结果作为字符串返回。

        Parameters:
            params (dict):  chat_params
            prompt_str (str):  用户输入

        Returns:
            str:  服务器回复
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
                
                #  enable_thinking  该参数用于控制是否启用思考功能
                if "enable_thinking" in params["options"]:
                    payload["enable_thinking"] = params["options"]["enable_thinking"]
                
                response = requests.request("POST", url, json=payload, headers=headers)
                if response.status_code == 200:
                    return json.loads(response.text)["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        return None

    def chat_with_stream(self, params, prompt_array):
        """
        该函数使用 API key 通过流式方式与 OpenAI 的 chat 模型进行交互。

        该函数将输入的 prompt_array 和 context_array  merge 到一起，并将其作为 chat 模型的输入。
        然后，它将使用 OpenAI 的 chat API 通过流式方式与模型进行交互，并将生成的文本块yield到流式响应中。

        :param params: 一个字典，包含 chat 模型的参数，例如模型的名称、max_tokens 等
        :param prompt_array: 一个数组，每个元素是一个字典，包含用户输入的 prompt 信息
        :return: 一个生成器，yield 每个文本块的信息，包括 content、reasoning_content、token_count、total_token_count 和 token_rate
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
        该函数使用同步方式来调用 embedding API，以获取输入内容的 embedding 向量。

        该函数将遍历 content_array，并将每个元素作为参数，传递给 embedding API。
        之后，将 embedding API 的返回值，加入到 embedding_array 中。
        最后，将 embedding_array 作为返回值返回。

        参数：
            params (str):  embedding API 的参数，例如模型名称。
            content_array (list):  一个包含输入内容的列表。

        返回：
            list:  一个包含输入内容的 embedding 向量的列表。
        """
        url = self.base_url+"/embeddings"
        embedding_array = []
        for content in content_array:
            max_retries = self.max_retries
            for attempt in range(max_retries):
                try:
                    api_key = self._create_circular_reader(self.api_keys)
                    headers = {
                        "Authorization": f"Bearer {api_key()}",
                        "Content-Type": CALL_CONTENT_TYPE
                    }
                    payload = {
                        "model": params,
                        "input": content,
                        "encoding_format": "float"
                    }
                    response = requests.request("POST", url, json=payload, headers=headers)
                    if response.status_code == 200:
                        embedding_array.append(json.loads(response.text)["data"][0]["embedding"])
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        return embedding_array
    
    def rerank_with_sync(self, params, lastest_ask, search_knowledges):
        """
        该函数使用同步方式来调用 rerank API，以对搜索结果进行重排序。

        该函数将遍历 search_knowledges，并将每个元素作为参数，传递给 rerank API。
        之后，将 rerank API 的返回值，加入到 results 中。
        最后，将 results 中的文本部分，按照 relevance_score 排序，并将其作为返回值返回。

        参数：
            params (dict):  一个字典，包含 rerank API 的参数，例如模型名称和 top_n。
            lastest_ask (str):  最近的提问内容。
            search_knowledges (list):  一个包含搜索结果的列表。

        返回：
            list:  一个包含重排序后的文本的列表。
        """
        url = self.base_url+"/rerank"
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
                    "query": lastest_ask,
                    "documents": search_knowledges,
                    "top_n": params["options"]["top_n"],
                    "return_documents": params["options"]["return_documents"]
                }
                response = requests.request("POST", url, json=payload, headers=headers)
                if response.status_code == 200:
                    results = json.loads(response.text)["results"]
                    sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
                    return [result["document"]["text"] for result in sorted_results[:int(params["options"]["top_n"])]]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        return None

    # def chat_with_ollama(self, prompt_str):
    #     url = self.ollama_url+"/generate"
    #     max_retries = self.ollama_max_retries
    #     for attempt in range(max_retries):
    #         try:
    #             payload = {
    #                 "model": self.ollama_model,
    #                 "prompt": prompt_str,
    #                 "stream": False
    #             }
                
    #             response = requests.request("POST", url, json=payload)
    #             if response.status_code == 200:
    #                 return json.loads(response.text)["response"]
    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 raise
    #             time.sleep(1)
    #     return None