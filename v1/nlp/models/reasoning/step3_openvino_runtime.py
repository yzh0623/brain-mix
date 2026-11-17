"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step3_openvino_runtime.py
Author: yuanzhenhui
Date: 2025-11-25 09:46:31
LastEditTime: 2025-12-04 17:35:15
"""

import openvino_genai as ov_genai
import threading
from threading import Lock, Event
from queue import Queue, Empty
from typing import Dict, List, Generator, Union
import time
import os
import re
import sys
import tiktoken

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))
import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

nlp_cnf = os.path.join(project_dir, 'resources','config', CU.ACTIVATE, 'nlp_cnf.yml')
YamlNLP = YamlUtil(nlp_cnf)

THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

class PromptTemplates:

    SYSTEM_PROMPT_TCM = """
        你是康养问答助手。

        【强制规则】
        1. 只用中文回答，禁止英文
        2. 禁止使用任何符号：' " ` * # @ $ % ^ & ( ) [ ] { } < > / \ | ~
        3. 标点只用：，。、！？
        4. 回答完毕输出【EOA】后立即停止

        【输出格式】
        【证型】具体证型名称
        【调理】分点建议
        【EOA】

        【举例】
        问题：经常头晕乏力，脸色发白
        回答：
        【证型】气血两虚证
        【调理】一、饮食方面，多吃红枣、桂圆、枸杞煮粥，每周炖一次乌鸡汤或猪肝汤补血
            二、作息方面，晚上十一点前入睡，午休半小时
            三、运动方面，每天散步二十分钟，不宜剧烈运动
        【EOA】
        
        注意：禁止输出英文单词。禁止输出特殊符号。
        """

    @classmethod
    def get_messages(cls, user_query: str) -> List[Dict[str, str]]:
        """
        Convert a user query into a list of messages.

        The messages will be used to ask a language model to generate health advice.
        The language model will be given two messages:
        - A system message with a prompt to generate health advice.
        - A user message with the user query.

        Parameters:
            user_query (str): The user query.

        Returns:
            List[Dict[str, str]]: A list of two messages.
        """

        messages = [{
            # The role of this message is "system".
            "role": "system", 
            # The content of this message is the system prompt.
            "content": cls.SYSTEM_PROMPT_TCM
        }]
        # Add a user message with the user query.
        messages.append({"role": "user", "content": user_query})
        return messages

class ChunkingStreamer:
    
    def __init__(
        self,
        token_queue: Queue,
        stop_strings: List[str],
        timeout_seconds: float = 30.0,
        chunk_size: int = 15,
    ):
        """
        Initialize the ChunkingStreamer object.

        Parameters:
            token_queue (Queue): The queue to store the generated tokens.
            stop_strings (List[str]): The list of stop strings to stop the generation.
            timeout_seconds (float, optional): The timeout seconds to stop the generation. Defaults to 30.0.
            chunk_size (int, optional): The chunk size to generate tokens. Defaults to 15.
        """
        self.token_queue = token_queue
        self.stop_strings = stop_strings
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        
        self.generated_text = ""
        self.buffer = ""
        self.stop_requested = Event()
        self.stop_reason = ""
        self.start_time = time.time()
        self.token_count = 0
        
        # The punctuation to be used to split the text into sentences.
        self.punctuation = ['。', '！', '？', '；', '\n']
        
        # The tags to be used to detect the thought mode.
        self.in_thought_mode = False 
        

    def __call__(self, subword: str) -> bool:
        """
        Process a subword and return True if the generation should be stopped.

        Parameters:
            subword (str): The subword to be processed.

        Returns:
            bool: True if the generation should be stopped, False otherwise.
        """
        if self.stop_requested.is_set():
            return True
        
        # Check if the generation has timed out
        if time.time() - self.start_time > self.timeout_seconds:
            self._flush_buffer_and_stop("timeout")
            return True

        processed_subword = subword
        
        # Process the subword and handle the thought mode
        while True:
            if self.in_thought_mode:
                # Find the end of the thought mode
                end_pos = processed_subword.find(THINK_END_TAG)
                if end_pos == -1:
                    # If the end of the thought mode is not found, stop the generation
                    return False
                else:
                    # Trim the processed subword and exit the thought mode
                    self.in_thought_mode = False
                    processed_subword = processed_subword[end_pos + len(THINK_END_TAG):]
                    if not processed_subword: return False
            else:
                # Find the start of the thought mode
                start_pos = processed_subword.find(THINK_START_TAG)
                if start_pos == -1:
                    break
                else:
                    # Trim the processed subword and enter the thought mode
                    safe_content = processed_subword[:start_pos]
                    if safe_content:
                        self.buffer += safe_content
                        self.generated_text += safe_content
                        self.token_count += 1

                    self.in_thought_mode = True
                    processed_subword = processed_subword[start_pos + len(THINK_START_TAG):]
                    if not processed_subword: return False
        
        # Check if the processed subword contains any stop strings
        temp_text = self.buffer + processed_subword
        for stop_str in self.stop_strings:
            if stop_str in temp_text:
                self._stop_and_trim(stop_str, processed_subword.strip())
                return True

        # Add the processed subword to the buffer
        self.buffer += processed_subword
        self.generated_text += processed_subword
        self.token_count += 1

        # Check if the buffer is full
        if len(self.buffer) >= self.chunk_size:
            # Find the last punctuation in the buffer
            split_pos = -1
            for p in self.punctuation:
                pos = self.buffer.rfind(p)
                if pos > split_pos:
                    split_pos = pos

            # Trim the buffer and send the content to the queue
            content_to_send = ""
            if split_pos != -1 and len(self.buffer) - (split_pos + 1) < self.chunk_size:
                content_to_send = self.buffer[:split_pos + 1]
                self.buffer = self.buffer[split_pos + 1:]
            else:
                content_to_send = self.buffer[:self.chunk_size]
                self.buffer = self.buffer[self.chunk_size:]
            self.token_queue.put(content_to_send)
        return False

    def _stop_and_trim(self, stop_str: str, last_subword: str):
        """
        Trim the buffer and stop the generation when a stop string is found.

        Parameters:
            stop_str (str): The stop string to look for.
            last_subword (str): The last subword generated by the model.

        Returns:
            None
        """
        full_check_string = self.buffer + last_subword
        stop_pos = full_check_string.find(stop_str)
        if stop_pos != -1:
            # Trim the buffer and send the content to the queue
            content_to_send = full_check_string[:stop_pos]
            if content_to_send:
                self.token_queue.put(content_to_send)
            # Update the generated text and buffer
            current_buffer_len = len(self.buffer)
            self.generated_text = self.generated_text[:len(self.generated_text) - current_buffer_len] + content_to_send
            self.buffer = ""
        # Request to stop the generation
        self._request_stop(f"stop_string: {stop_str}")
    
    def _flush_buffer_and_stop(self, reason: str):
        """
        Flush the buffer and stop the generation.

        Parameters:
            reason (str): The reason to stop the generation.

        Returns:
            None
        """
        # Flush the buffer and send the content to the queue
        if self.buffer:
            self.token_queue.put(self.buffer)
        self.buffer = ""
        # Request to stop the generation
        self._request_stop(reason)

    def _request_stop(self, reason: str) -> bool:
        """
        Requests to stop the generation of text.

        Parameters:
            reason (str): The reason to stop the generation.

        Returns:
            bool: True if the stop was successful, False otherwise.
        """
        self.stop_reason = reason
        self.stop_requested.set()
        #print(f"\nEarly stop triggered: {reason} (tokens={self.token_count}, chars={len(self.generated_text)})")
        return True
    
    def end(self):
        """
        Ends the generation of text.

        This method is used to end the generation of text. It will flush the buffer and
        send the content to the queue.

        Returns:
            None
        """
        # Flush the buffer and send the content to the queue
        if self.buffer:
            self.token_queue.put(self.buffer)
        # Add a sentinel to indicate the end of the generation
        self.token_queue.put(None)
        self.buffer = ""
    
    def get_clean_texts(self) -> str:
        """
        Get the cleaned text from the generated text.

        This method removes any remaining <think> tags, stop strings, and formats the text
        to ensure consistency with the previous generation.

        Returns:
            str: The cleaned text.
        """
        text = self.generated_text
        
        # 1. Remove any remaining <think> tags
        # This is the final insurance to remove any <think> tags that may have been left over
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 2. Remove stop strings
        # Stop strings are words or phrases that are not allowed to appear in the generated text
        for stop_str in self.stop_strings:
            text = re.sub(re.escape(stop_str), '', text, flags=re.I)
        
        # Replace 3 or more consecutive newline characters with 2 newline characters
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

class OpenvinoRuntime:

    _instance = None
    _initialized = False
    _init_lock = Lock()
    _inference_lock = Lock()

    nlp_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
    YamlNLP = YamlUtil(nlp_cnf)

    DEFAULT_CONFIG = {
        "model_path": os.path.join(YamlNLP.get_value('models.reasoning.openvino.model'), 'OVMS', YamlNLP.get_value('models.reasoning.openvino.ovms.model_name')),
        "device": YamlNLP.get_value('models.reasoning.openvino.model_device'),

        "generation": {
            "max_new_tokens": YamlNLP.get_value('models.reasoning.openvino.genai.max_tokens'),
            "temperature": YamlNLP.get_value('models.reasoning.openvino.genai.temperature'),
            "top_p": YamlNLP.get_value('models.reasoning.openvino.genai.top_p'),
            "top_k": YamlNLP.get_value('models.reasoning.openvino.genai.top_k'),
            "do_sample": YamlNLP.get_value('models.reasoning.openvino.genai.do_sample'),
            "repetition_penalty": YamlNLP.get_value('models.reasoning.openvino.genai.repetition_penalty')
        },

        "early_stop": {
            "timeout_seconds": YamlNLP.get_value('models.reasoning.openvino.genai.timeout_seconds')
        },

        "stop_strings": [
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "\n\n\n",
            "【EOA】",
            "\nUser:",
            "\nuser:",
            "Human:",
            "Assistant:",
        ]
    }

    def __init__(self):
        """
        Initialize the OpenvinoRuntime instance.
        
        The initialization process involves loading the model from the specified path
        and setting up the generation configuration.
        """
        if OpenvinoRuntime._initialized:
            return
        with OpenvinoRuntime._init_lock:
            if OpenvinoRuntime._initialized:
                return
            try:
                # Initialize the configuration with default values
                self.config = self.DEFAULT_CONFIG.copy()
                
                # Initialize the components, including the model and generation configuration
                self._init_components()
                
                # Mark the instance as initialized
                OpenvinoRuntime._initialized = True
                logger.info("OpenvinoRuntime initialized successfully")
            except Exception as e:
                # Log an error if the initialization fails
                logger.error(f"Failed to initialize OpenvinoRuntime: {e}")
                raise

    def _init_components(self):
        """
        Initializes the components of the OpenvinoRuntime instance, including
        the model and generation configuration.

        This method loads the model from the specified path and sets up the generation
        configuration based on the provided configuration.
        """
        model_path = self.config["model_path"]
        device = self.config["device"]
        
        logger.info(f"Loading model from {model_path} on {device}...")
        # Load the model from the specified path
        self.pipe = ov_genai.LLMPipeline(model_path, device)
        logger.info("Model loaded successfully (Mock)")
        
        # Set up the generation configuration
        self._setup_generation_config()
        
        # Initialize the prompt templates
        self.prompt_templates = PromptTemplates()
        
        # Initialize the stop strings
        self.stop_strings = self.config["stop_strings"]

    def _setup_generation_config(self):
        """
        Setup the generation configuration based on the provided configuration.

        The generation configuration is used to control the text generation process.
        """
        gen_cfg = self.config["generation"]
        self.gen_config = ov_genai.GenerationConfig()
        # Maximum number of new tokens to generate
        self.gen_config.max_new_tokens = gen_cfg["max_new_tokens"]
        # Temperature of the softmax distribution
        self.gen_config.temperature = gen_cfg["temperature"]
        # Number of tokens to sample from the top-p tokens
        self.gen_config.top_p = gen_cfg["top_p"]
        # Number of tokens to sample from the top-k tokens
        self.gen_config.top_k = gen_cfg["top_k"]
        # Whether to sample from the top tokens or not
        self.gen_config.do_sample = gen_cfg["do_sample"]
        # Repetition penalty to prevent the model from generating the same tokens
        self.gen_config.repetition_penalty = gen_cfg["repetition_penalty"]
        logger.info(f"Generation config: do_sample={self.gen_config.do_sample}, rep_penalty={self.gen_config.repetition_penalty}, max_tokens={self.gen_config.max_new_tokens}")

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply the chat template to the given messages.

        The chat template is used to format the messages in a way that is compatible
        with the OpenVINO model.

        Args:
            messages (List[Dict[str, str]]): The list of messages to apply the chat template to.
                Each message should have a "role" key with a value of either "system" or "user",
                and a "content" key with the message content.

        Returns:
            str: The formatted prompt string.
        """
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            # Add a comment to explain the formatting
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        # Add a comment to explain the joining of the prompt parts
        return "\n".join(prompt_parts)
    
    def _build_prompt(self, msg: Union[str, List[Dict[str, str]]]) -> str:
        """
        Build the prompt string based on the given message.

        The message can be either a string or a list of message dictionaries.
        If the message is a string, it will be converted into a list of message dictionaries
        using the `get_messages` method of the `PromptTemplates` class. If the message is
        a list of message dictionaries, it will be used as is.

        Args:
            msg (Union[str, List[Dict[str, str]]]): The message to build the prompt string from.

        Returns:
            str: The formatted prompt string.
        """
        if isinstance(msg, str):
            # Convert the string message into a list of message dictionaries
            messages = self.prompt_templates.get_messages(user_query=msg)
        else:
            # Use the given list of message dictionaries
            messages = msg
            # Check if the list of messages contains a system message
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                # Add a system message to the list of messages if it doesn't contain one
                system_msg = {"role": "system", "content": self.prompt_templates.SYSTEM_PROMPT_TCM}
                messages = [system_msg] + list(messages)
        # Apply the chat template to the list of messages
        return self._apply_chat_template(messages)
        
    def transfor_stream_msg(
        self,
        msg: Union[str, List[Dict[str, str]]],
        **kwargs
    ) -> Generator[Dict[str, str], None, None]:
        """
        Stream the output of the OpenVINO model as it is generated.

        The method uses a ChunkingStreamer to read the output of the OpenVINO model
        in chunks of 15 tokens. The output is then yielded as a generator.

        Args:
            msg (Union[str, List[Dict[str, str]]]): The message to generate text from.
                Can be either a string or a list of message dictionaries.
            **kwargs: Additional keyword arguments to control the generation process.

        Yields:
            Dict[str, str]: A dictionary containing the generated text and other information.
        """
        # Check if the system is busy
        if not OpenvinoRuntime._inference_lock.acquire(blocking=False):
            logger.warning("Inference request denied: System is busy.")
            yield {"content": "系统繁忙，请稍后再试，CPU资源已被占用。","finished": True,"stop_reason": "busy_lock"}
            return
        try:
            # Build the prompt string
            prompt = self._build_prompt(msg)
            # Get the early stop configuration
            early_stop_cfg = self.config["early_stop"]
            # Get the timeout from the early stop configuration
            timeout = kwargs.get("timeout_seconds", early_stop_cfg["timeout_seconds"])
            # Create a token queue to store the generated tokens
            token_queue: Queue = Queue()
            # Create a ChunkingStreamer to read the output of the OpenVINO model
            streamer = ChunkingStreamer(
                token_queue=token_queue,
                stop_strings=self.stop_strings,
                timeout_seconds=timeout,
                chunk_size=15, 
            )
            # Get the generation configuration
            gen_config = self.gen_config
            # Update the generation configuration with the provided keyword arguments
            if kwargs:
                temp_gen_config = ov_genai.GenerationConfig()
                temp_gen_config.max_new_tokens = kwargs.get("max_new_tokens", self.config["generation"]["max_new_tokens"])
                temp_gen_config.temperature = kwargs.get("temperature", self.config["generation"]["temperature"])
                temp_gen_config.top_p = kwargs.get("top_p", self.config["generation"]["top_p"])
                temp_gen_config.top_k = kwargs.get("top_k", self.config["generation"]["top_k"])
                temp_gen_config.do_sample = kwargs.get("do_sample", self.config["generation"]["do_sample"])
                temp_gen_config.repetition_penalty = kwargs.get("repetition_penalty", self.config["generation"]["repetition_penalty"])
                gen_config = temp_gen_config
            # Create a list to store any generation errors
            generation_error = [None]
            # Define a function to run the generation
            def run_generation():
                try:
                    # Run the generation
                    self.pipe.generate(prompt, gen_config, streamer)
                except Exception as e:
                    # Store any generation errors
                    generation_error[0] = e
                    logger.error(f"Generation error: {e}")
                finally:
                    # End the streamer
                    streamer.end()
            # Create a thread to run the generation
            gen_thread = threading.Thread(target=run_generation, daemon=True)
            # Start the generation thread
            gen_thread.start()
            # Run a loop to yield the generated text
            while True:
                try:
                    # Get a token from the token queue
                    token = token_queue.get(timeout=timeout + 5)
                    # If the token is None, break the loop
                    if token is None:
                        break
                    token = token.strip()
                    token = token.replace(THINK_END_TAG, '')
                    # Yield the generated text
                    yield {"content": token,"finished": False,"stop_reason": "",}
                except Empty:
                    logger.warning("Token queue timeout")
                    break
            # Join the generation thread
            gen_thread.join(timeout=2)
            # If there is a generation error, raise it
            if generation_error[0]:
                raise generation_error[0]
            # Get the clean texts from the streamer
            final_text = streamer.get_clean_texts()
            # Yield the final text
            yield {"content": "","finished": True,"stop_reason": streamer.stop_reason or "completed","full_response": final_text,}
        except Exception as e:
            # Yield an error message if there is an exception
            yield {"content": "","finished": True,"stop_reason": f"runtime_error: {e}","error": str(e)}
        finally:
            # Release the inference lock
            OpenvinoRuntime._inference_lock.release()
            #logger.debug("Inference lock released.")

if __name__ == '__main__':

    logger.info("=" * 60)
    logger.info("Qwen3 0.6B OpenVINO Runtime 推理测试")
    logger.info("=" * 60)

    input_prompt = "中医药理论是否能解释并解决全身乏力伴随心跳过速的症状？"
    
    enc = tiktoken.get_encoding("cl100k_base")

    try:
        llm = OpenvinoRuntime()
        logger.info(f"提示词: {input_prompt}")
        logger.info("=" * 60)
        full_response = ""
        token_count = 0
        start_time = time.time()
        for chunk in llm.transfor_stream_msg(input_prompt):
            if not chunk["finished"]:
                print(chunk["content"], end="", flush=True)
                full_response += chunk["content"]
                token_count += len(enc.encode(chunk["content"]))
            else:
                if 'full_response' in chunk:
                    final_text = chunk['full_response']
                    if not full_response:
                        print(final_text)
                        token_count += len(enc.encode(final_text))
        end_time = time.time()
        duration = end_time - start_time
        logger.info("=" * 60)
        logger.info(f"总用时: {duration:.2f} 秒")
        logger.info(f"总生成 token 数: {token_count}")
        if duration > 0:
            logger.info(f"生成速率: {token_count/duration:.2f} token/秒")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"主程序运行失败: {e}")
