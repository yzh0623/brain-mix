"""
Copyright (c) 2025 by yuanzhenhui. All right reserved.
FilePath: /brain-mix/utils/llms/pytorch_util.py
Author: yuanzhenhui
Date: 2024-11-25 22:13:54
LastEditTime: 2025-01-11 00:17:57
"""

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer, StoppingCriteriaList, BitsAndBytesConfig

from typing import Dict, Any, Iterator

import time
import json
import torch

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlConfig
from llm_util import StopOnTokens
from cuda_util import CudaMonitor
from logging_util import LoggingUtil 
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

nvidia_cuda = "cuda"

class PytorchUtil:

    _instance = None
    _initialized = False 

    def __init__(self):
        """
        Initializes the PyTorch Util instance.
        
        This method reads the configurations from the YAML files and sets the
        configurations to the instance variables. It also initializes the CUDA
        monitor and the chat model.
        
        """
        if not PytorchUtil._initialized:
            
            # Read the configurations from the YAML files
            pytorch_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config', 'llms', 'pytorch_cnf.yml')
            )
            self.use_4bit = pytorch_config.get_value('pytorch.quant.use_4bit')
            self.use_double_quant = pytorch_config.get_value('pytorch.quant.use_double_quant')
            self.quant_type = pytorch_config.get_value('pytorch.quant.quant_type')
            
            base_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config', 'llms', 'base_cnf.yml')
            )
            self.offload_folder = base_config.get_value('vino_torch.offload_folder')
            self.offload_state_dict = base_config.get_value('vino_torch.offload_state_dict')
            self.skip_special_tokens = base_config.get_value('vino_torch.skip_special_tokens')
            self.tokenize = base_config.get_value('vino_torch.tokenize')
            self.add_generation_prompt = base_config.get_value('vino_torch.add_generation_prompt')
            self.model_device = base_config.get_value('vino_torch.model_device')
            self.do_sample = base_config.get_value('vino_torch.do_sample')
            self.no_repeat_ngram_size = int(base_config.get_value('vino_torch.no_repeat_ngram_size'))
            self.use_cache = base_config.get_value('vino_torch.use_cache')
            self.skip_prompt = base_config.get_value('vino_torch.skip_prompt')
            self.llm_temperature = float(base_config.get_value('llm.system.temperature'))
            self.llm_top_p = float(base_config.get_value('llm.system.top_p'))
            self.llm_top_k = int(base_config.get_value('llm.system.top_k'))
            self.llm_repetition_penalty = float(base_config.get_value('llm.system.repetition_penalty'))
            self.llm_max_token = int(base_config.get_value('llm.system.max_token'))
            self.llm_timeout = int(base_config.get_value('llm.system.timeout'))
            
            # Set the device to the NVIDIA GPU
            self.device = torch.device(nvidia_cuda)
            
            # Initialize the CUDA monitor
            self.memory_manager = CudaMonitor()
            self.memory_manager.start_monitoring()
            
            # Initialize the chat model
            chat_model_path = pytorch_config.get_value('pytorch.chat_path')
            self.chat_model = self._init_model(chat_model_path)
            self.chat_tokenizer = self._init_tokenizer(chat_model_path)
            
            # Set the initialized flag to True
            PytorchUtil._initialized = True

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of PytorchUtil is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one and
        assigns it to the `_instance` class variable. If an instance already exists,
        it simply returns the existing instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            PytorchUtil: The instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _init_model(self,model_path):
        """
        Initializes the PyTorch model from a given pre-trained model path.

        This method loads the pre-trained model from the given path and sets up
        the model configurations according to the settings in the YAML files.

        Args:
            model_path (str): The path to the pre-trained model.

        Returns:
            torch.nn.Module: The initialized PyTorch model.
        """
        quant_config = BitsAndBytesConfig(
                # Load the model in 4-bit floating-point format
                load_in_4bit=self.use_4bit,
                # Use double quantization for 4-bit models
                bnb_4bit_use_double_quant=self.use_double_quant,
                # Set the quantization type to either 'qint8' or 'fp16'
                bnb_4bit_quant_type=self.quant_type,
                # Set the compute dtype to bfloat16
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=AutoConfig.from_pretrained(model_path),
            # Offload the state dictionary to CPU
            offload_state_dict=self.offload_state_dict,
            # Set the torch dtype to bfloat16
            torch_dtype=torch.bfloat16,
            # Set the quantization configuration
            quantization_config=quant_config,
            # Set the device map to NVIDIA GPU
            device_map=nvidia_cuda,
            # Set the attention implementation to flash attention
            attn_implementation="flash_attention_2",
            # Trust remote code
            trust_remote_code=True
        )
        # Move the model to the NVIDIA GPU
        model = model.to(self.device)
        # Compile the model to reduce overhead
        model = torch.compile(model,mode="reduce-overhead")
        return model
    
    def _init_tokenizer(self, model_path: str) -> AutoTokenizer:
        """
        Initializes the PyTorch tokenizer from a given pre-trained model path.

        This method loads the pre-trained tokenizer from the given path and
        sets up the tokenizer configurations according to the settings in
        the YAML files.

        Args:
            model_path (str): The path to the pre-trained model.

        Returns:
            AutoTokenizer: The initialized PyTorch tokenizer.
        """
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _pytorch_model_input(self, message, tokenizer) -> torch.Tensor:
        """
        Converts a message into a PyTorch tensor using the provided tokenizer.

        This function applies the chat template to the message, tokenizes it,
        and then converts it into a tensor suitable for model input.

        Args:
            message (str): The input message to be tokenized.
            tokenizer: The tokenizer to use for converting the message.

        Returns:
            torch.Tensor: The tokenized message as a PyTorch tensor.
        """
        # Apply the chat template to the message and tokenize it
        text = tokenizer.apply_chat_template(
            message,
            tokenize=self.tokenize,
            add_generation_prompt=self.add_generation_prompt
        )
        # Convert the tokenized text to a tensor and move it to the specified device
        return tokenizer(text, return_tensors=self.model_device).to(self.device)

    def _setup_generate_kwargs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets up the keyword arguments for the `generate` method of the PyTorch model.

        This method takes the model inputs as a dictionary and returns a new dictionary
        containing the keyword arguments for the `generate` method.

        Args:
            model_inputs (Dict[str, Any]): The model inputs as a dictionary.

        Returns:
            Dict[str, Any]: The keyword arguments for the `generate` method.
        """
        return dict(
            # The input IDs to pass to the model
            inputs=model_inputs.input_ids,
            # The attention mask to pass to the model
            attention_mask=model_inputs.attention_mask,
            # Whether to sample from the model's output probabilities
            do_sample=self.do_sample,
            # The maximum number of tokens to generate
            max_new_tokens=self.llm_max_token,
            # The temperature to use for sampling from the model's output probabilities
            temperature=self.llm_temperature,
            # The top-p value to use for sampling from the model's output probabilities
            top_p=self.llm_top_p,
            # The top-k value to use for sampling from the model's output probabilities
            top_k=self.llm_top_k,
            # The repetition penalty to use when generating text
            repetition_penalty=self.llm_repetition_penalty,
            # The number of n-grams to not repeat when generating text
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            # Whether to use caching to speed up the generation process
            use_cache=self.use_cache
        )

    def chat_with_sync(self, msg) -> str:
        """
        This function takes a user message and generates a response using a PyTorch model.

        Args:
            msg (str): The user message to generate a response for.

        Returns:
            str: The generated response in JSON format.

        """
        start_time = time.time()
        response_text = ''

        try:
            # Get the stop tokens for the generation process
            stop_tokens = [self.chat_tokenizer.eos_token_id]
            stop_tokens = [StopOnTokens(stop_tokens)]

            # Prepare the model inputs
            model_inputs = self._pytorch_model_input(msg, self.chat_tokenizer)

            # Prepare the keyword arguments for the `generate` method
            generate_kwargs = self._setup_generate_kwargs(model_inputs)

            # Add the stopping criteria and special tokens to the keyword arguments
            generate_kwargs.update({
                'stopping_criteria': StoppingCriteriaList(stop_tokens),
                'pad_token_id': self.chat_tokenizer.pad_token_id,
                'eos_token_id': self.chat_tokenizer.eos_token_id,
                'bos_token_id': self.chat_tokenizer.bos_token_id
            })

            # Generate text using the PyTorch model
            with torch.inference_mode(), torch.amp.autocast(nvidia_cuda):
                generated_ids = self.chat_model.generate(**generate_kwargs)

            # Remove the input IDs from the generated IDs
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            # Decode the generated IDs into text
            decoded_outputs = self.chat_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=self.skip_special_tokens
            )[0]

            if decoded_outputs is not None:
                # Add the generated response to the user message
                msg.append({"role": "assistant", "content": decoded_outputs})
                response_text = json.dumps(msg, ensure_ascii=False)
        except Exception as e:
            logger.error(e)
        finally:
            # Clean up the memory
            self.memory_manager.clean_memory()
            logger.info(f"totally use {time.time() - start_time} seconds")
        return json.loads(response_text)

    def chat_with_stream(self, msg) -> Iterator[Dict[str, Any]]:
        """
        This function takes a user message and generates a response using a PyTorch model.
        The response is generated in a streaming fashion, where each chunk of text is yielded
        as soon as it is generated.

        Args:
            msg (str): The user message to generate a response for.

        Yields:
            dict: A dictionary containing the generated text, the number of tokens in the
                  generated text, the total number of tokens generated so far, and the token
                  rate (i.e. the number of tokens generated per second).

        """
        start_time = time.time()
        token_count = 0
        buffer = ""

        try:
            # Prepare the model inputs
            model_inputs = self._pytorch_model_input(msg, self.chat_tokenizer)

            # Prepare the streamer
            streamer = TextIteratorStreamer(
                self.chat_tokenizer,
                timeout=self.llm_timeout,
                skip_prompt=self.skip_prompt,
                skip_special_tokens=self.skip_special_tokens,
            )

            # Start the generation thread
            generation_thread = Thread(target=self._generate_tokens, args=(model_inputs, streamer))
            generation_thread.start()

            # Iterate over the generated text and yield each chunk
            for new_text in streamer:
                if not new_text.strip():
                    continue
                buffer += new_text
                if len(buffer) >= 2 or new_text.endswith(('.', '!', '?', '\n')):
                    chunk_token_count = len(self.chat_tokenizer.encode(buffer))
                    token_count += chunk_token_count
                    yield {
                        'text': buffer,
                        'token_count': chunk_token_count,
                        'total_token_count': token_count,
                        'token_rate': token_count / (time.time() - start_time)
                    }
                    buffer = ""
            if buffer:
                yield {
                    'text': buffer,
                    'token_count': len(self.chat_tokenizer.encode(buffer)),
                    'total_token_count': token_count,
                    'token_rate': token_count / (time.time() - start_time)
                }
        except Exception as e:
            yield f"Error: {str(e)}"
        finally:
            # Clean up the memory
            self.memory_manager.clean_memory()
            
    def _generate_tokens(self, model_inputs: Dict[str, Any], streamer) -> None:
        """
        This function takes a model input and a streamer as input and generates tokens
        using the PyTorch model. The generated tokens are written to the streamer.

        Args:
            model_inputs (Dict[str, Any]): The input to the PyTorch model.
            streamer (TextIteratorStreamer): The streamer to write the generated tokens to.

        Raises:
            Exception: If an error occurs during generation.
        """
        try:
            # Get the stop tokens for the generation process
            stop_tokens = [self.chat_tokenizer.eos_token_id]
            stop_tokens = [StopOnTokens(stop_tokens)]
            
            # Prepare the keyword arguments for the `generate` method
            generate_kwargs = self._setup_generate_kwargs(model_inputs)
            generate_kwargs.update({
                'streamer': streamer,
                'stopping_criteria': StoppingCriteriaList(stop_tokens),
                'pad_token_id': self.chat_tokenizer.pad_token_id,
                'eos_token_id': self.chat_tokenizer.eos_token_id,
                'bos_token_id': self.chat_tokenizer.bos_token_id
            })
            # Generate tokens using the PyTorch model
            with torch.inference_mode(), torch.amp.autocast(nvidia_cuda):
                self.chat_model.generate(**generate_kwargs)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            
    def __del__(self):
        """
        Destructor for the ChatModel class.
        
        This method is automatically called when the instance is about to be destroyed.
        It stops the GPU memory monitoring and frees up any resources used by the instance.
        """
        # Stop the memory monitoring thread
        self.memory_manager.stop_monitoring()
