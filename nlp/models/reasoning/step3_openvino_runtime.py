import openvino_genai as ov_genai
from threading import Thread, Lock, Semaphore
from queue import Queue, Empty
from typing import Dict, Any, Generator, List, Union
import json
import time
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class OpenvinoRuntime:
    
    _instance = None
    _initialized = False
    _init_lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not OpenvinoRuntime._initialized:
            with OpenvinoRuntime._init_lock:
                if not OpenvinoRuntime._initialized:
                    try:
                        self._load_config()
                        self._init_pipeline()
                        OpenvinoRuntime._initialized = True
                    except Exception as e:
                        logger.error(f"Failed to initialize OpenVINO GenAI: {e}")
                        raise e

    def _load_config(self):
        """加载配置并构建 GenAI 所需的 GenerationConfig"""
        self.nlp_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        nlp_cnf = YamlUtil(self.nlp_cnf)
        
        # 加载量化 int4 的模型路径
        self.model_path = os.path.join(nlp_cnf.get_value('models.reasoning.openvino.model') ,"INT4")
        
        # 读取并发限制
        max_workers = nlp_cnf.get_value('models.reasoning.openvino.genai.worker_threads')
        # 信号量：用于控制同时进入 C++ 推理引擎的请求数
        self.semaphore = Semaphore(max_workers)
        
        # 推理设备
        self.device = nlp_cnf.get_value('models.reasoning.openvino.model_device')

        # 构建 GenerationConfig
        # 这些参数直接传递给 C++ 引擎
        self.gen_config = ov_genai.GenerationConfig()
        self.gen_config.max_new_tokens = nlp_cnf.get_value('models.reasoning.openvino.genai.max_tokens')
        self.gen_config.temperature = nlp_cnf.get_value('models.reasoning.openvino.genai.temperature')
        self.gen_config.top_p = nlp_cnf.get_value('models.reasoning.openvino.genai.top_p')
        self.gen_config.top_k = nlp_cnf.get_value('models.reasoning.openvino.genai.top_k')
        self.gen_config.repetition_penalty = nlp_cnf.get_value('models.reasoning.openvino.genai.repetition_penalty')
        self.gen_config.do_sample = nlp_cnf.get_value('models.reasoning.openvino.do_sample')

    def _init_pipeline(self):
        logger.info(f"Loading OpenVINO GenAI Pipeline from {self.model_path} ...")
        start_t = time.time()
        
        # LLMPipeline 是核心类，它会自动加载 Tokenizer 和 模型
        self.pipe = ov_genai.LLMPipeline(self.model_path, self.device)
        
        # 获取 Tokenizer 用于处理聊天模板
        self.tokenizer = self.pipe.get_tokenizer()
        logger.info(f"Pipeline loaded in {time.time() - start_t:.2f}s")

    def _apply_template(self, message: Union[str, List[Dict]]) -> str:
        """
        处理输入格式。
        如果输入是字符串，直接当作 Prompt (或者封装成 user message)。
        如果输入是 List (聊天记录)，使用 apply_chat_template 转换。
        """
        try:
            if isinstance(message, str):
                # 如果只是简单的一句话，封装成标准格式
                messages = [{"role": "user", "content": message}]
                return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            elif isinstance(message, list):
                return self.tokenizer.apply_chat_template(message, add_generation_prompt=True)
            else:
                return str(message)
        except Exception as e:
            logger.warning(f"Template application failed, using raw string: {e}")
            return str(message)

    def _count_tokens(self, text: str) -> int:
        """
        使用 OpenVINO GenAI Tokenizer 计算 token 数量
        """
        if not text:
            return 0
        try:
            # openvino-genai tokenizer encode 返回的是 TokenizedString
            encoded = self.tokenizer.encode(text)
            if hasattr(encoded, 'input_ids'):
                return len(encoded.input_ids)
            elif isinstance(encoded, list):
                return len(encoded)
            else:
                logger.debug(f"Unknown tokenizer output type: {type(encoded)}")
                return len(text) // 3  
        except Exception:
            return 0

    def transfor_msg(self, msg: Union[str, List[Dict]]) -> Dict:
        """
        同步非流式接口 (简单封装流式)
        """
        full_content = ""
        full_reasoning = ""
        
        for chunk in self.transfor_stream_msg(msg):
            if "content" in chunk:
                full_content += chunk["content"]
            if "reasoning_content" in chunk:
                full_reasoning += chunk["reasoning_content"]
                
        # 按照你的旧格式返回列表
        return [
            {"role": "assistant", "content": full_content, "reasoning_content": full_reasoning}
        ]

    def transfor_stream_msg(self, msg: Union[str, List[Dict]]):
        """
        流式推理 (桥接 C++ Callback 到 Python Generator)
        """
        prompt = self._apply_template(msg)
        
        # 创建一个线程安全的队列，用于存放 C++ 吐出来的 token
        token_queue = Queue()
        
        # 定义 C++ 回调函数
        # subword: 生成的片段
        # 作用: 将生成的片段放入队列，供主线程消费
        def streamer_callback(subword: str) -> bool:
            token_queue.put(subword)
            return False # 返回 False 表示继续生成，True 表示停止

        # 定义在子线程中运行的任务
        def run_inference():
            with self.semaphore: # 同样受并发限制
                try:
                    self.pipe.generate(prompt, self.gen_config, streamer_callback)
                except Exception as e:
                    logger.error(f"Stream Error: {e}")
                    token_queue.put(f"[ERROR: {e}]")
                finally:
                    # 放入 None 作为结束信号
                    token_queue.put(None)

        # 启动子线程运行 C++ 推理
        # 注意：这里必须用线程，否则 pipe.generate 会阻塞主线程，无法 yield
        thread = Thread(target=run_inference)
        thread.start()

        # 主线程：从队列中读取数据并 yield
        start_time = time.time()
        token_count = 0
        buffer = ""
        in_think = False
        
        # 定义可能的标签部分，用于防止切割
        # 如果 buffer 结尾是 "<", "<t", "<think" 等，先不 yield
        tag_start_marker = "<think>"
        tag_end_marker = "</think>"

        while True:
            try:
                # 等待新片段
                new_text = token_queue.get(timeout=self.gen_config.max_new_tokens * 1.0)
                
                if new_text is None:
                    break
                
                # 如果是错误信息
                if new_text.startswith("[ERROR:"):
                    yield {"content": new_text, "reasoning_content": ""}
                    break

                buffer += new_text
                
                # 循环处理 buffer，直到无法再分割
                while buffer:
                    processed_chunk = False # 标记本次循环是否处理了数据
                    
                    if not in_think:
                        # 检查是否有开始标签
                        start_idx = buffer.find(tag_start_marker)
                        if start_idx != -1:
                            # 1. 发现 <think>
                            # yield 标签前的内容 (普通 content)
                            if start_idx > 0:
                                chunk = buffer[:start_idx]
                                chunk_len = self._count_tokens(chunk)
                                token_count += chunk_len
                                yield {
                                    "content": chunk,
                                    "reasoning_content": "",
                                    "token_count": chunk_len,
                                    "total_token_count": token_count,
                                    "token_rate": token_count / (time.time() - start_time)
                                }
                            
                            # 移除已处理部分和标签
                            buffer = buffer[start_idx + len(tag_start_marker):]
                            in_think = True
                            processed_chunk = True
                        else:
                            # 没有发现完整标签，检查是否有"半截"标签的风险
                            # 例如 buffer 结尾是 "<thi"
                            partial_match = False
                            for i in range(1, len(tag_start_marker)):
                                if buffer.endswith(tag_start_marker[:i]):
                                    partial_match = True
                                    break
                            
                            if partial_match:
                                # 有风险，跳出内部循环，等待下一个 new_text 拼接
                                break 
                            
                            # 安全，全部 yield
                            if buffer:
                                chunk_len = self._count_tokens(buffer)
                                token_count += chunk_len
                                yield {
                                    "content": buffer,
                                    "reasoning_content": "",
                                    "token_count": chunk_len,
                                    "total_token_count": token_count,
                                    "token_rate": token_count / (time.time() - start_time)
                                }
                                buffer = ""
                                processed_chunk = True

                    else: # if in_think
                        # 检查是否有结束标签
                        end_idx = buffer.find(tag_end_marker)
                        if end_idx != -1:
                            # 1. 发现 </think>
                            # yield 标签前的内容 (思考 content)
                            chunk = buffer[:end_idx]
                            if chunk:
                                chunk_len = self._count_tokens(chunk)
                                token_count += chunk_len
                                yield {
                                    "content": "",
                                    "reasoning_content": chunk,
                                    "token_count": chunk_len,
                                    "total_token_count": token_count,
                                    "token_rate": token_count / (time.time() - start_time)
                                }
                            
                            # 移除已处理部分和标签
                            buffer = buffer[end_idx + len(tag_end_marker):]
                            in_think = False
                            processed_chunk = True
                        else:
                            # 同样检查半截结束标签 "</th..."
                            partial_match = False
                            for i in range(1, len(tag_end_marker)):
                                if buffer.endswith(tag_end_marker[:i]):
                                    partial_match = True
                                    break
                            
                            if partial_match:
                                break
                            
                            # 安全，全部 yield 为思考内容
                            if buffer:
                                chunk_len = self._count_tokens(buffer)
                                token_count += chunk_len
                                yield {
                                    "content": "",
                                    "reasoning_content": buffer,
                                    "token_count": chunk_len,
                                    "total_token_count": token_count,
                                    "token_rate": token_count / (time.time() - start_time)
                                }
                                buffer = ""
                                processed_chunk = True
                    
                    if not processed_chunk:
                        # 如果没有处理任何数据（通常是因为等待半截标签），强制跳出等待更多输入
                        break

            except Empty:
                break
            except Exception as e:
                logger.error(f"Generator Loop Error: {e}")
                break

        # 处理剩余的 buffer (循环结束后的残余)
        if buffer:
            chunk_len = self._count_tokens(buffer)
            token_count += chunk_len
            if in_think:
                yield {
                    "content": "",
                    "reasoning_content": buffer,
                    "token_count": chunk_len,
                    "total_token_count": token_count,
                    "token_rate": token_count / (time.time() - start_time)
                }
            else:
                yield {
                    "content": buffer,
                    "reasoning_content": "",
                    "token_count": chunk_len,
                    "total_token_count": token_count,
                    "token_rate": token_count / (time.time() - start_time)
                }
        
        thread.join()
        
        
if __name__ == '__main__':
    llm = OpenvinoRuntime()
    # 测试一下带有思考过程的 prompt (假设模型支持)
    prompt = "请详细分析一下中医理论中'阴阳'的概念，并给出思考过程。 /think"
    
    print("--- Start ---")
    for chunk in llm.transfor_stream_msg(prompt):
        if chunk.get('reasoning_content'):
            print(f"\033[90m{chunk['reasoning_content']}\033[0m", end="", flush=True) # 灰色打印思考
        else:
            print(chunk['content'], end="", flush=True)
    print("\n--- Done ---")