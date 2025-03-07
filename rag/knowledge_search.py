"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/rag/knowledge_search.py
Author: yuanzhenhui
Date: 2024-07-12 17:48:09
LastEditTime: 2025-01-14 22:34:33
"""

import threading
import time
from itertools import combinations
import json

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))
sys.path.append(os.path.join(project_dir, 'utils', 'llms'))

from elastic_util import ElasticUtil
from yaml_util import YamlConfig
from jieba_util import JiebaUtil
from baai_util import BaaiUtil

from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class KnowledgeSearch:
    
    _instance = None
    _initialized = False
    
    def __init__(self) -> None:
        """
        Initialize the knowledge search instance.
        
        The instance is only initialized once and subsequent calls to the
        constructor will return the same instance.
        """
        if not KnowledgeSearch._initialized:
            self._knowledge_search_init()
            KnowledgeSearch._initialized = True
            
    def __new__(cls, *args, **kwargs):
        """
        Creates a new instance of the KnowledgeSearch class.

        This method ensures that only one instance of the KnowledgeSearch class is created.
        If the instance does not exist, it creates a new one and assigns it to the `_instance` class variable.
        If an instance already exists, it returns the existing instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _knowledge_search_init(self) -> None:
        """
        Initializes the knowledge search instance.

        This method creates an instance of the ElasticUtil class to interact with Elasticsearch,
        an instance of the BaaiUtil class to perform reranking, and an instance of the JiebaUtil class to perform keyword extraction.
        It also reads the configuration of the Elasticsearch indices and creates the indices if they do not exist.
        """
        try:
            # Create an instance of the ElasticUtil class to interact with Elasticsearch
            self.es = ElasticUtil()
            # Create an instance of the BaaiUtil class to perform reranking
            self.bu = BaaiUtil()
            # Create an instance of the JiebaUtil class to perform keyword extraction
            self.jieba = JiebaUtil()
            
            # Read the configuration of the Elasticsearch indices
            self.elastic_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'elastic_cnf.yml'))
            # Get the common mapping configuration
            common_mapping = json.dumps(self.elastic_config.get_value('es.rag.common_mapping'))
            # Get the list of knowledge base items
            self.knowledge_base_array = self.elastic_config.get_value('es.rag.knowledge')
            
            # Loop through the knowledge base items and create the indices if they do not exist
            for knowledge_base_item in self.knowledge_base_array:
                knowledge_base = self.knowledge_base_array[knowledge_base_item]
                # Replace the placeholders in the common mapping configuration with the actual values
                knowledge_base_mapping = json.loads(common_mapping.replace("vector_field", knowledge_base['vector_field']).replace("text_field", knowledge_base['text_field']))
                # Create the index if it does not exist
                self.es.find_and_create_index(knowledge_base['index_name'],knowledge_base_mapping)
            logger.info("Knowledge search init completed...")
        except Exception as e:
            logger.error(f"Knowledge search init failed: {e}")
            raise
    
    def _get_keyword(self, want_to_ask) -> list:
        """
        Extracts keywords from the input text and converts them into vectors.

        This function utilizes the JiebaUtil instance to extract keywords from
        the provided text. If keywords are successfully extracted, they are
        then converted into vectors using the BaaiUtil instance.

        Args:
            want_to_ask (str): The input text from which to extract keywords.

        Returns:
            list: A list of vectors corresponding to the extracted keywords, 
                  or an empty list if no keywords are found.
        """
        # Extract keywords from the input text
        matches = self.jieba.get_keyword(want_to_ask)
        
        # Convert the extracted keywords to vectors if any keywords are found
        return self.bu.array_to_vetor(matches) if matches else []
    
    def find_summary_search(self, ask_content: list) -> list:
        """
        Finds related summary information from the knowledge database
        based on the input text and returns a formatted response.

        This function takes a list of question items as input and
        returns a list of three items, where the first two items are
        the two previous questions and the third item is the response
        to the current question.

        The function first extracts keywords from the current question
        and the last question, and then uses these keywords to search
        the knowledge database. The search results are then sorted
        based on relevance and the top 10 results are returned.

        The function also adds a formatted prompt to the response,
        which includes the current question, the search results, and
        a reminder to the user to ask more questions.

        Args:
            ask_content (list): A list of question items, where each item
                                is a dictionary containing the question
                                content and the role of the question (user
                                or assistant).

        Returns:
            list: A list of three items, where the first two items are
                  the two previous questions and the third item is the
                  response to the current question.
        """

        # Start timing
        search_start_time = time.time()

        # Extract keywords from the input text
        totally_ask = ''
        threads, total_summary = [], []

        dict_keyword = {}
        question_item_buffer = ''
        for item in ask_content[:-1]:
            if item['role'] == 'user':
                question_item_buffer += item['content']
        dict_keyword["last"] = self._get_keyword(question_item_buffer)

        current_ask = ask_content[-1]['content']
        dict_keyword["current"] = self._get_keyword(current_ask)

        # Search the knowledge database for related information
        for knowledge_base_item in self.knowledge_base_array:
            knowledge_base = self.knowledge_base_array[knowledge_base_item]
            thread = threading.Thread(target=self._vector_answer_search, args=(knowledge_base, dict_keyword, total_summary))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        # Log the search time
        logger.info(f" knowledge database query {len(total_summary)} records and totally use {time.time() - search_start_time} secords")

        if len(total_summary) > 0:
            # Sort the search results based on relevance
            rerank_array = self.bu.array_to_reranker(current_ask, total_summary)
            rerank_str = "\n".join(rerank_array)
            totally_ask = f"""您是一位资深的中药学专家，拥有扎实的中医药理论基础、丰富的临床经验和专业的品质鉴别能力，并且能够结合现代研究视角进行分析。
                
                参考信息：
                {rerank_str}

                当前问题：
                {current_ask}

                重要提示：这是流式输出！必须严格按照以下格式要求！

                一、格式规范：
                1. 列表项格式（最重要）：
                使用数字编号，如：
                1. 第一点内容
                
                2. 第二点内容
                
                3. 第三点内容

                2. 段落格式：
                - 使用自然流畅的语言
                - 不使用标题和表格
                - 禁止使用方括号[]
                - 全部使用简体中文

                3. 结尾格式：
                段落结束后空一行
                添加："若想了解更多，您可以继续提问哦。"
                最后一行：["问题1","问题2","问题3"]
                （每个问题不超过20字）

                示例回答：
                新鲜人参的处理方法如下：

                1. 挑选新鲜人参时，应选择外表饱满、无虫蛀痕迹的

                2. 将人参表面的泥土轻轻刷洗干净

                3. 切片时要求厚度均匀，建议约2-3毫米

                接下来可以根据个人需求选择烹饪方式。

                若想了解更多，您可以继续提问哦。
                ["人参泡水的最佳时间","人参搭配哪些食材最佳","人参有哪些食用禁忌"]

                内容要求：
                - 结合参考信息和专业知识
                - 使用生动例子和形象比喻
                - 语言既专业又通俗易懂
                - 补充实用的生活建议
                """
           
            ask_content[-1]['content'] = totally_ask
        return ask_content[-3:]
        
    def _vector_answer_search(self, knowledge_base, dict_keyword, total_summary):
        """
        Searches the knowledge base using vector search and returns the results.

        :param knowledge_base: The knowledge base to search
        :param dict_keyword: The keyword dictionary
        :param total_summary: The summary of the answers
        :return: The updated summary of the answers
        """
        
        # Get the index name and vector field from the knowledge base
        index_name = knowledge_base['index_name']
        vector_field = knowledge_base['vector_field']
        text_field = knowledge_base['text_field']
        
        # Get the last keyword and set the number of results to retrieve
        last_keyword = dict_keyword["last"]
        last_top_k = len(last_keyword)
        if last_top_k > 0:
            # Increase the number of results for the last keyword
            last_top_k = last_top_k*10
            response_arr = self._find_elastic(index_name,vector_field,last_keyword,last_top_k)
            # Add the results to the total summary
            total_summary.extend(f"{response[text_field]}" for response in response_arr)
        else:
            # Set the number of results for the last keyword to 10 if there are no keywords
            last_top_k = 10
        
        # Get the current keyword and set the number of results to retrieve
        current_keyword = dict_keyword["current"]
        current_top_k = last_top_k * 3
        if len(current_keyword) > 0:
            # Search the knowledge base using vector search and retrieve the results
            response_arr = self._find_elastic(index_name,vector_field,current_keyword,current_top_k)
            # Add the results to the total summary
            total_summary.extend(f"{response[text_field]}" for response in response_arr)
        
    def _find_elastic(self, index_name, vector_field, keyword_array, top_k):
        """
        Perform a vector search on the Elasticsearch index using the provided keywords.

        Args:
            index_name (str): The name of the Elasticsearch index.
            vector_field (str): The field in the index containing vector data.
            keyword_array (list): A list of keywords to use in the search.
            top_k (int): The number of top results to retrieve.

        Returns:
            list: A list of documents matching the search criteria.
        """
        response_arr, should_queries = [], []
        # Calculate the minimum score threshold based on the number of keywords
        min_score_threshold = len(keyword_array) * 0.6
        
        # Generate combinations of keywords and create should queries
        for i in range(len(keyword_array), 0, -1):
            for combo in combinations(keyword_array, i):
                query_vectors = list(combo)
                # Construct the script to calculate cosine similarity
                script_source = "double score = 0; "
                script_source += "".join([
                    f"score += Math.max(0, cosineSimilarity(params.query_vector{j}, params.vector_field)); "
                    for j in range(len(query_vectors))
                ])
                script_source += "return score;"

                # Add the script score query to the should queries
                should_queries.append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": script_source,
                            "params": {
                                **{f"query_vector{j}": query_vectors[j] for j in range(len(query_vectors))},
                                "vector_field": vector_field
                            }
                        }
                    }
                })

        # Construct the search body for Elasticsearch query
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": should_queries,
                    "boost": 1.0
                }
            },
            "min_score": min_score_threshold
        }

        # Perform the search query and process the results
        responses = self.es.find_by_body(index_name, body)
        response_arr.extend([response['_source'] for response in responses])
        return response_arr
