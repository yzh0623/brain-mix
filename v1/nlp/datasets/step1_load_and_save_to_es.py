"""
FilePath: /brain-mix/nlp/datasets/step1_load_and_save_to_es.py
Author: yuanzhenhui
Date: 2025-09-22 10:06:01
LastEditTime: 2025-10-10 21:17:45
"""

from tqdm import tqdm
import threading
import requests
import json
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))
sys.path.append(os.path.join(project_dir, 'nlp'))
import tiktoken

from common_util import CommonUtil
import const_util as CU

from persistence.elastic_util import ElasticUtil
from persistence.mysql_util import MysqlUtil
from thirdparty.silicon_util import SiliconUtil
from thirdparty.embedding_util import EmbeddingUtil

from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class LoadAndSaveToEs:
    
    def __init__(self):
        """
        Initialize the LoadAndSaveToEs class.
        
        This class is used to load the datasets from the local file system and save them to Elasticsearch.
        
        :param base_path: The base path of the datasets.
        :type base_path: str
        :param dirs_name: The list of folder names in the base path.
        :type dirs_name: list
        """
        
        nlp_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'nlp_cnf.yml')
        base_path = YamlUtil(nlp_cnf).get_value('datasets.base_path')
        dirs_name = YamlUtil(nlp_cnf).get_value('datasets.dir_name')
        
        self.embeddings_url = YamlUtil(nlp_cnf).get_value('embeddings.url')
        self.datasets_path = {}
        for dir_name in dirs_name:
            folder_path = os.path.join(base_path, dir_name)
            self.datasets_path[dir_name] = CommonUtil.get_file_paths(folder_path,".json")
        
        """
        The Elasticsearch client
        """
        self.elastic = ElasticUtil()
        
        self.mysql = MysqlUtil()
        
        """
        The DataEmbedding class is used to get the embedding of a given text.
        """
        self.embedding = EmbeddingUtil()
        
        """
        The tiktoken encoding used to get the length of a given text.
        """
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        """
        The SiliconUtil class is used to get the scores of a given text from the Silicon API.
        """
        self.api = SiliconUtil()
        
        utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'utils_cnf.yml')
        self.embedding_model = YamlUtil(utils_cnf).get_value('silicon.agent.content_embedding.model')
        
        """
        Create the Elasticsearch index and mapping.
        
        The mapping is used to define the structure of the index.
        """
        es_gather_qa_mapping = YamlUtil(nlp_cnf).get_value('datasets.tmp_gather_db')
        self.elastic.create_index(name=CU.TMP_ES_INDEX, mapping=es_gather_qa_mapping[CU.TMP_ES_INDEX])
    
    def save_hwtcm_deepseek_data(self):
        """
        Save the data from the hwtcm-deepseek dataset to ES
        """
        dataset_array = CommonUtil.load_json_file(self.datasets_path["hwtcm-deepseek"][0])
        
        qa_array = []
        for dataset in dataset_array:
            question = dataset["instruction"]
            answer = dataset["output"]
            answer = answer.split("</think>")[1]
            answer = answer.strip().replace("\n", " ")
            qa_array.append({
                "question": question, 
                "answer": answer,
                "data_source": "hwtcm-deepseek"
                })
        
        self._save_to_es(qa_array)
        
    def save_hwtcm_sft_data(self):
        """
        Save the data from the hwtcm-sft-v1 dataset to ES
        
        This function loads the data from the hwtcm-sft-v1 dataset and saves it to ES.
        
        The data in the hwtcm-sft-v1 dataset is in the form of a JSON file, where each line is a
        dictionary containing a question and an answer. The question is in the form of a string, and
        the answer is in the form of a string. We need to strip the answer of any leading or trailing
        whitespace and replace any newline characters with spaces.
        
        After loading the data, we create a list of dictionaries, where each dictionary contains a
        question and an answer. We then save this list to ES using the _save_to_es function.
        """
        dataset_array = CommonUtil.load_json_file(self.datasets_path["hwtcm-sft-v1"][0])
        
        qa_array = []
        for dataset in dataset_array:
            question = dataset["instruction"]
            answer = dataset["output"]
            answer = answer.strip().replace("\n", " ")
            qa_array.append({
                "question": question, 
                "answer": answer,
                "data_source": "hwtcm-sft-v1"
                })
        
        self._save_to_es(qa_array)
        
    def save_shennong_tcm_data(self):
        """
        Save the data from the shennong-tcm dataset to ES
        
        The data in the shennong-tcm dataset is in the form of a list of dictionaries, where each
        dictionary contains a question and an answer. The question is in the form of a string, and
        the answer is in the form of a list of strings. We need to split the question into multiple
        parts and remove the parts that are not necessary.
        """
        dataset_array = CommonUtil.load_json_file(self.datasets_path["shennong-tcm"][0])
        
        qa_array = []
        for dataset in dataset_array:
            question = dataset["query"]
            # Split the question into multiple parts
            question_parts = question.split("要求：")
            # Remove the parts that are not necessary
            question = question_parts[0]
            
            answer = dataset["response"]
            # Remove the newline characters and strip the leading and trailing spaces
            answer = answer.strip().replace("\n", " ")
            qa_array.append({
                "question": question, 
                "answer": answer,
                "data_source": "shennong-tcm"
                })
        
        self._save_to_es(qa_array)
        
    def mysql_turning_data_to_es(self):
        """
        Save the question answer pairs in the TMP_MYSQL_TURNING_DATA_TABLE table to ES.

        This function will query the TMP_MYSQL_TURNING_DATA_TABLE table to get the records that
        have not been processed (generate_flag = 0). It will then split the content of the records
        into multiple parts and use multiple threads to convert the text to vectors using the
        embedding model. After that, it will save the data to ES.

        The generate_flag of the processed records will be updated to 1. The records that failed
        to be processed will be updated to 2.
        """
        search_sql = f"""
            select id,content,sources
            from {CU.TMP_MYSQL_TURNING_DATA_TABLE}
            where generate_flag = 0
            limit 10000
        """
        counter = 1
        results = self.mysql.query_by_list(search_sql)
        while len(results)>0:
            logger.info(f"Processing batch {counter}")
            update_ids = []
            update_not_ids = []
            qa_array = []
            
            for id, content, sources in results:
                try:
                    # Try to parse the content as a JSON object
                    json_content = json.loads(content)
                    # If the content is a valid JSON object, append it to the qa_array
                    qa_array.append({"question": json_content["question"], "answer": json_content["answer"],"data_source": sources})
                    # Add the id to the update_ids list
                    update_ids.append(id)
                except:
                    # If the content is not a valid JSON object, add the id to the update_not_ids list
                    update_not_ids.append(id)
                
            if qa_array:
                # Call the _save_to_es function to save the data to ES
                self._save_to_es(qa_array)
                # Update the generate_flag of the processed records to 1
                if update_ids:
                    update_sql = f"""
                        update {CU.TMP_MYSQL_TURNING_DATA_TABLE}
                        set generate_flag = 1
                        where id in ({','.join(map(str, update_ids))})
                    """
                    self.mysql.save_or_update(update_sql)
            
            # Update the generate_flag of the failed records to 2
            if update_not_ids:
                update_sql = f"""
                        update {CU.TMP_MYSQL_TURNING_DATA_TABLE}
                        set generate_flag = 2
                        where id in ({','.join(map(str, update_not_ids))})
                    """
                self.mysql.save_or_update(update_sql)    
                    
            # Query the database again to get the next batch of records
            results = self.mysql.query_by_list(search_sql)
            counter += 1
            
    def _save_to_es(self, qa_array):
        """
        Save the question answer pairs to ES.

        This function will split the input list into multiple parts and use multiple threads to
        convert the text to vectors using the embedding model. After that, it will save the data to
        ES.

        Args:
            qa_array (list): A list of question answer pairs in the form of a dictionary.
        """
        if qa_array:
            # Split the input list into multiple parts
            qa_split_array = CommonUtil.split_array(qa_array, 5)
            search_threads, qa_batch_array = [], []
            
            # Split the list again to create multiple threads
            qa_split_result = [
                qa_split_array[0],
                [item for sublist in qa_split_array[1:] for item in sublist]
            ]
            
            # Create multiple threads to convert the text to vectors
            for idx, qa_split in enumerate(qa_split_result):
                search_thread = threading.Thread(
                    target=self._thread_to_get_vectors,
                    args=(qa_split, qa_batch_array, idx),
                    daemon=True
                )
                search_threads.append(search_thread)
                search_thread.start()
            
            # Wait for all threads to finish
            for search_thread in search_threads:
                search_thread.join()
            
            # Save the data to ES
            self.elastic.batch_insert(CU.TMP_ES_INDEX, qa_batch_array)
    
    def _thread_to_get_vectors(self,qa_split,qa_batch_array,flag):
        """
        A function that gets called by multiple threads to convert the text to vectors.
        This function is used to convert a batch of text to vectors using the embedding model.

        Args:
            qa_split (list): A list of question answer pairs in the form of a dictionary.
            qa_batch_array (list): A list to store the result.
            flag (int): An int that is used to decide which embedding model to use.
        """
        for qa_json in tqdm(qa_split, desc="Now changing to vectors..."):
            # Create a new field in the dictionary which contains the question and answer
            # This is for the embedding model to use
            qa_json["gather_text"] = f"【问题】{qa_json['question']}【答案】{qa_json['answer']}"
            content = qa_json["gather_text"]
            
            # Use the embedding model to convert the text to a vector
            vector_content = self._get_embedding(content, flag)
            if vector_content is None:
                continue
            qa_json[CU.TMP_ES_VECTOR_FIELDS] = vector_content
                    
            # Set the process status to 0
            qa_json["process_status"] = 0
            qa_batch_array.append(qa_json)
    
    def _get_embedding(self, content, flag):
        """
        A function that gets the embeddings of a given content.

        This function takes a content string and a flag as input. 
        The flag is used to decide which embedding model to use. 
        If the length of the content is larger than 400, use the online service to get the embeddings.
        Otherwise, use the local embedding model.

        Args:
            content (str): The content to get the embeddings for.
            flag (int): A flag that is used to decide which embedding model to use.

        Returns:
            list or None: The embeddings of the content or None if the content is too long.
        """
        if flag == 0 or len(self.enc.encode(content)) >= 400:
            # If the length of the content is larger than 400, use the online service to get the embeddings
            return self._request_embedding(content)
        else:
            # Use the local embedding model to get the embeddings
            embed_array = self.api.embedding_with_sync(self.embedding_model, [content])
            return embed_array[0] if embed_array else None
    
    def _request_embedding(self,text):
        """
        Request the embedding of a given text from the server.

        This function sends a POST request to the server with the given text and
        receives the embedding of the text in return.

        Args:
            text (str): The text to request the embedding for.

        Returns:
            list or None: The embedding of the text if the request was successful,
                otherwise None.
        """
        ret_msg = None
        # Send the request
        try:
            respnse = requests.post(self.embeddings_url, json={"text_array": [text], "use_large": True})
            if respnse.status_code == 200:
                # Parse the response
                ret_msg = json.loads(respnse.text)["result"][0]
        except Exception as e:
            pass
        return ret_msg
    
if __name__ == "__main__":
    laste = LoadAndSaveToEs()
    # laste.save_hwtcm_deepseek_data()
    # laste.save_hwtcm_sft_data()
    # laste.save_shennong_tcm_data()
    laste.mysql_turning_data_to_es()