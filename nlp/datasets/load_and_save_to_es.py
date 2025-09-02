import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))
sys.path.append(os.path.join(project_dir, 'nlp'))
import tiktoken

from common_util import CommonUtil
from api_util import ApiUtil
from tqdm import tqdm
import const_util as CU
import threading

from models.embedding.data_embedding import DataEmbedding

from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.elastic_util import ElasticUtil


class LoadAndSaveToEs:
    def __init__(self):
        
        nlp_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'nlp_cnf.yml')
        base_path = YamlUtil(nlp_cnf).get_value('datasets.base_path')
        dirs_name = YamlUtil(nlp_cnf).get_value('datasets.dir_name')
        self.datasets_path = {}
        for dir_name in dirs_name:
            folder_path = os.path.join(base_path, dir_name)
            self.datasets_path[dir_name] = CommonUtil.get_file_paths(folder_path,".json")
        
        self.elastic = ElasticUtil()
        self.embedding = DataEmbedding()
        
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.api = ApiUtil()
        utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'utils_cnf.yml')
        self.embedding_model = YamlUtil(utils_cnf).get_value('silicon.agent.content_embedding.model')
        
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
        """
        dataset_array = CommonUtil.load_json_file(self.datasets_path["shennong-tcm"][0])
        
        qa_array = []
        for dataset in dataset_array:
            question = dataset["query"]
            question = question.split("要求：")[0]
            
            answer = dataset["response"]
            answer = answer.strip().replace("\n", " ")
            qa_array.append({
                "question": question, 
                "answer": answer,
                "data_source": "shennong-tcm"
                })
        
        self._save_to_es(qa_array)
        
    def save_five_phases_mindset_data(self):
        dataset_array = CommonUtil.load_json_file(self.datasets_path["five-phases-mindset"][0])
        
        qa_array = []
        for dataset in dataset_array:
            question = dataset["input"]
            answer = dataset["output"]
            answer = answer.strip().replace("\n", " ")
            qa_array.append({
                "question": question, 
                "answer": answer,
                "data_source": "five-phases-mindset"
                })
        
        self._save_to_es(qa_array)
    
    def _save_to_es(self,qa_array):
        """
        Save the question answer pairs to ES.

        Args:
            qa_array (list): A list of question answer pairs in the form of a dictionary.
        """
        if qa_array:
            qa_split_array = CommonUtil.split_array(qa_array,5)
            search_threads,qa_batch_array = [],[]
            
            qa_split_result = [
                qa_split_array[0],
                [item for sublist in qa_split_array[1:] for item in sublist]
            ]
            
            for idx, qa_split in enumerate(qa_split_result):
                search_thread = threading.Thread(
                    target=self._thread_to_get_vectors, 
                    args=(qa_split, qa_batch_array,idx),
                    daemon=True
                    )
                search_threads.append(search_thread)
                search_thread.start()
            for search_thread in search_threads:
                search_thread.join()
            
            # Save the data to ES
            self.elastic.batch_insert(CU.TMP_ES_INDEX,qa_batch_array)
    
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
            # Use the embedding model to convert the text to a vector
            if flag == 0:
                qa_json[CU.TMP_ES_VECTOR_FIELDS] = CommonUtil.request_embedding(qa_json["gather_text"])
            else:
                content = qa_json["gather_text"]
                if len(self.enc.encode(content))< 400:
                    qa_json[CU.TMP_ES_VECTOR_FIELDS] = self.api.embedding_with_sync(self.embedding_model,[content])[0]
                else:
                    qa_json[CU.TMP_ES_VECTOR_FIELDS] = self.embedding.array_to_embedding([content])[0]
                    
            # Set the process status to 0
            qa_json["process_status"] = 0
            qa_batch_array.append(qa_json)
    
if __name__ == "__main__":
    laste = LoadAndSaveToEs()
    # laste.save_hwtcm_deepseek_data()
    # laste.save_hwtcm_sft_data()
    laste.save_shennong_tcm_data()
    laste.save_five_phases_mindset_data()