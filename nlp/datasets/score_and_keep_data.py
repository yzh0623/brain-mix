import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import re
import time
import random
import math
import threading
import const_util as CU
from yaml_util import YamlUtil
from api_util import ApiUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.elastic_util import ElasticUtil

class ScoreAndKeepData:
    
    def __init__(self):
        self.elastic = ElasticUtil()
        self.api = ApiUtil()
        
        silicon_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.content_scores = YamlUtil(silicon_cnf).get_value('silicon.agent.content_score')
        
    def search_for_score(self):
        """
        Search all the records in the tmp_gather_db index where process_status is 0,
        and then use the Silicon API to get the scores for the text in the gather_text field.
        After getting the scores, update the records with the scores and change the process_status to 1.

        This function is called by the schedule job every 30 minutes,
        and it will search all the records in the tmp_gather_db index and get their scores.
        It will then update the records with the scores and change the process_status to 1.
        """
        # Search all the records in the tmp_gather_db index where process_status is 0
        
        search_not_ready = {
            "size": CU.SCORE_THREADS_SIZE,
            "query": {
                "term": {
                    "process_status": {
                        "value": 0
                    }
                }
            }
        }
        results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX,body=search_not_ready)
        batch_count = 1
        while len(results)>0:
            start_time = time.time()
            search_threads,update_array = [],[]
            for result in results:
                logger.info(f"search_for_score: {result['_id']}")
                # Start a thread to get the score from silicon
                search_thread = threading.Thread(
                    target=self._thread_to_silicon_get_score, 
                    args=(result, update_array),
                    daemon=True
                    )
                search_threads.append(search_thread)
                search_thread.start()
            for search_thread in search_threads:
                search_thread.join()
                
            if update_array:
                logger.info(f"------> search_for_score {batch_count} batch search use time:{math.ceil(time.time() - start_time)} seconds.")
                batch_count += 1
                for update_entity in update_array:
                    es_id = update_entity["id"]
                    avg_score = round((int(update_entity["qwen"])+ int(update_entity["glm"]) + int(update_entity["ds"]))/3,2)
                    update_data = {
                        "doc": {
                            "process_status": 1,
                            "qwen_score": update_entity["qwen"],
                            "glm_score": update_entity["glm"],
                            "ds_score": update_entity["ds"],
                            "avg_score": avg_score
                        }
                    }
                    self.elastic.update(name=CU.TMP_ES_INDEX,data=update_data,id=es_id)
                    logger.info(f"{es_id} update success...")
            # Search all the records in the tmp_gather_db index where process_status is 0 again
            time.sleep(random.randint(3,8))
            results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX,body=search_not_ready)
                
    def _thread_to_silicon_get_score(self, result, update_array):
        """
        A thread that gets the score of a document from silicon and updates the document.

        Args:
            result (dict): The result from ES.
            update_array (list): The array to store the updated documents.
        """
        id = result["_id"]
        gather_text = result["_source"]["gather_text"]
        
        start_time = time.time()
        # Start threads to get scores from different LLMs
        llm_threads, resp_array = [], []
        for key, params in self.content_scores.items():
            llm_thread = threading.Thread(
                    target=self._get_and_check_digit_return,
                    args=(gather_text, key, params, resp_array),
                    daemon=True
                    )
            llm_threads.append(llm_thread)
            llm_thread.start()
            
        # Wait for all threads to finish
        for llm_thread in llm_threads:
            llm_thread.join()
        
        # Update the document with the scores
        if resp_array:
            logger.info(f"------> _thread_to_silicon_get_score finished getting scores for {id} use time:{math.ceil(time.time() - start_time)} seconds.")
            resp_entiry = {"id": id}
            for resp in resp_array:
                for key, value in resp.items():
                    resp_entiry[key] = value
            update_array.append(resp_entiry)
        time.sleep(random.randint(10,30))
    
    def _get_and_check_digit_return(self, gather_text, key, params, resp_array):
        """
        A function that gets the score of a document from silicon and checks if it is a digit.

        Args:
            gather_text (str): The text of the document.
            key (str): The key of the LLM.
            params (dict): The parameters for the LLM.
            resp_array (list): The array to store the response.
        """
        # Get the prompt for the LLM
        prompt_str = self._llm_get_score_prompts(gather_text)
        
        digit_batch_count = 1
        # Keep trying until we get a valid score
        while True:
            
            start_time = time.time()
            try:
                # Get the score from the LLM
                score = self.api.chat_with_sync(params, prompt_str)
                logger.info(f"------> _get_and_check_digit_return got score in {digit_batch_count} batch from LLM {key}: {score} use time:{math.ceil(time.time() - start_time)} seconds.")
                
                # Remove any non-digit characters from the score
                score = re.sub(r'[^\w\u4e00-\u9fff]', '', score)
                
                # Check if the score is a digit
                if score.isdigit():
                    
                    # Add the score to the response array
                    resp_array.append({key: int(score)})
                    logger.info(f"------> _get_and_check_digit_return normal {digit_batch_count} batch use time:{math.ceil(time.time() - start_time)} seconds.")
                    
                    # Break out of the loop
                    break
                else:
                    logger.info(f"------> _get_and_check_digit_return not digit {digit_batch_count} batch use time:{math.ceil(time.time() - start_time)} seconds.")
                    time.sleep(random.randint(15,30))
            except Exception as e:
                logger.info(f"------> _get_and_check_digit_return exception {digit_batch_count} batch use time:{math.ceil(time.time() - start_time)} seconds.")
                # If there is an error, just ignore it and try again
                logger.info(f"Error getting score from LLM {key}: {e}")
                time.sleep(random.randint(15,30))
                
            digit_batch_count += 1
    
    def _llm_get_score_prompts(self,qa_content):
        return f"""
            我将提供一条中医药领域的“问答对”（包含问题和回答）。  
            你的任务是：  
            1. 只根据问答对的完整性、准确性、逻辑性和专业性进行质量评估。  
            2. 给出一个 **0 到 10 之间的分数**（10 分表示极高质量，0 分表示极低质量）。  
            3. 只返回一个阿拉伯数字，不要输出任何解释或其他内容。  

            问答对如下：  
            
            {qa_content}

            请直接输出一个 0-10 的整数，不要输出任何解释或符号。
        """
        
if __name__ == "__main__":
    s = ScoreAndKeepData()
    s.search_for_score()