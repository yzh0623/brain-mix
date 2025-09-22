"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/nlp/datasets/4.score_and_filter_data.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-10 16:10:52
"""

import re
import time
import random
import math
import os
import sys
import threading

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

from thirdparty.silicon_util import SiliconUtil
from persistence.elastic_util import ElasticUtil

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class ScoreAndFilterData:
    
    def __init__(self):
        """
        Initialize the ScoreAndFilterData class.
        """
        # Get the Elasticsearch client
        self.elastic = ElasticUtil()
        
        # Get the SiliconUtil instance
        self.api = SiliconUtil()
        
        # Read the Silicon API configuration
        silicon_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.content_scores = YamlUtil(silicon_cnf).get_value('silicon.agent.content_score')
        
    def search_for_score(self):
        """
        Search for documents in ES that need to be scored and score them using Silicon API.
        """
        # Search for documents that need to be scored
        search_not_ready = {
            "size": 1,
            "query": {
                "term": {
                    "process_status": {
                        "value": 0
                    }
                }
            }
        }
        results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX, body=search_not_ready)
        batch_count = 1
        while len(results) >0:
            start_time = time.time()
            update_entity = self._thread_to_silicon_get_score(results[0])
            if "qwen" in update_entity and "glm" in update_entity and "ds" in update_entity: 
                # Log the time taken to score the batch of documents
                logger.info(f"batch {batch_count} scored use time: {math.ceil(time.time() - start_time)}s.")
                es_id = update_entity["id"]
                avg_score = round((int(update_entity["qwen"]) + int(update_entity["glm"]) + int(update_entity["ds"]))/3, 2)
                update_data = {
                    "doc": {
                        "process_status": 1,
                        "qwen_score": update_entity["qwen"],
                        "glm_score": update_entity["glm"],
                        "ds_score": update_entity["ds"],
                        "avg_score": avg_score
                    }
                }
                logger.info(f"{es_id} update success...use time: {math.ceil(time.time() - start_time)}s.")
            else:
                update_data = {
                    "doc": {
                        "process_status": 2
                    }
                }
                logger.info(f"{es_id} update fail...use time: {math.ceil(time.time() - start_time)}s.")   
            # Update the document in ES with the scores
            self.elastic.update(name=CU.TMP_ES_INDEX, data=update_data, id=es_id)
            batch_count += 1 
            # Search for the next batch of documents that need to be scored
            results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX, body=search_not_ready)
            
    def _thread_to_silicon_get_score(self, result):
        """
        A function that gets the scores of a document from silicon and checks if they are digits.

        This function takes a document from ES as input, and gets the scores of the document
        from silicon. It will remove any non-digit characters from the scores, and check if the
        scores are digits. If they are digits, it will store the scores in a dictionary.
        The function will return the dictionary containing the scores.

        Parameters:
            result (dict): A document from ES.

        Returns:
            dict: A dictionary containing the scores of the document.
        """
        id = result["_id"]
        gather_text = result["_source"]["gather_text"]
        
        start_time = time.time()
        
        resp_array,resp_threads = [],[]
        for key, params in self.content_scores.items():
            # Create a thread to get the score from silicon
            _thread = threading.Thread(
                target=self._get_and_check_digit_return,
                args=(gather_text, key, params, resp_array),
                daemon=True
                )
            resp_threads.append(_thread)
            _thread.start()
        # Wait for all the threads to finish
        for _thread in resp_threads:
            _thread.join()
        
        logger.info(f"{id}(scored),use time:{math.ceil(time.time() - start_time)}s.")    
        time.sleep(random.randint(2,5))    

        # Create a dictionary to store the scores
        resp_entity = {"id": id}
        if resp_array:
            # Iterate over the scores and update the dictionary
            for resp in resp_array:
                resp_entity.update(resp)
        return resp_entity
    
    def _get_and_check_digit_return(self, gather_text, key, params, resp_array):
        """
        A function that gets the score of a document from silicon and checks if it is a digit.

        This function takes a document text, a key of the LLM, the parameters for the LLM,
        and an array to store the response as input. It will get the score from the LLM,
        remove any non-digit characters from the score, and check if the score is a digit.
        If it is a digit, it will add the score to the response array and return. If not,
        it will sleep for a random time and try again.

        Args:
            gather_text (str): The text of the document.
            key (str): The key of the LLM.
            params (dict): The parameters for the LLM.
            resp_array (list): The array to store the response.
        """
        # Keep trying until we get a valid score
        digit_batch_count = 0
        while digit_batch_count < 3:
            try:
                start_time = time.time()
                score = self.api.chat_with_sync(params, CU.get_score_prompts(gather_text))
                # Remove any non-digit characters from the score
                score = re.sub(r'[^\w\u4e00-\u9fff]', '', score)
                # Check if the score is a digit
                if score.isdigit():
                    logger.info(f"LLM({key}) score:{score},use time:{math.ceil(time.time() - start_time)}s.")
                    resp_array.append({key: int(score)})
                    time.sleep(1)
                    break
                else:
                    logger.info(f"Not digit detected,Next round to fix it.")
                    time.sleep(random.randint(5,10))
                    digit_batch_count += 1
            except Exception:
                logger.info(f"Exception detected,Next round to fix it.")
                time.sleep(random.randint(5,10))
                digit_batch_count += 1
        
if __name__ == "__main__":
    s = ScoreAndFilterData()
    s.search_for_score()