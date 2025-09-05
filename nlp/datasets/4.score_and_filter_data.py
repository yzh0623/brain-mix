"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/nlp/datasets/4.score_and_filter_data.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-05 19:47:41
"""

import re
import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from queue import Queue
from threading import Lock, Semaphore

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from api_util import ApiUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.elastic_util import ElasticUtil

class ScoreAndFilterData:
    
    def __init__(self):
        """
        Initialize the ScoreAndFilterData class.
        """
        # Get the Elasticsearch client
        self.elastic = ElasticUtil()
        
        # Get the ApiUtil instance
        self.api = ApiUtil()
        
        # Create a thread pool of size SCORE_THREADS_SIZE
        self.thread_pool = ThreadPoolExecutor(max_workers=CU.SCORE_THREADS_SIZE)
        self.lock = Lock()
        self.semaphore = Semaphore(CU.SCORE_THREADS_SIZE)
        
        # Read the Silicon API configuration
        silicon_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.content_scores = YamlUtil(silicon_cnf).get_value('silicon.agent.content_score')
        
    def search_for_score(self):
        """
        Search for documents in ES that need to be scored and score them using Silicon API.
        """
        # Search for documents that need to be scored
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
        results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX, body=search_not_ready)
        task_queue = Queue()

        # Add the documents to a queue
        for result in results:
            task_queue.put(result)
        
        batch_count = 1
        while not task_queue.empty():
            start_time = time.time()
            update_array = []

            futures = []
            # Get a batch of documents from the queue
            while not task_queue.empty():
                result = task_queue.get()
                # Submit the document to a thread to get the scores from Silicon API
                futures.append(self.thread_pool.submit(self._thread_to_silicon_get_score, result, update_array))

            # Wait for all threads to finish and get the results
            for future in as_completed(futures):
                future.result()
                
            if update_array:
                # Log the time taken to score the batch of documents
                logger.info(f"{batch_count} batch search use time: {math.ceil(time.time() - start_time)}s.")
                batch_count += 1
                for update_entity in update_array:
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
                    # Update the document in ES with the scores
                    self.elastic.update(name=CU.TMP_ES_INDEX, data=update_data, id=es_id)
                    logger.info(f"{es_id} update success...")
                    
            # Sleep for a random time to avoid overloading the Silicon API
            time.sleep(random.randint(5, 10))
            
            # Search for the next batch of documents that need to be scored
            results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX, body=search_not_ready)
            for result in results:
                task_queue.put(result)
            
    def _thread_to_silicon_get_score(self, result, update_array):
        """
        A thread that gets the score of a document from silicon and updates the document.

        This function is called by the `search_for_score` function to get the scores of a batch
        of documents from Silicon API. It takes a result from ES and an array to store the
        updated documents as input. It will get the scores from Silicon API and update the
        document in ES with the scores.

        Args:
            result (dict): The result from ES.
            update_array (list): The array to store the updated documents.
        """
        id = result["_id"]
        gather_text = result["_source"]["gather_text"]
        
        start_time = time.time()
        resp_array = []
        
        # Get the scores from Silicon API
        for key, params in self.content_scores.items():
            self._get_and_check_digit_return(gather_text, key, params, resp_array)
            time.sleep(2)

        if resp_array:
            logger.info(f"Finished getting scores for {id} use time: {math.ceil(time.time() - start_time)}s.")
            resp_entity = {"id": id}
            for resp in resp_array:
                resp_entity.update(resp)
            with self.lock:
                update_array.append(resp_entity)
    
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
        start_time = time.time()
        timeout = 300
        
        # Keep trying until we get a valid score
        digit_batch_count = 1
        while True:
            try:
                
                if time.time() - start_time > timeout:
                    logger.error(f"Timeout while getting score from LLM {key}.")
                    return
                
                with self.semaphore:
                    score = self.api.chat_with_sync(params, CU.get_score_prompts(gather_text))
                
                logger.info(f"Got score in {digit_batch_count} batch from LLM {key}: {score} use time: {math.ceil(time.time() - start_time)}s.")
                
                # Remove any non-digit characters from the score
                score = re.sub(r'[^\w\u4e00-\u9fff]', '', score)
                
                # Check if the score is a digit
                if score.isdigit():
                    with self.lock:
                        resp_array.append({key: int(score)})
                    return
                else:
                    logger.info(f"Not digit {digit_batch_count} batch use time: {math.ceil(time.time() - start_time)}s.")
                    time.sleep(random.randint(15,30))
            except Exception:
                logger.info(f"Exception {digit_batch_count} batch use time: {math.ceil(time.time() - start_time)}s.")
                time.sleep(random.randint(15,30))
                
            digit_batch_count += 1
            time.sleep(1) 
        
if __name__ == "__main__":
    s = ScoreAndFilterData()
    s.search_for_score()