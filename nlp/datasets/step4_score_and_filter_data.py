"""
FilePath: /brain-mix/nlp/datasets/step4_score_and_filter_data.py
Author: yuanzhenhui
Date: 2025-09-22 10:06:01
LastEditTime: 2025-10-11 11:01:36
"""

import os
import sys
import multiprocessing

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

from persistence.elastic_util import ElasticUtil

import const_util as CU
from yaml_util import YamlUtil

BATCH_SIZE=5

class ScoreAndFilterData:

    def __init__(self):
        """
        Initialize the ScoreAndFilterData class.

        This class is used to score and filter data using the Silicon API.

        :return: None
        """
        
        # Initialize the Elasticsearch connection
        self.elastic = ElasticUtil()

        # Get the Silicon API configuration from the YAML file
        silicon_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        # Get the content scores from the Silicon API configuration
        self.content_scores = YamlUtil(silicon_cnf).get_value('silicon.agent.content_score')
        # self.content_scores is a dictionary mapping the content score type to the weight of the score

    def search_for_score(self):
        """
        Search for records in the TMP_ES_INDEX with a process status of 0,
        and update their content scores using the Silicon API.

        This function will search for records in the TMP_ES_INDEX with a process status of 0,
        and update their content scores using the Silicon API. It will use multiprocessing to
        process multiple records in parallel.

        :return: None
        """
        search_not_ready = {
            "size": BATCH_SIZE,
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
        while len(results) > 0:
            with multiprocessing.Pool(processes=BATCH_SIZE) as pool:
                
                # Use multiprocessing to process multiple records in parallel
                process_results = pool.starmap(
                    process_one_record,
                    [(result, self.content_scores) for result in results]
                )
            for es_id, update_data in process_results:
                
                # Update the records in the TMP_ES_INDEX with the processed content scores
                self.elastic.update(name=CU.TMP_ES_INDEX,data=update_data, id=es_id)
            batch_count += 1
            
            # Search for the next batch of records
            results = self.elastic.find_by_body(name=CU.TMP_ES_INDEX, body=search_not_ready)

def process_one_record(result, content_scores):
    """
    Process one record from the TMP_ES_INDEX.

    This function will get the content scores from the Silicon API and update the records in the TMP_ES_INDEX.

    Args:
        result (dict): The result from the Elasticsearch.
        content_scores (dict): The content scores to update the records with.

    Returns:
        tuple: A tuple containing the Elasticsearch ID and the update data.
    """
    import time
    import math
    import random
    import re
    import threading
    
    from thirdparty.silicon_util import SiliconUtil
    
    from logging_util import LoggingUtil
    logger = LoggingUtil(os.path.basename(__file__).replace(".py", "")+"SubProcess")

    silicon = SiliconUtil()

    id = result["_id"]
    gather_text = result["_source"]["gather_text"]

    resp_array, resp_threads = [], []

    def get_and_check_digit_return(gather_text, key, params, resp_array):
        """
        Get the digit from the LLM model and check if it is a digit.

        This function will get the digit from the LLM model and check if it is a digit.
        If the digit is not a digit, it will sleep for a random time between 5 and 10 seconds
        and retry to get the digit.

        Args:
            gather_text (str): The text to get the digit from.
            key (str): The type of the LLM model.
            params (dict): The parameters to pass to the Silicon API.
            resp_array (list): A list to store the result.

        Returns:
            None
        """
        digit_batch_count = 0
        while digit_batch_count < 3:
            try:
                st = time.time()
                
                # Get the digit from the LLM model
                score = silicon.chat_with_sync(params, CU.get_score_prompts(gather_text))
                score = re.sub(r'[^\w\u4e00-\u9fff]', '', score)
                if score.isdigit():
                    
                    # If the digit is a digit, log the time it takes to get the digit
                    logger.info(f"Batch key_id:{id}, LLM({key}) score:{score}, use time:{math.ceil(time.time() - st)}s.")
                    
                    # Append the digit to the result list
                    resp_array.append({key: int(score)})
                    
                    # Sleep for 1 second
                    time.sleep(1)
                    break
                else:
                    
                    # If the digit is not a digit, log an error message
                    logger.error(f"Not digit detected,Next round to fix it.")
                    
                    # Sleep for a random time between 5 and 10 seconds
                    time.sleep(random.randint(5, 10))
                    digit_batch_count += 1
            except Exception:
                
                # If an exception is detected, log an error message
                logger.error(f"Exception detected,Next round to fix it.")
                
                # Sleep for a random time between 5 and 10 seconds
                time.sleep(random.randint(5, 10))
                digit_batch_count += 1

    """
    Get the digit from the LLM model and check if it is a digit.

    This function will get the digit from the LLM model and check if it is a digit.
    If the digit is not a digit, it will sleep for a random time between 5 and 10 seconds
    and retry to get the digit.

    Args:
        gather_text (str): The text to get the digit from.
        key (str): The type of the LLM model.
        params (dict): The parameters to pass to the Silicon API.
        resp_array (list): A list to store the result.

    Returns:
        None
    """
    for key, params in content_scores.items():
        _thread = threading.Thread(
            target=get_and_check_digit_return,
            args=(gather_text, key, params, resp_array),
            daemon=True
        )
        resp_threads.append(_thread)
        _thread.start()
    for _thread in resp_threads:
        _thread.join()

    """
    Process the result from the LLM model.

    This function will process the result from the LLM model and calculate the average score.
    If the average score is greater than or equal to 90, it will update the process status to 1.
    If the average score is less than 90, it will update the process status to 2.

    Args:
        None

    Returns:
        dict: A dictionary that contains the update data.
    """
    resp_entity = {"id": id}
    if resp_array:
        for resp in resp_array:
            resp_entity.update(resp)

    if "qwen" in resp_entity and "glm" in resp_entity and "ds" in resp_entity:
        avg_score = round((int(resp_entity["qwen"]) + int(resp_entity["glm"]) + int(resp_entity["ds"]))/3, 2)
        update_data = {
            "doc": {
                "process_status": 1,
                "qwen_score": resp_entity["qwen"],
                "glm_score": resp_entity["glm"],
                "ds_score": resp_entity["ds"],
                "avg_score": avg_score
            }
        }
    else:
        update_data = {"doc": {"process_status": 2}}
    return (id, update_data)


if __name__ == "__main__":
    s = ScoreAndFilterData()
    s.search_for_score()
