"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/nlp/datasets/3.delete_low_quality_data.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-10 16:10:23
"""

import time
from tqdm import tqdm
from elasticsearch import Elasticsearch
import schedule
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from persistence.clean_util import CleanUtil
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.elastic_util import ElasticUtil

class DeleteLowQualityData:
    
    def __init__(self):
        """
        Initialize the DeleteLowQualityData class.
        
        This class will delete low quality data from the Elasticsearch index.
        """
        # Initialize the Elasticsearch client
        self.elastic = ElasticUtil()
        
        # Initialize the CleanUtil class
        self.dh = CleanUtil()
        
        # Read the config file and get the config values
        elastic_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        
        # Initialize the Elasticsearch client for deleting data
        self.delete_conn = Elasticsearch(
            # Elasticsearch host
            hosts=YamlUtil(elastic_cnf).get_value('persistence.elastic.host'),
            
            # Elasticsearch username and password
            basic_auth=(
                YamlUtil(elastic_cnf).get_value('persistence.elastic.username'),
                YamlUtil(elastic_cnf).get_value('persistence.elastic.password')
            ),
            
            # Maximum number of retries to connect to Elasticsearch
            max_retries=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.max_retries')),
            
            # The number of connections to make to each node
            connections_per_node=min(50, os.cpu_count() * 4),
            
            # The amount of time to wait for a request to be completed
            request_timeout=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.timeout')),
        )

    def delete_dulpicate_data(self):
        """
        Delete duplicate data in the Elasticsearch index.
        
        :return: None
        """
        # Construct the SQL query to find the duplicate data
        search_sql = f"select question,answer from {CU.TMP_ES_INDEX} group by question,answer having count(1) > 1"
        
        # Find all the duplicate data
        results = self.elastic.find_by_sql(sql=search_sql)
        
        # Delete all the duplicate data except for the first one
        if results:
            response_array = results.body["rows"]
            for response in tqdm(response_array, desc="Now delete dulpicate data..."):
                
                # Construct the search query to find the duplicate data
                search_single_body = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"question": {"value": response[0]}}},
                                {"term": {"answer": {"value": response[1]}}}
                            ]
                        }
                    }
                }
                dsl_results = self.elastic.find_by_body_nopaging(name=CU.TMP_ES_INDEX, body=search_single_body)
                for idx, dsl_result in enumerate(dsl_results):
                    if idx > 0:
                        self.elastic.delete_by_id(name=CU.TMP_ES_INDEX, id=dsl_result["_id"])
    
    def delete_similar_data(self):
        """
        Delete similar data in the Elasticsearch index.

        This method uses the DBSCAN clustering algorithm to find and remove duplicate vectors from the Elasticsearch index.
        The similarity threshold is set to 0.95, which means that vectors with a cosine similarity greater than 0.95 will be considered as duplicates.

        :return: None
        """
        # Find and remove duplicate vectors from the Elasticsearch index
        self.dh.find_and_remove_duplicate_vectors(
            self.delete_conn,
            CU.TMP_ES_INDEX,
            vector_field=CU.TMP_ES_VECTOR_FIELDS,
            text_field="gather_text",
            similarity_threshold=0.95
        )

dlqd = DeleteLowQualityData()        
schedule.every(3).hours.do(dlqd.delete_similar_data)

if __name__ == "__main__":
    dlqd.delete_similar_data()
    while True:
        schedule.run_pending()
        time.sleep(1)
