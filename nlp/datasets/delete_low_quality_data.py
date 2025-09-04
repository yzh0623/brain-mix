
import time
from tqdm import tqdm
from elasticsearch import Elasticsearch
import schedule
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from clean_util import CleanUtil
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.elastic_util import ElasticUtil

@DeprecationWarning
class DeleteLowQualityData:
    def __init__(self):
        self.elastic = ElasticUtil()
        self.dh = CleanUtil()
        
        elastic_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        self.delete_conn = Elasticsearch(
            hosts=YamlUtil(elastic_cnf).get_value('persistence.elastic.host'),
            basic_auth=(
                YamlUtil(elastic_cnf).get_value('persistence.elastic.username'),
                YamlUtil(elastic_cnf).get_value('persistence.elastic.password')
            ),
            max_retries=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.max_retries')),
            connections_per_node=min(50, os.cpu_count() * 4),
            request_timeout=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.timeout')),
        )

    def delete_dulpicate_data(self):
        """
        Delete duplicate data in the Elasticsearch index.
        """
        search_sql = f"select question,answer from {CU.TMP_ES_INDEX} group by question,answer having count(1) > 1"
        results = self.elastic.find_by_sql(sql=search_sql)
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
                # Find all the duplicate data
                dsl_results = self.elastic.find_by_body_nopaging(name=CU.TMP_ES_INDEX, body=search_single_body)
                # Delete all the duplicate data except for the first one
                for idx, dsl_result in enumerate(dsl_results):
                    if idx > 0:
                        self.elastic.delete_by_id(name=CU.TMP_ES_INDEX, id=dsl_result["_id"])
    
    def delete_similar_data(self):
        """
        Delete similar data in the Elasticsearch index.
        
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
schedule.every(30).minutes.do(dlqd.delete_dulpicate_data)
schedule.every(90).minutes.do(dlqd.delete_similar_data)

if __name__ == "__main__":
    
    dlqd.delete_dulpicate_data()
    dlqd.delete_similar_data()
    
    while True:
        schedule.run_pending()
        time.sleep(1)
