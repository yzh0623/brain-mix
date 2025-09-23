"""
FilePath: /brain-mix/nlp/datasets/step5_turning_data_preparer.py
Author: yuanzhenhui
Date: 2025-09-22 10:06:01
LastEditTime: 2025-09-23 14:25:13
"""

import os
import sys
import json
from typing import List, Dict
from sklearn.model_selection import train_test_split

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from elasticsearch import Elasticsearch
from persistence.elastic_util import ElasticUtil

ES_BATCH_COUNT = 10000

class TurningDataPreparer:

    def __init__(self):
        """
        Initialize the TurningDataPreparer class.
        
        Load the configuration from the YAML files and initialize the Elasticsearch connection.
        """
        self.model_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        self.elastic_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        
        cnf = YamlUtil(self.model_cnf)
        self.dataset_size = cnf.get_value('models.reasoning.finetuning.dataset_size')
        self.validation_split = cnf.get_value('models.reasoning.finetuning.validation_split')
        self.data_path = cnf.get_value('datasets.base_path')
        self.turning_dir = cnf.get_value('datasets.turning_path')
        
        ecnf = YamlUtil(self.elastic_cnf)
        self._conn = Elasticsearch(
            hosts=ecnf.get_value('persistence.elastic.host'),
            basic_auth=(
                ecnf.get_value('persistence.elastic.username'),
                ecnf.get_value('persistence.elastic.password')
            ),
            max_retries=int(ecnf.get_value('persistence.elastic.max_retries')),
            connections_per_node=min(50, os.cpu_count() * 4),
            request_timeout=int(ecnf.get_value('persistence.elastic.timeout')),
            retry_on_timeout=True
        )
        
        self.elastic = ElasticUtil()

    def clean_and_filter_data(self, data: List[Dict]) -> List[Dict]:
        """
        Clean and filter the given data.

        The data will be filtered based on the following rules:
        1. The length of the question and answer should be at least 5 characters.
        2. The length of the question and answer should be no more than 10000 characters.
        3. The question and answer should not be the same.

        Args:
            data (List[Dict]): The data to be cleaned and filtered.

        Returns:
            List[Dict]: The cleaned and filtered data.
        """
        cleaned_data = []

        for item in data:
            question = item["messages"][0]["content"]
            answer = item["messages"][1]["content"]

            # Check if the question and answer meet the rules
            if len(question) < 5 or len(answer) < 5:
                # If the question or answer is too short, skip it
                continue
            if len(question) > 10000 or len(answer) > 10000:
                # If the question or answer is too long, skip it
                continue
            if question == answer:
                # If the question and answer are the same, skip it
                continue

            # If the question and answer meet the rules, add it to the cleaned data
            cleaned_data.append(item)

        logger.info(f"After data cleaning, there are {len(cleaned_data)} entries remaining")
        return cleaned_data

    def extract_data_from_es(self) -> List[Dict]:
        """
        Extract data from Elasticsearch.

        This function will extract data from Elasticsearch and return a list of dictionaries, where each dictionary
        represents a record in the index.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a record in the index.
        """
        logger.info("Begin to extract data from Elasticsearch...")
        all_data = []
        
        # Get the average score of all records
        search_avg_sql = f"select avg(avg_score) from {CU.TMP_ES_INDEX} where process_status = 1"
        avg_score_results = self.elastic.find_by_sql(search_avg_sql)
        avg_score = avg_score_results.body["rows"][0][0]
        
        # Get all data sources
        self.data_sources = []
        data_source_group_sql = f"select data_source from {CU.TMP_ES_INDEX} group by data_source"
        data_source_results = self.elastic.find_by_sql(data_source_group_sql)
        data_sources_rows = data_source_results.body["rows"]
        self.data_sources.extend(data_sources_row[0] for data_sources_row in data_sources_rows)
        
        for source in self.data_sources:
            logger.info(f"Loading data source: {source}")

            search_body = {
                "size": ES_BATCH_COUNT,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"data_source": {"value": source}}},
                            {"term": {"process_status": {"value": 1}}},
                            {"range": {"avg_score": {"gt": avg_score,"boost": 1}}}
                        ],
                        "boost": 1
                    }
                }
            }

            # Perform the initial search
            response = self._conn.search(index=CU.TMP_ES_INDEX,body=search_body,scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            while hits:
                for hit in hits:
                    row_entity = hit["_source"]
                    all_data.append({
                        "messages": [
                            {"role": "user","content": row_entity["question"]},
                            {"role": "assistant","content": row_entity["answer"]}
                        ],
                        "source": source
                    })

                # Get the next batch of data
                response = self._conn.scroll(scroll_id=scroll_id,scroll='2m')
                hits = response['hits']['hits']
                if len(all_data) >= self.dataset_size * 1.2:
                    break
            if len(all_data) >= self.dataset_size * 1.2:
                break
        logger.info(f"Total {len(all_data)} records extracted from Elasticsearch")
        return all_data

    def prepare_and_save_dataset(self):
        """
        Prepare the dataset and save it to a JSON file.

        This function will extract the data from Elasticsearch, clean and filter the data, and
        then save it to a JSON file. If the validation split is larger than 0, it will split
        the data into a training set and a validation set.

        Args:
            None

        Returns:
            None
        """
        output_dir = os.path.join(self.data_path, self.turning_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Extract the data from Elasticsearch
        all_data = self.extract_data_from_es()
        logger.info(f"Total {len(all_data)} records extracted from Elasticsearch")

        # Clean and filter the data
        all_data = self.clean_and_filter_data(all_data)
        logger.info(f"Total {len(all_data)} records cleaned and filtered")

        # If the validation split is larger than 0, split the data into a training set and a validation set
        if self.validation_split > 0:
            train_data, val_data = train_test_split(
                all_data,
                test_size=self.validation_split,
                random_state=42
            )
            logger.info(f"Training set size: {len(train_data)}")
            logger.info(f"Validation set size: {len(val_data)}")

            # Save the training set to a JSON file
            train_path = os.path.join(output_dir, "train_dataset.json")
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Training set saved to: {train_path}")

            # Save the validation set to a JSON file
            val_path = os.path.join(output_dir, "validation_dataset.json")
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Validation set saved to: {val_path}")
        else:
            # Save the data to a JSON file
            train_path = os.path.join(output_dir, "train_dataset.json")
            with open(train_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Training set saved to: {train_path}")

        # Save the metadata to a JSON file
        metadata = {
            "total_samples": len(all_data),
            "train_samples": len(train_data) if self.validation_split > 0 else len(all_data),
            "validation_samples": len(val_data) if self.validation_split > 0 else 0,
            "validation_split": self.validation_split,
            "data_sources": self.data_sources
        }

        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    preparer = TurningDataPreparer()
    preparer.prepare_and_save_dataset()
