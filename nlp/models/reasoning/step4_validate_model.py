"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step4_validate_model.py
Author: yuanzhenhui
Date: 2025-12-02 17:13:34
LastEditTime: 2025-12-05 10:43:11
"""
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))
import const_util as CU

from common_util import CommonUtil 
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from persistence.sqlite_util import SqliteUtil
from step3_openvino_runtime import OpenvinoRuntime

class ValidateModel:
    
    
    def __init__(self):
        self.sql = SqliteUtil()
        nlp_cnf = os.path.join(project_dir, 'resources','config', CU.ACTIVATE, 'nlp_cnf.yml')
        YamlNLP = YamlUtil(nlp_cnf)
        self.dataset_path = YamlNLP.get_value('models.reasoning.openvino.validate.dataset_path')
        self.score_path = YamlNLP.get_value('models.reasoning.openvino.score.dataset_path')
        
    def insert_all_validate_data_to_sqlite(self):
        """
        Insert all validate data to sqlite database.

        This function will loop through all the files in the validate dataset directory.
        For each file, it will read the content and insert the question and answer
        into the sqlite database.

        Args:
            None

        Returns:
            None
        """
        for i in os.listdir(self.dataset_path):
            if 'txt' not in i:
                continue
            new_path = os.path.join(self.dataset_path, i)
            logger.info(f"Processing file: {new_path}")
            with open(new_path, "r", encoding="utf-8") as f:
                try:
                    loop_index = 0
                    for idx,line in enumerate(f):
                        loop_index = idx
                        question = line.split(". ")[1]
                        data = {"question":question.strip()}
                        self.sql.insert("check_dataset_qa",data)
                except:
                    logger.exception(f"Error inserting data at index {loop_index}")
            logger.info(f"{i} inserted completed...")
            
    def clean_dulplicate_data(self):
        """
        Delete duplicate records from the check_dataset_qa table.

        This function will first query the check_dataset_qa table to find all the duplicate records.
        It will then loop through the results and delete all the duplicate records except for the one with the maximum id.

        Args:
            None

        Returns:
            None
        """
        # Query the database to find all the duplicate records
        sql = "select question,max(id) id from check_dataset_qa group by question having count(1) > 1"
        results = self.sql.fetch_all(sql)
        # Loop through the results and delete all the duplicate records
        while len(results) > 0:
            # Get the ids of all the duplicate records
            ids = [result[1] for result in results]
            # Convert the ids to a string
            ids_str = ','.join(map(str, ids))
            # Construct the delete sql query
            del_sql = f"delete from check_dataset_qa where id in ({ids_str})"
            # Execute the delete sql query
            self.sql.execute(del_sql)
            # Query the database again to get the next batch of results
            results = self.sql.fetch_all(sql)
        logger.info(f"delete dulplicate data completed...")
        
    def generate_answers(self):
        """
        Generate answers for the questions in the check_dataset_qa table.

        This function will first query the check_dataset_qa table to get the records that do not have an answer.
        It will then loop through the results and use the OpenVINO runtime to generate answers for each question.
        Finally, it will update the check_dataset_qa table with the generated answers.

        Args:
            None

        Returns:
            None
        """
        ort = OpenvinoRuntime()
        sql = "SELECT id,question FROM check_dataset_qa WHERE answer IS NULL LIMIT 10"
        results = self.sql.fetch_all(sql)
        batch_count = 0
        while len(results) > 0:
            for id,question in results:
                answer = ""
                # Use the OpenVINO runtime to generate answers for the question
                for chunk in ort.transfor_stream_msg(question):
                    if not chunk["finished"]:
                        # Append the generated text to the answer
                        answer += chunk["content"]
                    else:
                        # Check if the generated text contains the final response
                        if 'full_response' in chunk:
                            final_text = chunk['full_response']
                            # If the answer is empty, append the final response to the answer
                            if not answer:
                                answer += final_text
                # Update the check_dataset_qa table with the generated answer
                update_sql = f"UPDATE check_dataset_qa SET answer = '{answer}' WHERE id = {id}"
                self.sql.execute(update_sql)
                logger.info(f"Update id: {id} completed...")
            # Query the check_dataset_qa table again to get the next batch of records
            results = self.sql.fetch_all(sql)
            batch_count += 1
            logger.info(f"Generate batch {batch_count} answers completed...")
    
    def load_json_and_update_sqlite(self):
        """
        Load the JSON files and update the scores in the check_dataset_qa table.

        This function will loop through the JSON files in the score_path directory and update the scores in the check_dataset_qa table.
        
        Args:
            None

        Returns:
            None
        """
        for i in os.listdir(self.score_path):
            new_path = self.score_path + i
            type = i.split(".")[0]  
            json_data = CommonUtil.load_json_file(new_path)
            for json_obj in json_data:
                id = json_obj['id']
                score = json_obj['score']
                # Update the scores in the check_dataset_qa table
                update_sql = f"update check_dataset_qa set {type}_score = {score} where id = {id}"
                self.sql.execute(update_sql)
                logger.info(f"Update id: {id} completed...")
                
            logger.info(f"{type} inserted completed...")
        
if __name__ == '__main__':
    vm = ValidateModel()
    #vm.insert_all_validate_data_to_sqlite()
    #vm.clean_dulplicate_data()
    #vm.generate_answers()
    vm.load_json_and_update_sqlite()