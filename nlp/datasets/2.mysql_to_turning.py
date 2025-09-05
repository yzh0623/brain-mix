"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/nlp/datasets/2.mysql_to_turning.py
Author: Zhenhui Yuan
Date: 2025-09-05 10:24:16
LastEditTime: 2025-09-05 17:02:41
"""

import random
import sys
import os
import threading
import schedule
import time
import json

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

from persistence.mysql_util import MysqlUtil
from api_util import ApiUtil

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'utils_cnf.yml')
model_params = YamlUtil(utils_cnf).get_value('silicon.agent.generate_qaa')

api = ApiUtil()
myclient_local = MysqlUtil()

class MysqlToTurning:
    
    def clean_dulpicate_data(self):
        """
        Delete duplicate records from my_fine_turning_datas table.

        This method will delete duplicate records from the my_fine_turning_datas table.
        The search_sql query will group the records by content and find the minimum id.
        The records with the minimum id will be kept and the rest will be deleted.
        """
        search_sql = f"""
            SELECT content, MIN(id) AS id
            FROM {CU.TMP_MYSQL_TURNING_DATA_TABLE}
            GROUP BY content
            HAVING COUNT(1) > 1
            LIMIT 1000
        """
        results = myclient_local.query_by_list(search_sql)
        while len(results) > 0:
            ids_result = [row[1] for row in results]
            delete_sql = f"DELETE FROM {CU.TMP_MYSQL_TURNING_DATA_TABLE} WHERE id IN ({', '.join(map(str, ids_result))})"
            counter, _ = myclient_local.save_or_update(delete_sql)
            if counter > 0:
                logger.info(f"Deleted {counter} duplicate records from {CU.TMP_MYSQL_TURNING_DATA_TABLE}.")
                
                # Query the database again to get the next batch of records
                results = myclient_local.query_by_list(search_sql)
    
    def setup_data(self,params,source_table,batch_size,call_func):
        """
        This method will generate data based on the given table and fields.

        This method will query the database to get the records from the given table.
        The records will be passed to the given call_func to generate the data.
        The generated data will be inserted into the my_fine_turning_datas table.
        The generate_flag of the original records will be updated to 1.

        Parameters:
            params (list): A list of field names.
            source_table (str): The name of the table to query.
            batch_size (int): The number of records to query in each batch.
            call_func (function): The function to call to generate the data.
        """
        # Construct the SQL query to query the records from the given table
        search_sql = f"SELECT `id`,`{'`,`'.join(params)}` FROM {source_table} WHERE `generate_flag` = 0 LIMIT {batch_size}"
        
        # Construct the SQL query to insert the generated data into the my_fine_turning_datas table
        insert_turning_sql = "insert into my_fine_turning_datas(`content`,`sources`) values(%s,%s)"
        
        # Query the database to get the records from the given table
        result_resp = myclient_local.query_by_list(search_sql)
        while len(result_resp) > 0:
            
            # Call the given call_func to generate the data
            update_id_batch,insert_batch = call_func(result_resp,source_table)
            
            # Insert the generated data into the my_fine_turning_datas table
            if insert_batch:    
                insert_counter = myclient_local.batch_save_or_update(insert_turning_sql, insert_batch)
                
                # Update the generate_flag of the original records to 1
                if insert_counter > 0:
                    update_id_batch_str = "','".join(update_id_batch)
                    update_sql = f"update {source_table} set `generate_flag` = 1 where `id` in ('{update_id_batch_str}')"
                    update_counter,_ = myclient_local.save_or_update(update_sql)
                    logger.info(f"共生成{insert_counter}条数据,并更新了“{source_table}”表{update_counter}条数据的状态")   
            
            # Query the database again to get the next batch of records
            result_resp = myclient_local.query_by_list(search_sql) 
            
            # Sleep for a random time between 1 and 60 seconds
            time.sleep(random.randint(1,60))
     
def _thread_genrate_turning(prompt_str, insert_batch, source_table):
    """
    This function generates the turning data using the given prompt_str and
    inserts the generated data into the my_fine_turning_datas table.

    Parameters:
        prompt_str (str): The prompt string to generate the data.
        insert_batch (list): A list to store the generated data.
        source_table (str): The name of the source table.
    """
    while True:
        # Generate the data using the given prompt_str
        resp = api.chat_with_sync(model_params, prompt_str)
        try:
            
            # Parse the response as a JSON object
            resp_array = json.loads(resp)
            if resp_array:
                
                # Insert the generated data into the my_fine_turning_datas table
                # Each record will contain the generated data and the source table name
                insert_batch.extend((json.dumps(resp, ensure_ascii=False), source_table) for resp in resp_array)
                break
            else:
                
                # Sleep for a random time between 1 and 5 seconds if the response is empty
                time.sleep(random.randint(1, 5))
        except Exception:
            
            # Sleep for a random time between 1 and 5 seconds if there is an exception
            time.sleep(random.randint(1, 5))

def setup_ancient_books_data(results, source_table):
    """
    This function will generate data based on the given table and fields.

    This function will iterate over the given results and generate data for each record.
    The generated data will be inserted into the my_fine_turning_datas table.
    The generate_flag of the original records will be updated to 1.

    Parameters:
        results (list): A list of records from the given table.
        source_table (str): The name of the table to query.

    Returns:
        tuple: A tuple of two lists. The first list contains the ids of the records that have been
        generated, and the second list contains the generated data.
    """
    insert_batch = []
    update_id_batch = []
    search_threads = []
    
    # Iterate over the given results and generate data for each record
    for id, book_name, book_author, book_dynasty, book_release, book_content in results:
        update_id_batch.append(str(id))
        
        # Construct the prompt string
        montage = f"{{\"书名\":\"{book_name}\",\"朝代\":\"{book_dynasty}\",\"作者\":\"{book_author}\",\"出版时间\":\"{book_release}\",\"书中内容\":\"{book_content}\"}}"
        prompt_str = CU.get_question_and_answer_prompts(montage)
        
        # Create a thread to generate the data
        search_thread = threading.Thread(
            target=_thread_genrate_turning,
            args=(prompt_str, insert_batch, source_table),
            daemon=True
            )
        search_threads.append(search_thread)
        search_thread.start()
    
    # Wait for all the threads to finish
    for search_thread in search_threads:
        search_thread.join()
    
    # Return the generated data and the ids of the records that have been generated
    return update_id_batch, insert_batch

def setup_herbal_medicines_data(results, source_table):
    """
    This function will generate data based on the given table and fields.

    This function will iterate over the given results and generate data for each record.
    The generated data will be inserted into the my_fine_turning_datas table.
    The generate_flag of the original records will be updated to 1.

    Parameters:
        results (list): A list of records from the given table.
        source_table (str): The name of the table to query.

    Returns:
        tuple: A tuple of two lists. The first list contains the ids of the records that have been
        generated, and the second list contains the generated data.
    """
    insert_batch = []
    update_id_batch = []
    search_threads = []
    
    # Iterate over the given results and generate data for each record
    for id, variety_name, variety_data_type, variety_source, variety_content in results:
        update_id_batch.append(str(id))
        
        # Construct the prompt string
        montage = f"{{\"品种名称\":\"{variety_name}\",\"{variety_data_type}\":\"{variety_content}\",\"出处\":\"{variety_source}\"}}"
        prompt_str = CU.get_question_and_answer_prompts(montage)
        
        # Create a thread to generate the data
        search_thread = threading.Thread(
            target=_thread_genrate_turning, 
            args=(prompt_str, insert_batch, source_table),
            daemon=True
            )
        search_threads.append(search_thread)
        search_thread.start()
    
    # Wait for all the threads to finish
    for search_thread in search_threads:
        search_thread.join()
    
    # Return the generated data and the ids of the records that have been generated
    return update_id_batch, insert_batch

def setup_medicine_formulas_data(results, source_table):
    """
    This function will iterate over the given results and generate data for each record.
    The generated data will be inserted into the my_fine_turning_datas table.
    The generate_flag of the original records will be updated to 1.

    Parameters:
        results (list): A list of records from the given table.
        source_table (str): The name of the table to query.

    Returns:
        tuple: A tuple of two lists. The first list contains the ids of the records that have been
        generated, and the second list contains the generated data.
    """
    insert_batch = []
    update_id_batch = []
    search_threads = []
    
    # Iterate over the given results and generate data for each record
    for id, formula_name, formula_data_type, formula_source, formulas_content in results:
        update_id_batch.append(str(id))
        
        # Construct the prompt string
        montage = f"{{\"方剂名称\":\"{formula_name}\",\"{formula_data_type}\":\"{formulas_content}\",\"出处\":\"{formula_source}\"}}"
        prompt_str = CU.get_question_and_answer_prompts(montage)
        
        # Create a thread to generate the data
        search_thread = threading.Thread(
            target=_thread_genrate_turning, 
            args=(prompt_str, insert_batch, source_table),
            daemon=True
            )
        search_threads.append(search_thread)
        search_thread.start()
    
    # Wait for all the threads to finish
    for search_thread in search_threads:
        search_thread.join()
    
    # Return the generated data and the ids of the records that have been generated
    return update_id_batch, insert_batch

def setup_pharmacopeia_data(results, source_table):
    """
    This function will iterate over the given results and generate data for each record.
    The generated data will be inserted into the my_fine_turning_datas table.
    The generate_flag of the original records will be updated to 1.

    Parameters:
        results (list): A list of records from the given table.
        source_table (str): The name of the table to query.

    Returns:
        tuple: A tuple of two lists. The first list contains the ids of the records that have been
        generated, and the second list contains the generated data.
    """
    insert_batch = []
    update_id_batch = []
    search_threads = []
    
    # Iterate over the given results and generate data for each record
    for id,pharmacopeiacol_name,pharmacopeiacol_data_type,pharmacopeia_source,pharmacopeiacol_content in results:
        update_id_batch.append(str(id))
        
        # Construct the prompt string
        montage = f"{{\"药典药品名称\":\"{pharmacopeiacol_name}\",\"{pharmacopeiacol_data_type}\":\"{pharmacopeiacol_content}\",\"出处\":\"{pharmacopeia_source}\"}}"
        prompt_str = CU.get_question_and_answer_prompts(montage)
        
        # Create a thread to generate the data
        search_thread = threading.Thread(
            target=_thread_genrate_turning, 
            args=(prompt_str, insert_batch, source_table),
            daemon=True
            )
        search_threads.append(search_thread)
        search_thread.start()
    
    # Wait for all the threads to finish
    for search_thread in search_threads:
        search_thread.join()
    
    # Return the generated data and the ids of the records that have been generated
    return update_id_batch, insert_batch

mtt = MysqlToTurning()
schedule.every(2).hours.do(mtt.clean_dulpicate_data)
    
if __name__ == "__main__":
    threads = []
    params_array = [
        {
            "table_name":"my_ancient_books",
            "table_fields": ["book_name","book_author","book_dynasty","book_release","book_content"],
            "function_call": setup_ancient_books_data
        },
        {
            "table_name":"my_herbal_medicines",
            "table_fields": ["variety_name","variety_data_type","variety_source","variety_content"],
            "function_call": setup_herbal_medicines_data
        },
        {
            "table_name":"my_medicine_formulas",
            "table_fields": ["formula_name","formula_data_type","formula_source","formulas_content"],
            "function_call": setup_medicine_formulas_data
        },
        {
            "table_name":"my_pharmacopeia",
            "table_fields": ["pharmacopeiacol_name","pharmacopeiacol_data_type","pharmacopeia_source","pharmacopeiacol_content"],
            "function_call": setup_pharmacopeia_data
        }
    ]
    
    # Create a data processing thread
    for params in params_array:
        thread =threading.Thread(
                target=mtt.setup_data, 
                args=(params["table_fields"],params["table_name"],5,params["function_call"]),
                daemon=True
                )
        threads.append(thread)
        thread.start()
        
    # Put the timer task into a separate thread
    def run_scheduler():
        """Run the scheduler in a separate thread.

        This function is used to run the scheduler in a separate thread.
        It will block until the scheduler thread is finished.
        """
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Waiting for the data processing thread to complete
    for thread in threads:
        thread.join()
    
    # The main thread remains running to prevent the program from exiting
    scheduler_thread.join()
