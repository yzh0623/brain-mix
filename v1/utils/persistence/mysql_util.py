"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/persistence/mysql_util.py
Author: yuanzhenhui
Date: 2024-09-12 16:45:07
LastEditTime: 2025-09-10 15:28:22
"""

import pymysql
import threading
mutex = threading.Lock()
import datetime
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class MysqlUtil:
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MysqlUtil instance.

        This constructor establishes a connection to the MySQL database using
        either the provided keyword arguments or configuration from a YAML file.

        :param kwargs: Optional keyword arguments for database connection:
            - host: Database host address
            - port: Database port number
            - username: Username for authentication
            - password: Password for authentication
            - schemas: Database schema to connect to
            - charset: Character set for the connection
        :return: None
        """
        if kwargs:
            # Connect to MySQL using provided keyword arguments
            self._conn = pymysql.connect(
                host=kwargs['host'],
                port=kwargs['port'],
                user=kwargs['username'],
                passwd=kwargs['password'],
                db=kwargs['schemas'],
                charset=kwargs['charset']
            )
        else:
            # Load MySQL configuration from YAML file
            mysql_cnf = YamlUtil(os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml'))
            
            # Connect to MySQL using configuration from YAML file
            self._conn = pymysql.connect(
                host=mysql_cnf.get_value('persistence.mysql.host'),
                port=mysql_cnf.get_value('persistence.mysql.port'),
                user=mysql_cnf.get_value('persistence.mysql.username'),
                passwd=mysql_cnf.get_value('persistence.mysql.password'),
                db=mysql_cnf.get_value('persistence.mysql.schemas'),
                charset=mysql_cnf.get_value('persistence.mysql.charset')
            )

    def default(self,o) -> str:
        """
        A function to convert non-serializable objects into serializable ones.

        This function is used to convert datetime objects into strings when
        serializing data to JSON.

        Parameters:
            o (object): The object to serialize.

        Returns:
            str: The serialized object.

        Raises:
            TypeError: If the object is not serializable.
        """
        if isinstance(o, datetime.datetime):
            return o.strftime("%Y-%m-%d %H:%M:%S")
        raise TypeError("Unserializable object {}".format(o))
        
    def batch_save_or_update(self, exec_sql, batch) -> int:
        """
        Execute a batch of SQL statements to save or update records in the database.

        This method is useful for batch inserting or updating a large number of records
        in a single database call.

        Parameters:
            exec_sql (str): The SQL statement to execute.
            batch (list of tuple): The data to be inserted or updated.

        Returns:
            int: The number of affected rows.
        """
        cursor = self._conn.cursor()
        exec_count = 0
        try:
            mutex.acquire()
            
            # Execute the batch of SQL statements
            exec_count = cursor.executemany(exec_sql, batch)
            
            # Commit the changes to the database
            self._conn.commit()
            mutex.release()
        except Exception as e:
            self._conn.rollback()
            logger.error(e)
        finally:
            cursor.close()
        return exec_count

    def save_or_update(self, exec_sql) -> tuple:
        """
        Execute a SQL statement to save or update a record in the database.

        Parameters:
            conn (pymysql.connections.Connection): The connection to the MySQL database.
            exec_sql (str): The SQL statement to execute.

        Returns:
            tuple: A tuple of two elements. The first element is the number of affected rows.
                   The second element is the last insert id.
        """
        cursor = self._conn.cursor()
        exec_count = 0
        last_insert_id = 0
        try:
            mutex.acquire()
            
            # Execute the SQL statement
            exec_count = cursor.execute(exec_sql)
            
            # Get the last insert id
            last_insert_id = cursor.lastrowid
            
            # Commit the changes to the database
            self._conn.commit()
            mutex.release()
        except Exception as e:
            self._conn.rollback()
            logger.error(e)
        finally:
            cursor.close()
            
        # Return the number of affected rows and the last insert id
        return exec_count, last_insert_id

    def query_by_list(self, exec_sql) -> tuple:
        """
        Execute the given SQL query and return all the results.

        Parameters:
            conn (pymysql.connections.Connection): The connection to the MySQL database.
            exec_sql (str): The SQL query to execute.

        Returns:
            tuple: The results of the query.
        """
        result_set = None
        cursor = self._conn.cursor()
        try:
            mutex.acquire()
            
            # Execute the SQL query
            cursor.execute(exec_sql)
            
            # Fetch all the results
            result_set = cursor.fetchall()
            mutex.release()
        except Exception as e:
            logger.error(e)
        finally:
            cursor.close()
        return result_set

    def query_by_one(self, exec_sql) -> tuple:
        """
        Execute the given SQL query and return the first result.

        Parameters:
            conn (pymysql.connections.Connection): The connection to the MySQL database.
            exec_sql (str): The SQL query to execute.

        Returns:
            tuple: The first result of the query.
        """
        result_set = None
        cursor = self._conn.cursor()
        try:
            mutex.acquire()
            
            # Execute the SQL query
            cursor.execute(exec_sql)
            
            # Fetch the first result
            result_set = cursor.fetchone()
            mutex.release()
        except Exception as e:
            logger.error(e)
        finally:
            cursor.close()
        return result_set

    def query_by_page(self, page, page_size, exec_sql) -> list:
        """
        Execute the given SQL query by page and return the results.

        Parameters:
            page (int): The page number to query.
            page_size (int): The number of records per page.
            exec_sql (str): The SQL query to execute.

        Returns:
            list: The list of results.
        """
        result_set = None
        
        # Construct the SQL query to execute
        sql = f"{exec_sql} LIMIT {page_size} OFFSET {(page - 1) * page_size}"
        cursor = self._conn.cursor()
        try:
            # Ensure the database operations are thread-safe
            mutex.acquire()
            
            # Execute the SQL query
            cursor.execute(sql)
            
            # Fetch all the results
            result_set = cursor.fetchall()
            mutex.release()
        except Exception as e:
            logger.error(e)
        finally:
            cursor.close()
        return result_set

    def update_by_ids(self, table_name, setup_sql, ids) -> int:
        """
        Update records in the specified table by the given IDs.

        Parameters:
            conn (pymysql.connections.Connection): The connection to the MySQL database.
            table_name (str): The name of the table to update.
            setup_sql (str): The SQL statement to set the values.
            ids (list): List of IDs to update.

        Returns:
            int: The number of affected rows.
        """
        sql = f"UPDATE {table_name} SET {setup_sql} WHERE ID IN ({','.join([str(i) for i in ids])})"
        counter,_ = self.save_or_update(sql)
        return counter
    
    def delete_by_ids(self, table_name, ids) -> int:
        """
        Delete records in the specified table by the given IDs.

        Parameters:
            table_name (str): The name of the table to delete from.
            ids (list): List of IDs to delete.

        Returns:
            int: The number of affected rows.
        """
        sql = f"DELETE FROM {table_name} WHERE ID IN ({','.join([str(i) for i in ids])})"
        counter,_ = self.save_or_update(sql)
        return counter
    
    def delete_in_fields(self, table_name,field_name, field_array) -> int:
        """
        Delete records in the specified table where the given field_name is in the given field_array.

        Parameters:
            table_name (str): The name of the table to delete from.
            field_name (str): The name of the field to check.
            field_array (list): List of values to check against.

        Returns:
            int: The number of affected rows.
        """
        condicate = "','".join([str(i) for i in field_array])
        sql = f"DELETE FROM {table_name} WHERE {field_name} IN ('{condicate}')"
        counter,_ = self.save_or_update(sql)
        return counter

    def __del__(self):
        self._conn.close()