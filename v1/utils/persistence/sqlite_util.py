"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/utils/persistence/sqlite_util.py
Author: yuanzhenhui
Date: 2025-09-22 16:09:16
LastEditTime: 2025-12-02 17:41:49
"""

import sqlite3
from typing import List, Dict, Any, Optional

import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class SqliteUtil:

    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        """
        Establish a connection to the SQLite database.

        This function establishes a connection to the SQLite database specified by the db_path attribute.
        It sets the row factory to sqlite3.Row, which allows for fetching results as a named tuple.

        Raises:
            sqlite3.Error: If the connection to the database fails.
        """
        try:
            sqlite_cnf = YamlUtil(os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml'))
            db_path = os.path.join(project_dir, 'resources', 'check', sqlite_cnf.get_value('persistence.sqlite.db_name'))
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error(f"数据库连接错误: {e}")
            raise

    def close(self):
        """
        Close the connection to the SQLite database.

        This function closes the connection to the SQLite database specified by the db_path attribute.
        It sets the conn attribute to None after closing the connection.

        Returns:
            None
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """
        Enter the runtime context for the SQLite database connection.

        This function is called when entering a with statement block that uses the SqliteUtil instance.
        If the connection to the database is not established, it will be established by calling the _connect method.

        Returns:
            SqliteUtil: The instance itself
        """
        if not self.conn:
            self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute an SQL query against the SQLite database.

        This function executes an SQL query against the SQLite database specified by the db_path attribute.
        It takes an SQL query and optional parameters as arguments, executes the query, and commits the changes.
        If an error occurs during execution, it rolls back the changes and raises an sqlite3.Error exception.

        Args:
            sql (str): The SQL query to execute.
            params (tuple): Optional parameters to pass to the SQL query.

        Returns:
            sqlite3.Cursor: The cursor object of the executed query.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            self.conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"执行 SQL 语句时出错: {sql}\n参数: {params}\n错误: {e}")
            self.conn.rollback()
            raise

    def execute_script(self, sql_script: str) -> None:
        """
        Execute an SQL script against the SQLite database.

        This function executes an SQL script against the SQLite database specified by the db_path attribute.
        It takes an SQL script as an argument, executes the script, and commits the changes.
        If an error occurs during execution, it rolls back the changes and raises an sqlite3.Error exception.

        Args:
            sql_script (str): The SQL script to execute.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        try:
            cursor = self.conn.cursor()
            cursor.executescript(sql_script)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"执行 SQL 脚本时出错: {e}")
            self.conn.rollback()
            raise

    def fetch_all(self, sql: str, params: tuple = ()) -> List[List[Any]]:
        """
        Fetch all records from the SQLite database.

        This function executes an SQL query against the SQLite database specified by the db_path attribute,
        and fetches all records from the result set.
        It takes an SQL query and optional parameters as arguments, executes the query, and fetches all records.

        Args:
            sql (str): The SQL query to execute.
            params (tuple): Optional parameters to pass to the query.

        Returns:
            List[List[Any]]: A list of all records fetched from the database. Each record is a list of values.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [list(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"查询所有记录时出错: {sql}\n参数: {params}\n错误: {e}")
            raise

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[List[Any]]:
        """
        Fetch one record from the SQLite database.

        This function executes an SQL query against the SQLite database specified by the db_path attribute,
        and fetches one record from the result set.
        It takes an SQL query and optional parameters as arguments, executes the query, and fetches one record.

        Args:
            sql (str): The SQL query to execute.
            params (tuple): Optional parameters to pass to the SQL query.

        Returns:
            Optional[List[Any]]: The fetched record as a list of values, or None if no records are found.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return list(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"查询单条记录时出错: {sql}\n参数: {params}\n错误: {e}")
            raise

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a record into the specified table.

        This function inserts a record into the specified table.
        It takes the table name and the data to be inserted as arguments, executes the INSERT statement,
        and commits the changes.

        Args:
            table (str): The name of the table to insert the record into.
            data (Dict[str, Any]): The data to be inserted into the table.

        Returns:
            int: The ID of the inserted record.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, tuple(data.values()))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"插入数据时出错: {sql}\n数据: {data}\n错误: {e}")
            self.conn.rollback()
            raise
        
    def batch_insert(self, table: str, data_list: List[List[Any]]) -> None:
        """
        Execute a batch insert of multiple records into the specified table.

        This function executes a batch insert of multiple records into the specified table.
        It takes the table name and a list of data to be inserted as arguments, executes the INSERT statement,
        and commits the changes.

        Args:
            table (str): The name of the table to insert the records into.
            data_list (List[List[Any]]): A list of data to be inserted into the table. Each item in the list is a list of values.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        if not data_list:
            return

        placeholders = ', '.join(['?'] * len(data_list[0]))
        sql = f"INSERT INTO {table} VALUES ({placeholders})"

        try:
            cursor = self.conn.cursor()
            cursor.executemany(sql, data_list)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"批量插入数据时出错: {sql}\n错误: {e}")
            self.conn.rollback()
            raise

    def update(self, table: str, data: Dict[str, Any], where_clause: str, where_params: tuple) -> int:
        """
        Update records in the specified table based on the given data and where clause.

        This function updates records in the specified table based on the given data and where clause.
        It takes the table name, the data to be updated, the where clause, and the parameters for the where clause as arguments,
        executes the UPDATE statement, and commits the changes.

        Args:
            table (str): The name of the table to update records in.
            data (Dict[str, Any]): The data to be updated.
            where_clause (str): The where clause to filter records.
            where_params (tuple): The parameters for the where clause.

        Returns:
            int: The number of affected rows.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        # Build the SET clause for the UPDATE statement
        set_clause = ', '.join([f"{key} = ?" for key in data.keys()])

        # Build the SQL statement
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

        # Build the parameters for the SQL statement
        params = tuple(data.values()) + where_params

        try:
            # Execute the SQL statement
            cursor = self.conn.cursor()
            cursor.execute(sql, params)

            # Commit the changes
            self.conn.commit()

            # Return the number of affected rows
            return cursor.rowcount
        except sqlite3.Error as e:
            # Log an error if an exception occurs
            logger.error(f"更新数据时出错: {sql}\n参数: {params}\n错误: {e}")

            # Rollback the changes
            self.conn.rollback()

            # Raise the exception
            raise

    def delete(self, table: str, where_clause: str, where_params: tuple) -> int:
        """
        Delete records from the specified table based on the given where clause.

        This function deletes records from the specified table based on the given where clause.
        It takes the table name, the where clause, and the parameters for the where clause as arguments,
        executes the DELETE statement, and commits the changes.

        Args:
            table (str): The name of the table to delete records from.
            where_clause (str): The where clause to filter records.
            where_params (tuple): The parameters for the where clause.

        Returns:
            int: The number of affected rows.

        Raises:
            sqlite3.Error: If an error occurs during execution.
        """
        # Build the SQL statement
        sql = f"DELETE FROM {table} WHERE {where_clause}"

        try:
            # Execute the SQL statement
            cursor = self.conn.cursor()
            cursor.execute(sql, where_params)

            # Commit the changes
            self.conn.commit()

            # Return the number of affected rows
            return cursor.rowcount
        except sqlite3.Error as e:
            # Log an error if an exception occurs
            logger.error(f"删除数据时出错: {sql}\n参数: {where_params}\n错误: {e}")

            # Rollback the changes
            self.conn.rollback()

            # Raise the exception
            raise

    def commit(self):
        """
        Commit the changes to the database.

        This function commits the changes to the database. It is a no-op if the connection is not established.

        Returns:
            None

        Raises:
            None
        """
        if self.conn:
            # Commit the changes
            self.conn.commit()

    def rollback(self):
        """
        Rollback the changes to the database.

        This function rolls back the changes to the database. It is a no-op if the connection is not established.

        Returns:
            None

        Raises:
            None
        """
        if self.conn:
            # Rollback the changes
            self.conn.rollback()
