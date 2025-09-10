"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/persistence/elastic_util.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2025-09-10 15:23:20
"""

import threading
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

elastic_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')

class ElasticUtil:

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, **kwargs):
        """
        Get the instance of ElasticUtil with the given config_key.
        
        Parameters:
            kwargs (dict): The config dict for the instance. If not provided, use the default config.
        
        Returns:
            ElasticUtil: The instance of ElasticUtil.
        """
        config_key = str(kwargs) if kwargs else 'default'
        with cls._lock:
            # Use a lock to ensure thread safety
            if config_key not in cls._instances:
                
                # Create a new instance if the config_key is not found
                instance = super().__new__(cls)
                cls._instances[config_key] = instance
            return cls._instances[config_key]

    def __init__(self, **kwargs) -> None:
        """
        Initialize the ElasticUtil instance with the given config.
        
        Parameters:
            kwargs (dict): The config dict for the instance. If not provided, use the default config.
        
        Returns:
            None
        """
        
        if hasattr(self, '_conn'):
            # If the instance already has a connection, just return
            return

        self._init_kwargs = kwargs
        self._get_connection()
        
        self._last_health_check = time.time()
        self._health_check_interval = 60

    def _get_connection(self):
        """
        Get the Elasticsearch connection using the provided config.

        The connection is created with the following parameters:

        - hosts: The Elasticsearch host(s)
        - basic_auth: The username and password for authentication
        - max_retries: The maximum number of retries to connect to Elasticsearch
        - connections_per_node: The number of connections to make to each node
        - request_timeout: The amount of time to wait for a request to be completed
        - retry_on_timeout: Whether to retry on timeout
        - sniff_on_start: Whether to sniff the cluster state on connection start
        - sniff_on_connection_fail: Whether to sniff the cluster state on connection failure

        If the config is not provided, use the default config from the YAML file.
        """
        try:
            if self._init_kwargs:
                self._conn = Elasticsearch(
                    hosts=self._init_kwargs["host"],
                    basic_auth=(self._init_kwargs["username"], self._init_kwargs["password"]),
                    max_retries=self._init_kwargs["max_retries"],
                    connections_per_node=min(50, os.cpu_count() * 4),
                    request_timeout=self._init_kwargs["timeout"],
                    retry_on_timeout=True,
                    sniff_on_start=False,
                    sniff_on_connection_fail=False
                )
            else:
                self._conn = Elasticsearch(
                    hosts=YamlUtil(elastic_cnf).get_value('persistence.elastic.host'),
                    basic_auth=(
                        YamlUtil(elastic_cnf).get_value('persistence.elastic.username'),
                        YamlUtil(elastic_cnf).get_value('persistence.elastic.password')
                    ),
                    max_retries=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.max_retries')),
                    connections_per_node=min(50, os.cpu_count() * 4),
                    request_timeout=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.timeout')),
                    retry_on_timeout=True,
                    sniff_on_start=False,
                    sniff_on_connection_fail=False
                )
        except Exception as e:
            logger.error(f"Create ES connection failed: {str(e)}")
            raise
    
    def _check_health(self):
        """
        Check the health of the Elasticsearch cluster and reconnect if necessary.

        This method is called periodically (every 60 seconds by default) to check the health of the Elasticsearch cluster.
        If the cluster is not in a "green" or "yellow" state, it will raise a ConnectionError.
        If an exception occurs during the health check, it will try to reconnect to the cluster.
        """
        now = time.time()
        if now - self._last_health_check > self._health_check_interval:
            try:
                # Get the health status of the cluster
                health = self._conn.cluster.health()
                if health['status'] not in ('green', 'yellow'):
                    
                    # If the cluster is not in a "green" or "yellow" state, raise a ConnectionError
                    raise ConnectionError(f"Cluster status: {health['status']}")
                self._last_health_check = now
            except Exception as e:
                
                # Log a warning if the health check fails
                logger.warning(f"Elasticsearch health check failed: {str(e)}")
                try:
                    
                    # Close the current connection
                    if hasattr(self, '_conn'):
                        self._conn.close()
                        
                    # Reconnect to the cluster
                    self._get_connection()
                except Exception as reconnect_error:
                    logger.critical(f"Reconnect failed: {str(reconnect_error)}")
                finally:
                    self._last_health_check = time.time()

    def insert(self, name, data) -> dict:
        """
        Insert a document into an Elasticsearch index.

        :param name: The name of the index.
        :type name: str
        :param data: The document to be inserted.
        :type data: dict
        :return: A tuple containing the number of successful shards and the ID of the inserted document.
        :rtype: tuple
        """
        self._check_health()

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.index(index=name, body=data, refresh=True)
        return response["_shards"]["successful"], response['_id']

    def batch_insert(self, name, datas) -> int:
        """
        Insert multiple documents into an Elasticsearch index in bulk.

        :param name: The name of the index.
        :type name: str
        :param datas: The list of documents to be inserted.
        :type datas: list
        :return: The number of successful operations.
        :rtype: int
        """
        self._check_health()

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        if not all(isinstance(doc, dict) for doc in datas):
            raise TypeError("All elements in datas must be of dict type")

        actions = [{"_index": name,"_source": doc} for doc in datas]
        response = bulk(self._conn, actions)
        self.refresh_index(name)
        return response[0]

    def refresh_index(self, name: str) -> None:
        """
        Refresh the given Elasticsearch index.

        This method will force the Elasticsearch index to refresh its data, which can be useful if you want to make sure that the data is up-to-date.

        :param name: The name of the index.
        :type name: str
        :raises Exception: If the index does not exist.
        """
        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Refresh the index to make sure the data is up-to-date
        self._conn.indices.refresh(index=name)

    def delete_by_body(self, name, body) -> dict:
        """
        Delete documents from an Elasticsearch index by body.

        This method will delete documents from the given Elasticsearch index based on the given query body.

        :param name: The name of the index.
        :type name: str
        :param body: The query body.
        :type body: dict
        :return: The deletion result.
        :rtype: dict
        :raises Exception: If the index does not exist.
        """
        self._check_health()
        
        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Delete documents from the index based on the given query body
        return self._conn.delete_by_query(index=name, query=body, refresh=True)

    def delete_by_id(self, name, id) -> dict:
        """
        Delete a document from an Elasticsearch index by ID.

        This method will delete a document from the given Elasticsearch index based on the given ID.

        :param name: The name of the index.
        :type name: str
        :param id: The ID of the document to be deleted.
        :type id: str
        :return: The deletion result.
        :rtype: dict
        :raises TypeError: If either the index name or the document ID is empty.
        :raises Exception: If the index does not exist.
        """
        self._check_health()
        
        if id == '' or name == '':
            raise TypeError("params cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Delete the document from the index based on the given ID
        return self._conn.delete(index=name, id=id, refresh=True)

    def find_by_id(self, name, id) -> dict:
        """
        Retrieve a document from an Elasticsearch index by ID.

        This method will retrieve a document from the given Elasticsearch index based on the given ID.

        :param name: The name of the index.
        :type name: str
        :param id: The ID of the document to be retrieved.
        :type id: str
        :return: The retrieved document.
        :rtype: dict
        :raises TypeError: If either the index name or the document ID is empty.
        :raises Exception: If the index does not exist.
        """
        self._check_health()
        
        if id == '' or name == '':
            raise TypeError("params cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Retrieve the document from the index based on the given ID
        return self._conn.get(index=name, id=id)

    def find_by_body(self, name, body) -> list:
        """
        Retrieve a list of documents from an Elasticsearch index based on the given query body.

        This method will retrieve a list of documents from the given Elasticsearch index based on the given query body.

        :param name: The name of the index.
        :type name: str
        :param body: The query body to retrieve the documents.
        :type body: dict
        :return: The retrieved documents.
        :rtype: list
        :raises TypeError: If either the index name or the query body is empty.
        :raises Exception: If the index does not exist.
        """
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if body == {}:
            raise KeyError("body cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.search(index=name, body=body)
        return response['hits']['hits']

    def find_by_body_nopaging(self, name, body) -> list:
        """
        Retrieve a list of documents from an Elasticsearch index based on the given query body without paging.

        This method will retrieve a list of documents from the given Elasticsearch index based on the given query body
        without any paging (i.e., it will retrieve all the documents that match the query). The method will return a
        list of dictionaries, where each dictionary represents a document in the index.

        :param name: The name of the index.
        :type name: str
        :param body: The query body to retrieve the documents.
        :type body: dict
        :return: The retrieved documents.
        :rtype: list
        :raises TypeError: If either the index name or the query body is empty.
        :raises Exception: If the index does not exist.
        """
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if body == {}:
            raise KeyError("body cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Retrieve the first batch of documents
        response = self._conn.search(index=name, scroll='1m', body=body)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        all_hits = hits
        while len(hits) > 0:
            
            # Retrieve the next batch of documents
            response = self._conn.scroll(scroll_id=scroll_id, scroll='1m')
            hits = response['hits']['hits']
            
            # Extend the list of all documents with the current batch
            all_hits.extend(hits)
            
        # Clear the scroll ID
        self._conn.clear_scroll(scroll_id=scroll_id)
        return all_hits

    def find_by_sql(self, sql, fetch_size=100) -> list:
        """
        Execute a SQL query on an Elasticsearch index and return the results.

        :param sql: The SQL query to execute.
        :type sql: str
        :param fetch_size: The number of records to retrieve per batch.
        :type fetch_size: int
        :return: The results of the query.
        :rtype: list
        """
        self._check_health()
        # Execute the SQL query and return the results
        return self._conn.sql.query(format="json", query=sql, fetch_size=fetch_size)

    def update(self, name: str, data: dict, id: str) -> dict:
        """
        Update a document in an Elasticsearch index.

        This method will update a document in the given Elasticsearch index with the given data.

        :param name: The name of the index.
        :type name: str
        :param data: The data to update the document with.
        :type data: dict
        :param id: The ID of the document to update.
        :type id: str
        :return: The result of the update operation.
        :rtype: dict
        """
        self._check_health()
        return self._conn.update(index=name, id=id, body=data, refresh=True)
    
    def update_by_query(self, name: str, data: dict) -> dict:
        """
        Update multiple documents in an Elasticsearch index by a query.

        This method will update multiple documents in the given Elasticsearch index with the given data
        based on the query.

        :param name: The name of the index.
        :type name: str
        :param data: The data to update the documents with.
        :type data: dict
        :return: The result of the update operation.
        :rtype: dict
        """
        self._check_health()
        return self._conn.update_by_query(index=name, body=data, conflicts="proceed", refresh=True, wait_for_completion=False)

    def create_index(self, name, mapping) -> None:
        """
        Create an Elasticsearch index with the given name and mapping.

        :param name: The name of the index.
        :type name: str
        :param mapping: The mapping of the index.
        :type mapping: dict
        :return: None
        :rtype: None
        :raises TypeError: If the index name is empty.
        """
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if not self._conn.indices.exists(index=name) and mapping is not None:
            # Create the index with the given mapping
            self._conn.indices.create(index=name, body=mapping)

    def __del__(self):
        self._conn.close()
