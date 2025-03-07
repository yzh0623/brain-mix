"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/elastic_util.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2025-01-13 23:38:25
"""

from yaml_util import YamlConfig 
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ElasticUtil:
    
    _instance = None
    _initialized = False 

    def __init__(self):
        """
        Initialize an instance of ElasticUtil.
        
        This constructor checks if the class has been initialized. If not, it
        calls the private method `__elastic_init_model` to initialize the Elasticsearch
        client and sets the initialized flag to True.
        """
        # Check if the class has been initialized
        if not ElasticUtil._initialized:
            # Initialize the Elasticsearch client
            self._elastic_init()
            # Set the initialized flag to True
            ElasticUtil._initialized = True

    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of ElasticUtil is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one
        and assigns it to the `_instance` class variable. If an instance
        already exists, it simply returns the existing instance.

        Returns:
            ElasticUtil: The instance of the class.
        """
        # Check if an instance already exists
        if cls._instance is None:
            # Create a new instance
            cls._instance = super().__new__(cls)
        # Return the instance
        return cls._instance

    def _elastic_init(self) -> None:
        """
        Initialize the Elasticsearch client object.

        This method reads the YAML configuration file, gets the Elasticsearch
        host, username, password, max_retries, max_size, and timeout from the
        configuration file, and initializes the Elasticsearch client with
        these parameters.
        """
        # Load elasticsearch config
        elastic_config = YamlConfig(
            os.path.join(project_dir, 'resources', 'config', 'elastic_cnf.yml')
        )
        host = elastic_config.get_value('es.host')
        username = elastic_config.get_value('es.username')
        password = elastic_config.get_value('es.password')
        max_retries = elastic_config.get_value('es.max-retries')
        max_size = elastic_config.get_value('es.max-size')
        timeout = elastic_config.get_value('es.timeout')
        
        # Initialize the Elasticsearch client
        self.es = Elasticsearch(
            host,
            basic_auth=(username, password),
            max_retries=max_retries,
            connections_per_node=max_size,
            request_timeout=timeout
        )

    def insert(self, name, data) -> tuple:
        """
        Insert a document into an Elasticsearch index.

        Args:
            name (str): The name of the Elasticsearch index to insert the document into.
            data (dict): The data to insert into the index.

        Returns:
            tuple: A tuple containing the number of successful shards (int) and the ID of the inserted document (str).

        Raises:
            Exception: If the index does not exist.
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        response = self.es.index(index=name, body=data)
        return response["_shards"]["successful"], response['_id']

    def batch_insert(self, name, datas) -> int:
        """
        Insert a list of documents into an Elasticsearch index.

        This method takes a list of documents and inserts them into an Elasticsearch
        index. The documents must be a list of dictionaries, where each dictionary
        represents a document to be inserted.

        Args:
            name (str): The name of the Elasticsearch index to insert the documents into.
            datas (list): A list of dictionaries, where each dictionary represents a document to be inserted.

        Returns:
            int: The number of documents successfully inserted.

        Raises:
            Exception: If the index does not exist.
            TypeError: If the documents are not a list of dictionaries.
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        if not all(isinstance(doc, dict) for doc in datas):
            raise TypeError("datas must be a list of dict")

        # Create the bulk API request
        actions = [
            {
                "_index": name,
                "_source": doc
            }
            for doc in datas
        ]

        # Execute the bulk API request
        response = bulk(self.es, actions)

        # Return the number of documents successfully inserted
        return response[0]
    
    def refresh_index(self, name: str) -> None:
        """
        Refresh an Elasticsearch index.

        This method refreshes the specified Elasticsearch index to make sure that
        the latest data is searchable.

        Args:
            name (str): The name of the Elasticsearch index to refresh.

        Raises:
            Exception: If the index does not exist.
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        self.es.indices.refresh(index=name)

    def delete_by_body(self, name: str, body: dict) -> None:
        """
        Delete documents from an Elasticsearch index by a query body.

        This method deletes documents from the specified Elasticsearch index
        that match the specified query body.

        Args:
            name (str): The name of the Elasticsearch index.
            body (dict): The query body to use to select the documents to delete.

        Raises:
            Exception: If the index does not exist.
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        self.es.delete_by_query(index=name, query=body, refresh=True)

    def delete_by_id(self, name: str, id: str) -> dict:
        """
        Delete a document from an Elasticsearch index by its ID.

        Args:
            name (str): The name of the Elasticsearch index.
            id (str): The ID of the document to delete.

        Returns:
            dict: The result of the delete operation.

        Raises:
            TypeError: If either the name or id is empty.
            Exception: If the index does not exist.
        """
        if id == '' or name == '':
            raise TypeError("params cannot be empty")
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        return self.es.delete(index=name, id=id, refresh=True)

    def find_by_id(self, name: str, id: str) -> dict:
        """
        Get a document from an Elasticsearch index by its ID.

        Args:
            name (str): The name of the Elasticsearch index.
            id (str): The ID of the document to retrieve.

        Returns:
            dict: The retrieved document.

        Raises:
            TypeError: If the index name or ID is empty.
            Exception: If the index does not exist.
        """
        if id == '' or name == '':
            raise TypeError("params cannot be empty")
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        return self.es.get(index=name, id=id)

    def find_by_body(self, name, body) -> list:
        """
        Search for documents in an Elasticsearch index using the specified body.

        Args:
            name (str): The name of the Elasticsearch index.
            body (dict): The search body.

        Returns:
            list: A list of search results.

        Raises:
            TypeError: If the index name is empty.
            KeyError: If the search body is empty.
            Exception: If the index does not exist.
        """
        if name == '':
            raise TypeError("index cannot be empty")
        if body == {}:
            raise KeyError("body cannot be empty")
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        # Perform the search query
        response = self.es.search(index=name, body=body)
        
        # Return the list of documents found
        return response['hits']['hits']

    def find_by_body_nopaging(self, name, body) -> list:
        """
        Search for documents in an Elasticsearch index using the specified body,
        without any pagination.

        This function performs a search query on the specified index and
        returns all the matching documents, without any pagination.

        Args:
            name (str): The name of the Elasticsearch index.
            body (dict): The search body.

        Returns:
            list: A list of search results.

        Raises:
            TypeError: If the index name is empty.
            KeyError: If the search body is empty.
            Exception: If the index does not exist.
        """
        if name == '':
            raise TypeError("index cannot be empty")
        if body == {}:
            raise KeyError("body cannot be empty")
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        # Perform a scroll search to get all matching documents
        response = self.es.search(index=name, scroll='1m', body=body)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        all_hits = hits
        
        # Scroll over the results to get all the documents
        while len(hits) > 0:
            response = self.es.scroll(scroll_id=scroll_id, scroll='1m')
            hits = response['hits']['hits']
            all_hits.extend(hits)
        
        # Clear the scroll context
        self.es.clear_scroll(scroll_id=scroll_id)
        
        # Return the list of documents found
        return all_hits

    def find_and_create_index(self, name, mapping) -> None:
        """
        Check if an index exists in Elasticsearch, and create it if it does not exist.

        Args:
            name (str): The name of the index.
            mapping (dict): The mapping of the index. If it is None, the index will be created without a mapping.

        Raises:
            TypeError: If the index name is empty.
        """
        if name == '':
            raise TypeError("index cannot be empty")
        
        # Check if the index already exists
        if not self.es.indices.exists(index=name):
            # If not, create the index
            if mapping is not None:
                # Create the index with the specified mapping
                self.es.indices.create(index=name, body=mapping)
    
    def find_by_sql(self, sql, fetch_size=100) -> list:
        """
        Search for documents in an Elasticsearch index using the specified SQL query.

        Args:
            sql (str): The SQL query to execute.
            fetch_size (int): The number of documents to return in each batch. Defaults to 100.

        Returns:
            list: A list of search results.
        """
        
        # Perform the SQL query
        return self.es.sql.query(format="json", query=sql, fetch_size=fetch_size)

    def update(self, name, data, id) -> dict:
        """
        Update a document in an Elasticsearch index.

        Args:
            name (str): The name of the Elasticsearch index.
            data (dict): The data to update the document with.
            id (str): The ID of the document to update.

        Returns:
            dict: The result of the update operation.

        Raises:
            TypeError: If the index name is empty.
            KeyError: If the document ID is empty.
            Exception: If the index does not exist.
        """
        if name == '':
            raise TypeError("index cannot be empty")
        if id == '':
            raise KeyError("id cannot be empty")
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        # Perform the update query
        return self.es.update(index=name, id=id, body=data)