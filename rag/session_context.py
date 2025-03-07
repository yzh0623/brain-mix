"""
Copyright (c) 2024 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/rag/session_context.py
Author: yuanzhenhui
Date: 2024-08-07 16:25:12
LastEditTime: 2025-01-14 22:33:14
"""

import time
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlConfig
from elastic_util import ElasticUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class SessionContext:
    
    _instance = None
    _initialized = False
    
    def __init__(self) -> None:
        """
        Initialize the SessionContext instance.

        This method initializes the SessionContext instance by calling the
        private method `_session_context_init` to set up the Elasticsearch
        client and index mappings. It also sets the `_initialized` flag to
        True.

        :return: None
        """
        if not SessionContext._initialized:
            self._session_context_init()
            SessionContext._initialized = True
            
    def __new__(cls, *args, **kwargs):
        """
        Creates a new instance of the SessionContext class.

        This method ensures that only one instance of the SessionContext class is created.
        If the instance does not exist, it creates a new one and assigns it to the `_instance` class variable.
        If an instance already exists, it simply returns the existing instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            SessionContext: The instance of the SessionContext class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
            
    def _session_context_init(self) -> None:
        """
        Initializes the SessionContext instance.

        This method reads the Elasticsearch configuration from the YAML file,
        creates the Elasticsearch client, and creates the indices for user
        sessions and context if they do not exist.
        """
        try:
            # Create an instance of the ElasticUtil class to interact with Elasticsearch
            self.es = ElasticUtil()
            
            # Read the Elasticsearch configuration from the YAML file
            elastic_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'elastic_cnf.yml'))
            # Get the index name and mapping configuration for user sessions
            self.session_idx = elastic_config.get_value('es.nlp.session_index')
            self.session_idx_mapping = elastic_config.get_value('es.nlp.session_mapping')
            # Create the index for user sessions if it does not exist
            self.es.find_and_create_index(self.session_idx,self.session_idx_mapping)
            
            # Get the index name and mapping configuration for context
            self.context_idx = elastic_config.get_value('es.nlp.context_index')
            self.context_idx_mapping = elastic_config.get_value('es.nlp.context_mapping')
            # Create the index for context if it does not exist
            self.es.find_and_create_index(self.context_idx,self.context_idx_mapping)
            
            # Read the base configuration for LLMs
            base_config = YamlConfig(os.path.join(project_dir, 'resources', 'config','llms', 'base_cnf.yml'))
            # Get the system content (prompt) from the configuration
            self.sys_content = base_config.get_value('llm.system.prompt')
        except Exception as e:
            # Log an error if the initialization fails
            logger.error(f"Session context init failed: {e}")
            raise

    def get_user_sessions(self, user_id) -> list:
        """
        Retrieve user sessions for a given user_id.

        This method queries the Elasticsearch index for sessions associated with the specified user_id.
        The sessions are sorted by chat_date in descending order.

        Args:
            user_id: The ID of the user whose sessions are to be retrieved.

        Returns:
            list: A list of dictionaries containing session IDs and session bodies.
                  Returns None if no sessions are found.
        """
        # Construct search body for querying Elasticsearch
        body = {
            "query": {
                "term": {
                    "user_id": user_id
                    }
                },
            "sort": [{
                "chat_date": {
                    "order": "desc"
                    }
                }]
        }
        
        # Perform search query without pagination
        responses = self.es.find_by_body_nopaging(self.session_idx, body)
        
        # Check if any sessions were found and return accordingly
        if len(responses) == 0:
            return None
        else:
            return [{'session_id': response['_id'], 'session_body': response['_source']} for response in responses]

    def save_user_session(self, user_id, title) -> str:
        """
        Save a new user session to the Elasticsearch index.

        This method creates a new session document with the provided user_id and title,
        and inserts it into the session index in Elasticsearch. The current timestamp is
        used as the chat_date.

        Args:
            user_id (str): The ID of the user for whom the session is being saved.
            title (str): The title of the session.

        Returns:
            str: The ID of the inserted session document in Elasticsearch.
        """
        # Prepare the session data to be inserted
        data = {
            "user_id": user_id,
            "chat_date": str(int(time.time() * 1000)),  # Current timestamp in milliseconds
            "title": title
        }
        
        # Insert the session data into the Elasticsearch index
        response = self.es.insert(self.session_idx, data)
        
        # Return the ID of the newly inserted session document
        return response['_id']
        
    def update_session_title(self, session_id, title) -> bool:
        """
        Update the title of a user session in the Elasticsearch index.

        This method updates the title of the session document with the specified
        session_id in the session index in Elasticsearch.

        Args:
            session_id (str): The ID of the session to update.
            title (str): The new title for the session.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        # Prepare the updated data for the session
        data = {
            "doc": {
                "title": title
                }
            }
        
        # Perform the update operation in the Elasticsearch index
        response = self.es.update(self.session_idx, data, session_id)
        
        # Return True if the update was successful, otherwise False
        return response['_shards']['successful']

    def delete_user_sessions(self, session_id: str) -> None:
        """
        Delete a user session from the Elasticsearch index.

        This method deletes the session document with the specified session_id
        from the session index in Elasticsearch.

        Args:
            session_id (str): The ID of the session to delete.
        """
        # Delete the session document from the Elasticsearch index
        self.es.delete_by_id(self.session_idx, session_id)
        
    def get_user_context(self, session_id, page, new_data) -> tuple:
        """
        Retrieve user context from the Elasticsearch index.

        This method queries the context index in Elasticsearch for all
        documents associated with the specified session_id, sorted by
        chat_date in ascending order. If page is -1, it returns all documents,
        otherwise it paginates the results and returns a subset of them.
        The method also saves any new data to the context index and
        returns the ID of the newly inserted document.

        Args:
            session_id (str): The ID of the session to retrieve context for.
            page (int): The page number to retrieve, or -1 for all documents.
            new_data (list): The new data to save to the context index.

        Returns:
            tuple: A tuple containing the ID of the newly inserted document,
                   or an empty string if no data was inserted, and the list
                   of context items.
        """
        context_id = ''
        current_context = []
        body = {
            "query": {
                "query_string": {
                    "query": f"session_id: '{session_id}'"
                    }
                },
            "sort": [{
                "chat_date": {"order": "asc"}
                }]
        }
        if page == -1:
            responses = self.es.find_by_body_nopaging(self.context_idx, body)
        else:
            body['size'] = 10
            body['from'] = page * 10
            responses = self.es.find_by_body(self.context_idx, body)
        
        # Prepend the system content to the current context
        current_context.append({
            "id": "", 
            "role": "system", 
            "content": self.sys_content
            })
        
        # Iterate over the query results and add them to the current context
        for response in responses:
            current_context.append({
                    "id": response['_id'], 
                    "role": response['_source']['role'], 
                    "content": response['_source']['content']
                    })
        
        # If there is new data, save it to the context index and get its ID
        if len(new_data) > 0:
            context_id = self.save_user_context(session_id, new_data)
        
        # Return the ID of the newly inserted document and the current context
        return context_id, current_context + new_data
        
    def save_user_context(self, session_id, new_data) -> str:
        """
        Save user context data to the Elasticsearch index.

        This method inserts a new context document into the context index
        in Elasticsearch. The document contains the session_id, role, content,
        and the current timestamp as chat_date.

        Args:
            session_id (str): The ID of the session to which the context belongs.
            new_data (list): A list containing the context data with role and content.

        Returns:
            str: The ID of the inserted context document in Elasticsearch.
        """
        # Construct the data to be inserted into the index
        data = {
            "session_id": session_id,
            "role": new_data[0]['role'],
            "content": new_data[0]['content'],
            "chat_date": str(int(time.time() * 1000))  # Current timestamp in milliseconds
        }
        
        # Insert the data into the Elasticsearch context index
        response = self.es.insert(self.context_idx, data)
        
        # Return the ID of the newly inserted document
        return response['_id']
        
    def delete_user_context_from_contextid(self, session_id, context_id) -> int:
        """
        Delete user context documents from the Elasticsearch index starting from a specific context ID.

        This method deletes context documents associated with the given session_id
        from the context index in Elasticsearch. The deletion starts from the specified
        context_id and continues for all subsequent documents.

        Args:
            session_id (str): The ID of the session whose context documents are to be deleted.
            context_id (str): The context ID from which deletion should start.

        Returns:
            int: The number of successfully deleted documents.
        """
        flag = False
        counter = 0

        # Construct the search body to query Elasticsearch for context documents
        body = {
            "query": {
                "query_string": {
                    "query": f"session_id: '{session_id}'"
                    }
                },
            "sort": [{
                    "chat_date": {
                        "order": "asc"
                        }
                    }]
        }
        
        # Perform search query without pagination
        responses = self.es.find_by_body_nopaging(self.context_idx, body)

        # Iterate over the query results and delete documents starting from the specified context_id
        for response in responses:
            if response['_id'] == context_id:
                flag = True
            if flag:
                # Delete the document by its ID and increment the success counter
                del_resp = self.es.delete_by_id(self.context_idx, response['_id'])
                counter += del_resp['_shards']['successful']

        # Return the number of successfully deleted documents
        return counter