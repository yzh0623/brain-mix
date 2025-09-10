"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/nlp/models/embedding/data_embedding.py
Author: yuanzhenhui
Date: 2025-07-25 15:14:31
LastEditTime: 2025-09-10 15:42:33
"""

from sentence_transformers import SentenceTransformer

import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from logging_util import LoggingUtil
from yaml_util import YamlUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class DataEmbedding:

    _instance = None
    _initialized = False

    def __init__(self):
        """
        Initialize the DataEmbedding class.
        
        This class is used to get the embeddings of a given text array.
        
        The embedding model is loaded from the path specified in the config file.
        The normalize_embeddings variable is used to decide whether to normalize the embeddings or not.
        """
        if not DataEmbedding._initialized:
            nlp_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE , 'nlp_cnf.yml')
            embedding_path = YamlUtil(nlp_cnf).get_value('models.embedding.path')
            self.normalize_embeddings = YamlUtil(nlp_cnf).get_value('models.embedding.normalize_embeddings')
            """
            Load the embedding model from the path.
            
            The device parameter is set to "cpu" to use the CPU for inference.
            """
            self.embedding_model = SentenceTransformer(embedding_path, device="cpu")
            DataEmbedding._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def array_to_embedding(self, text_array):
        """
        Convert a text array to embeddings.

        Args:
            text_array (list): The text array to convert to embeddings.

        Returns:
            list: The embeddings of the text array.
        """
        sentence_embeddings = []
        try:
            # Encode the text array using the embedding model
            sentence_embeddings = self.embedding_model.encode(
                text_array,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Convert the embeddings to a list
            if hasattr(sentence_embeddings, 'tolist'):
                return sentence_embeddings.tolist()
            else:
                return sentence_embeddings
        except Exception as e:
            # If there is an exception, log it and return an empty list
            logger.error(f"Failed to convert text array to embeddings: {str(e)}")
            return []
