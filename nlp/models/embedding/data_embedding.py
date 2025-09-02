"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/nlp/models/embedding/data_embedding.py
Author: yuanzhenhui
Date: 2025-07-25 15:14:31
LastEditTime: 2025-09-01 15:58:46
"""

from sentence_transformers import SentenceTransformer

import sys
import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

from logging_util import LoggingUtil
from yaml_util import YamlUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class DataEmbedding:

    _instance = None
    _initialized = False

    def __init__(self):
        if not DataEmbedding._initialized:
            nlp_cnf = os.path.join(project_dir, 'resources', 'config', 'nlp_cnf.yml')
            embedding_path = YamlUtil(nlp_cnf).get_value('models.embedding.path')
            self.normalize_embeddings = YamlUtil(nlp_cnf).get_value('models.embedding.normalize_embeddings')
            self.embedding_model = SentenceTransformer(embedding_path, device="cpu")
            DataEmbedding._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def array_to_embedding(self, text_array):
        sentence_embeddings = []
        try:
            sentence_embeddings = self.embedding_model.encode(
                text_array,
                normalize_embeddings=self.normalize_embeddings
            )
        except Exception as e:
            logger.error(e)
        return sentence_embeddings.tolist() if hasattr(sentence_embeddings, 'tolist') else sentence_embeddings
