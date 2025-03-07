"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/llms/baai_util.py
Author: yuanzhenhui
Date: 2024-10-22 15:15:58
LastEditTime: 2025-01-10 22:44:10
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
from tqdm import tqdm

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlConfig
from logging_util import LoggingUtil 
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class BaaiUtil:
    
    _instance = None
    _initialized = False 
    
    def __init__(self):
        """
        Initializes an instance of the BaaiUtil class.

        This constructor checks if the class has been initialized. If not, 
        it calls the private method `_baai_init` to set up necessary 
        configurations and marks the class as initialized.
        """
        if not BaaiUtil._initialized:
            # Initialize the BAAI model and configurations
            self._baai_init()
            # Set the initialized flag to True
            BaaiUtil._initialized = True
    
    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of BaaiUtil is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one
        and assigns it to the `_instance` class variable. If an instance
        already exists, it simply returns the existing instance.

        Returns:
            BaaiUtil: The instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _baai_init(self):
        """
        Initializes the BAAI model configurations and loads necessary models.

        This method sets up the device for computation, loads configuration
        values from the YAML file, and initializes the embedding and reranker
        models along with their tokenizers.
        """
        # Set the device to use GPU if available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BAAI-specific configurations
        baai_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'llms', 'baai_cnf.yml'))
        self.normalize_embeddings = baai_config.get_value('baai.embedding.normalize_embeddings')
        self.use_padding = baai_config.get_value('baai.use_padding')
        self.use_truncation = baai_config.get_value('baai.use_truncation')
        self.reranker_batch_size = int(baai_config.get_value('baai.reranker.batch_size'))
        self.reranker_top = int(base_config.get_value('baai.reranker.top_k'))
        
        # Load base configurations
        base_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'llms', 'base_cnf.yml'))
        self.llm_max_token = int(base_config.get_value('llm.system.max_token'))
        self.model_device = base_config.get_value('vino_torch.model_device')
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(
            baai_config.get_value('baai.embedding.path'), 
            device=self.device
        )
        
        # Load and configure the reranker model
        reranker_path = baai_config.get_value('baai.reranker.path')
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_path,
            torch_dtype=torch.bfloat16,
            use_cache=base_config.get_value('vino_torch.use_cache')
        )
        self.reranker_model.to(self.device)
        
        # Initialize the tokenizer for the reranker model
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path, use_fast=True)
    
    def array_to_vetor(self, array):
        """
        Converts an array of text into vectors using the embedding model.

        Parameters:
            array (list): A list of text strings to be converted into vectors.

        Returns:
            list: A list of sentence embeddings obtained from the embedding model.
        """
        sentence_embeddings = []
        try:
            # Encode the array of text into sentence embeddings
            sentence_embeddings = self.embedding_model.encode(array, normalize_embeddings=self.normalize_embeddings)
        except Exception as e:
            # Log any exceptions that occur during encoding
            logger.error(e)
        return sentence_embeddings
    
    def array_to_reranker(self, query: str, doc_array: list) -> list:
        """
        Reranks a list of documents based on a given query using the reranker model.

        This method takes a query string and a list of documents as input and returns a
        list of the top-k documents that are most relevant to the query.

        Parameters:
            query (str): The query string to use for reranking.
            doc_array (list): A list of documents to rerank.

        Returns:
            list: A list of the top-k documents that are most relevant to the query.
        """
        results = []
        for i in tqdm(range(0, len(doc_array), self.reranker_batch_size), desc="Reranking..."):
            batch_docs = doc_array[i:i + self.reranker_batch_size]
            pairs_text = [f"{query} [SEP] {doc}" for doc in batch_docs]
            
            with torch.inference_mode(), torch.amp.autocast(self.device):
                # Tokenize the input documents and prepare inputs for the reranker model
                inputs = self.reranker_tokenizer(
                    pairs_text,
                    padding=self.use_padding,
                    truncation=self.use_truncation,
                    return_tensors=self.model_device,
                    max_length=self.llm_max_token
                ).to(self.device)
                
                # Run the reranker model to get scores for the input documents
                scores = self.reranker_model(**inputs).logits.squeeze()
                batch_results = list(zip(batch_docs, scores.cpu().tolist()))
                results.extend(batch_results)
    
        # Sort the results in descending order based on the scores and return the top-k documents
        reverse_array = sorted(results, key=lambda x: x[1], reverse=True)[:self.reranker_top]
        return [item[0] for item in reverse_array]
    