"""
Copyright (c) 2024 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/utils/jieba_util.py
Author: yuanzhenhui
Date: 2024-11-11 12:00:09
LastEditTime: 2025-01-08 23:58:26
"""

import jieba.analyse
import jieba.posseg as pseg

from yaml_util import YamlConfig 
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class JiebaUtil:
    
    _instance = None
    _initialized = False 
    
    def __init__(self):
        """
        Initialize the JiebaUtil instance.
        The instance is only initialized once and subsequent calls to the
        constructor will return the same instance.
        """
        if not JiebaUtil._initialized:
            self._jieba_init()
            JiebaUtil._initialized = True
            
    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of JiebaUtil is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one
        and assigns it to the `_instance` class variable. If an instance
        already exists, it simply returns the existing instance.

        Returns:
            JiebaUtil: The instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _jieba_init(self) -> None:
        """
        Initialize the JiebaUtil instance with the configurations.

        This method is called once when the instance is created. It reads the
        Jieba configurations from the YAML file and sets the configurations to
        the instance variables.
        """
        jieba_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'jieba_cnf.yml'))
        self.top_k = jieba_config.get_value('jieba.top_k')
        self.weight = float(jieba_config.get_value('jieba.weight'))
        self.allowed_pos = jieba_config.get_value('jieba.allowed_pos')
        
        # Load the Traditional Chinese Medicine keywords from the file
        tcm_keyword_path = os.path.join(project_dir, 'resources', 'jieba', 'tcm_keyword_dict.txt')
        with open(tcm_keyword_path, 'r', encoding='utf-8') as f:
            tcm_keywords = f.read().split('\n')
        self.tcm_keyword_arr = tcm_keywords
    
    def get_keyword(self, texter) -> list:
        """
        Extract keywords from the given text with Jieba.

        The function uses Jieba's `extract_tags` method to extract keywords
        from the given text. The keywords are filtered by their part-of-speech
        tags and weights. The function returns a list of the extracted keywords.

        Args:
            texter (str): The text to extract keywords from.

        Returns:
            list: A list of the extracted keywords.
        """
        matches = []
        keywords = jieba.analyse.extract_tags(
            texter, 
            topK=self.top_k, 
            withWeight=True, 
            allowPOS=self.allowed_pos
            )
        for keyword, weight in keywords:
            if weight > self.weight:  # Filter the keywords by their weights
                current_matches = [line for line in self.tcm_keyword_arr if keyword == line]
                if current_matches:
                    matches.extend(current_matches)
        return list(set(matches))
    
    def extract_nouns(self, texter) -> list:
        """
        Extract all nouns from the given text.

        This function uses Jieba's `pseg.cut` method to extract all words
        from the given text and filter them by their part-of-speech
        tags. The function returns a list of all nouns found in the text.

        Args:
            texter (str): The text to extract nouns from.

        Returns:
            list: A list of all nouns found in the text.
        """
        nouns_dict = dict.fromkeys([])
        words = pseg.cut(texter)
        for word, flag in words:
            if flag.startswith('n'):
                nouns_dict[word] = None
        return list(nouns_dict.keys())