"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/llms/llm_util.py
Author: yuanzhenhui
Date: 2024-10-22 11:29:51 
LastEditTime: 2025-01-09 00:06:51
"""
import torch
from transformers import StoppingCriteria

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

class StopOnTokens(StoppingCriteria):
    
    def __init__(self, token_ids):
        """
        Initialize the StopOnTokens instance with the given token IDs.

        Args:
            token_ids (list): A list of token IDs that will be used to
            determine when to stop the generation process.
        """
        self.token_ids = token_ids  # Store the provided token IDs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if the last token in the input sequence matches any of the token IDs in the `token_ids` list.

        Args:
            input_ids (torch.LongTensor): The input tensor of shape `(batch_size, sequence_length)`.
            scores (torch.FloatTensor): The scores tensor of shape `(batch_size, sequence_length, vocab_size)`.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: `True` if the last token matches any of the `token_ids`, `False` otherwise.
        """
        # Iterate over the token IDs and check if the last token in the input sequence matches any of them
        for stop_id in self.token_ids:
            # Check if the last token ID is equal to the current token ID
            if input_ids[0][-1] == stop_id:
                # If it matches, return True
                return True
        # If no match is found, return False
        return False