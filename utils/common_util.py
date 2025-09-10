"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/common_util.py
Author: yuanzhenhui
Date: 2025-09-01 18:23:48
LastEditTime: 2025-09-10 15:57:27
"""
import json
from pathlib import Path

class CommonUtil:
    
    @staticmethod
    def split_array(lst, num) -> list:
        """
        Split an array into several sub arrays.

        This function takes an array and a number of sub arrays to split it into and
        returns a list of sub arrays.

        :param lst: The array to split.
        :type lst: list
        :param num: The number of sub arrays to split the array into.
        :type num: int
        :return: A list of sub arrays.
        :rtype: list
        """
        if len(lst) < num:
            return [lst]
        avg = len(lst) // num
        remainder = len(lst) % num
        result = []
        start = 0

        # Iterate over the array and split it into sub arrays
        for i in range(num):
            end = start + avg + (1 if i < remainder else 0)
            result.append(lst[start:end])
            start = end
        return result
    
    @staticmethod
    def get_file_paths(directory, suffix):
        """
        Get all unique file paths in a directory.

        This function will recursively search the directory for all files with the
        .json suffix and return a list of unique file paths.

        Args:
            directory (str): The directory to search for files in.

        Returns:
            list: A list of unique file paths.
        """
        unique_files = set()
        for path in Path(directory).rglob('*'):
            if path.is_file() and path.suffix == suffix:
                unique_files.add(str(path))
        return list(unique_files)
    
    @staticmethod
    def load_json_file(file_path):
        """
        Load a JSON file and return the data.

        Args:
            file_path (str): The path to the JSON file to load.

        Returns:
            list: The data from the JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data