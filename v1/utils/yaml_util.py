"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/yaml_util.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-10 16:01:15
"""

import yaml
import threading

class YamlUtil:
    _instances = {}
    _lock = threading.Lock()

    def __init__(self, path):
        self.path = path

    def __new__(cls, path):
        """
        Get a new instance of the YamlUtil class.

        This method is used to create a new instance of the YamlUtil class or
        return an existing one if it already exists.

        The instance is stored in a dictionary with the path as the key. The
        _lock is used to ensure that only one thread can access the dictionary
        at a time.

        Parameters:
            path (str): The path to the YAML file to load.

        Returns:
            YamlUtil: A new instance of the YamlUtil class or an existing one if
                it already exists.
        """
        with cls._lock:
            if path not in cls._instances:
                # Create a new instance and store it in the dictionary
                instance = super().__new__(cls)
                instance.path = path
                instance.config = instance.load_config()
                cls._instances[path] = instance
            return cls._instances[path]

    def load_config(self):
        """
        Load configuration data from a YAML file.

        This method opens the YAML file at the specified path, reads its
        contents, and parses it into a Python dictionary.

        Returns:
            dict: The parsed configuration data.
        """
        with open(self.path, 'r', encoding='utf-8') as file:
            # Use safe_load() instead of load() to avoid arbitrary code execution
            return yaml.safe_load(file)

    def get_value(self, key):
        """
        Get a value from the configuration data using a dot-separated key.

        This method traverses the configuration data dictionary using the
        provided key. The key is split by the '.' character and each part is
        used to access the corresponding key in the dictionary. If any part
        of the key does not exist in the dictionary, the method returns None.

        Parameters:
            key (str): The dot-separated key to use to access the value.

        Returns:
            object: The value stored in the configuration data dictionary at
                the specified key or None if the key does not exist.
        """
        value = self.config
        for part in key.split('.'):
            value = value.get(part)
            if value is None:
                break
        return value
