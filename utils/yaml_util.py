"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/yaml_util.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2025-01-08 23:19:47
"""

import yaml

class YamlConfig:
    
    _instance = None

    def __init__(self, path):
        """
        Initializes a new instance of the YamlConfig class.

        Args:
            path (str): The path to the YAML configuration file.

        Attributes:
            path (str): The path to the YAML configuration file.
            config (dict): The configuration loaded from the YAML file.
        """
        self.path = path
        self.config = self.load_config()

    def __new__(cls, path):
        """
        Creates a new instance of the YamlConfig class.

        This method ensures that only one instance of the YamlConfig class is created per path.
        If the instance does not exist, it creates a new one and loads the configuration from the YAML file.
        If the instance already exists, it returns the existing instance.

        Args:
            path (str): The path to the YAML configuration file.

        Returns:
            YamlConfig: The instance of the YamlConfig class.
        """
        if cls._instance is None:
            # Create a new instance
            cls._instance = super().__new__(cls)
            cls._instance.path = path
            cls._instance.config = cls._instance.load_config()
        return cls._instance

    def load_config(self):
        """
        Reads the YAML configuration file and loads its content.

        This method reads the YAML configuration file at the specified path and loads its content into a dictionary.
        The loaded configuration is stored in the `config` attribute.

        Returns:
            dict: The loaded configuration.
        """
        with open(self.path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get_value(self, key):
        """
        Retrieve a value from the configuration using a dot-separated key.

        This method traverses the configuration dictionary using parts of
        the provided key, which are split by dots, to retrieve the nested value.

        Args:
            key (str): A dot-separated string representing the keys to traverse.

        Returns:
            The value from the configuration if all keys are found, otherwise None.
        """
        # Split the key by dot to get the hierarchy of keys
        key_parts = key.split('.')
        
        # Start with the root configuration
        value = self.config

        # Traverse the hierarchy of keys
        for part in key_parts:
            # Try to get the value for the current part
            value = value.get(part)
            # If any part is not found, return None
            if value is None:
                break

        # Return the found value or None
        return value
