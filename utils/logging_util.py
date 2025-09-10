"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/logging_util.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-10 16:00:24
"""

import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import re
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import const_util as CU
from yaml_util import YamlUtil

class LoggingUtil(Logger):

    def __init__(self, name: str) -> None:
        """
        Initializes the logging system.

        Args:
            name (str): The name of the logger.
        """
        
        # Path to the logging configuration file
        logging_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        # Path to the logging directory
        logging_path = os.path.join(YamlUtil(logging_cnf).get_value('logging.path'), os.path.basename(project_dir))
        # Logging level
        logging_level = f"logging.{YamlUtil(logging_cnf).get_value('logging.level')}"
        # Logging interval in seconds
        logging_interval = int(YamlUtil(logging_cnf).get_value('logging.interval'))
        # Number of backup logs to keep
        logging_backup_count = int(YamlUtil(logging_cnf).get_value('logging.backup_count'))
        
        # Initialize the logger
        super().__init__(name, eval(logging_level))
        
        # Create the logging directory if it does not exist
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
            
        # Set the output format
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s]-%(filename)s:%(funcName)s - %(lineno)d - %(message)s")

        # Create a file handler
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(logging_path,name), 
            when="D", 
            interval=logging_interval, 
            backupCount=logging_backup_count,
            encoding="utf-8")
        
        # Set the suffix and extension for the log files
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        # Set the formatter and level for the file handler
        file_handler.setFormatter(formatter)
        file_handler.setLevel(eval(logging_level))
        # Add the file handler to the logger
        self.addHandler(file_handler)
        # Close the file handler
        file_handler.close()

        # Create a console handler
        console_handler = logging.StreamHandler()
        # Set the level for the console handler
        console_handler.setLevel(eval(logging_level))
        # Set the formatter for the console handler
        console_handler.setFormatter(formatter)
        # Add the console handler to the logger
        self.addHandler(console_handler)
        # Close the console handler
        console_handler.close()
