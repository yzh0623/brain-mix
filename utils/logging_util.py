"""
Copyright (c) 2024 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/utils/logging_util.py
Author: yuanzhenhui
Date: 2024-09-26 08:57:55
LastEditTime: 2025-01-05 21:57:02
"""

import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import re
import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from yaml_util import YamlConfig

class LoggingUtil(Logger):

    """
    A logging utility class for the Brain Mix project.
    This class provides a default logging configuration for the project.
    """
    def __init__(self, name: str) -> None:
        """
        Initializes a new instance of the LoggingUtil class.

        Args:
            name (str): The name of the logger.
        """
        
        logging_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'logging_cnf.yml'))
        logging_path = os.path.join(logging_config.get_value('logging.path'), os.path.basename(project_dir))
        logging_level = f"logging.{logging_config.get_value('logging.level')}"
        logging_interval = int(logging_config.get_value('logging.interval'))
        logging_backup_count = int(logging_config.get_value('logging.backup_count'))
        
        super().__init__(name, eval(logging_level))
        
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
            
        formatter = logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s")

        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(logging_path, name), 
            when="D", 
            interval=logging_interval, 
            backupCount=logging_backup_count,
            encoding="utf-8")
        # Set the log file suffix to the current date
        file_handler.suffix = "%Y-%m-%d.log"
        # Set the log file extension match to .log
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(eval(logging_level))
        self.addHandler(file_handler)
        # Close the file handler after adding it to the logger
        file_handler.close()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(eval(logging_level))
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        # Close the console handler after adding it to the logger
        console_handler.close()
