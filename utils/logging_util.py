"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/logging_util.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-08 15:46:19
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

    def __init__(self,name) -> None:
        
        # logging 配置
        logging_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        logging_path = os.path.join(YamlUtil(logging_cnf).get_value('logging.path'), os.path.basename(project_dir))
        logging_level = f"logging.{YamlUtil(logging_cnf).get_value('logging.level')}"
        logging_interval = int(YamlUtil(logging_cnf).get_value('logging.interval'))
        logging_backup_count = int(YamlUtil(logging_cnf).get_value('logging.backup_count'))
        
        super().__init__(name, eval(logging_level))
        
        # 创建日志存放目录
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
            
        # 设置输出格式
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s]-%(filename)s:%(funcName)s - %(lineno)d - %(message)s")

        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(logging_path,name), 
            when="D", 
            interval=logging_interval, 
            backupCount=logging_backup_count,
            encoding="utf-8")
        
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(eval(logging_level))
        self.addHandler(file_handler)
        file_handler.close()

        # 创建控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(eval(logging_level))
        console_handler.setFormatter(formatter)
        # 将控制台日志管理器添加到日志记录器
        self.addHandler(console_handler)
        console_handler.close()