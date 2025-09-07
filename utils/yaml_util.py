"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/yaml_util.py
Author: Zhenhui Yuan
Date: 2025-09-05 09:56:19
LastEditTime: 2025-09-07 13:14:03
"""

import yaml
import threading

class YamlUtil:
    _instances = {}  # 存储不同配置文件的实例
    _lock = threading.Lock()  # 线程锁

    def __init__(self, path):
        # 初始化方法仅用于实例属性设置
        self.path = path

    def __new__(cls, path):
        with cls._lock:
            if path not in cls._instances:
                instance = super().__new__(cls)
                instance.path = path
                instance.config = instance.load_config()  # 首次加载配置
                cls._instances[path] = instance
            return cls._instances[path]

    def load_config(self):
        """
        从YAML配置文件中加载配置数据。
        
        @return: 解析后的配置数据，通常为字典类型
        @rtype: dict
        """
        with open(self.path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get_value(self, key):
        """
        根据点分路径从配置中获取值

        参数:
            key (str): 点分路径字符串（如'section.subsection.key'）

        返回:
            Any: 对应的配置值，若路径不存在则返回None
        """
        value = self.config
        for part in key.split('.'):
            value = value.get(part)
            if value is None:
                break
        return value
