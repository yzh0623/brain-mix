"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/yaml_util.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:13
"""


import yaml

class YamlConfig:
    
    _instance = None

    def __init__(self, path):
        """
        初始化 YamlConfig 实例.
        此构造函数设置YAML文件的路径并加载其内容在`config`属性中。

        参数:
            path (str): YAML配置文档路径
        """
        self.path = path
        self.config = self.load_config()

    def __new__(cls, path):
        """
        一个静态方法，用于实例化 YamlConfig 的单例对象.
        由于 YamlConfig 仅需要一个实例，可以使用单例模式来确保只有一个实例被创建.

        参数:
            path (str): YAML配置文档路径.

        返回:
            YamlConfig: YamlConfig 的单例对象.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.path = path
            cls._instance.config = cls._instance.load_config()
        return cls._instance

    def load_config(self):
        """
        读取YAML配置文档的内容。

        读取并解析YAML配置文档，返回解析后的内容。

        返回:
            dict: 解析后的YAML配置文档内容。
        """
        with open(self.path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get_value(self, key):
        """
        通过key获取YAML配置文档中的值。

        参数:
            key (str): 键名，可能包含多个部分，例如a.b.c。

        返回:
            object: 通过key获取的值，可能是None。
        """
        key_parts = key.split('.')
        value = self.config
        for part in key_parts:
            value = value.get(part)
            if value is None:
                break
        return value