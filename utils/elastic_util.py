"""
Copyright (c) 2024 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/utils/elastic_util.py
Author: yuanzhenhui
Date: 2024-08-09 10:19:24
LastEditTime: 2024-11-04 00:35:10
"""

from yaml_util import YamlConfig 
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ElasticUtil:
    
    instance = None
    init_flag = False

    # 读取 elasticsearch 配置
    elastic_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'elastic_cnf.yml'))

    def __init__(self):
        """
        初始化 ElasticUtil 实例。

        此构造函数检查类的初始化标志。如果尚未初始化，则调用私有方法
        `__elastic_init_model` 来初始化 Elasticsearch 客户端，并将初始化标志设置为 True。
        """
        if not ElasticUtil.init_flag:
            self.es = None
            self.__elastic_init_model()
            ElasticUtil.init_flag = True


    def __new__(cls, *args, **kwargs):
        """
        一个静态方法，用于实例化 elastic_util 的单例对象.
        由于 elastic_util 仅需要一个实例，可以使用单例模式来确保只有一个实例被创建.
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


    def __elastic_init_model(self) -> None:
        """
        初始化Elasticsearch的client对象.

        该函数读取YAML配置文件，获取Elasticsearch的host、username、password、max_retries、max_size、timeout等配置项。
        然后使用这些配置项实例化Elasticsearch的client对象，并将其赋值给全局变量`es`。
        """
        host = ElasticUtil.elastic_config.get_value('es.host')
        username = ElasticUtil.elastic_config.get_value('es.username')
        password = ElasticUtil.elastic_config.get_value('es.password')
        max_retries = ElasticUtil.elastic_config.get_value('es.max-retries')
        max_size = ElasticUtil.elastic_config.get_value('es.max-size')
        timeout = ElasticUtil.elastic_config.get_value('es.timeout')
        
        self.es = Elasticsearch(host,
                           basic_auth=(username, password),
                           max_retries=max_retries,
                           connections_per_node=max_size,
                           request_timeout=timeout
                           )


    def insert(self, name, data) -> dict:
        """
        插入单个文档到Elasticsearch索引中。

        参数:
            name (str): Elasticsearch索引的名称。
            data (dict): 要插入的文档数据。

        返回:
            dict: 插入操作的结果。

        该函数在指定的Elasticsearch索引中插入一个文档。
        如果索引不存在，则抛出异常。
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        response = self.es.index(index=name, body=data)
        return response["_shards"]["successful"],response['_id']


    def batch_insert(self, name, datas) -> int:
        """
        批量插入文档到Elasticsearch索引中。

        该函数将多个文档插入到Elasticsearch索引中。

        参数:
            name (str): Elasticsearch索引的名称。
            datas (list): 要插入的文档列表，列表中的每个元素必须是字典类型。

        返回:
            None
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        if not all(isinstance(doc, dict) for doc in datas):
            raise TypeError("datas 中的所有元素必须是字典类型")
        
        actions = [
            {
                "_index": name,
                "_source": doc
            }
            for doc in datas
        ]
        response = bulk(self.es, actions)
        return response[0]
    
    
    def refresh_index(self, name) -> None:
        """
        重新刷新Elasticsearch索引，以便于最近插入的文档能够被搜索到。

        参数：
            name (str): Elasticsearch索引的名称。
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        self.es.indices.refresh(index=name)
           

    def delete_by_body(self, name, body) -> None:
        """
        根据给定的搜索体从Elasticsearch索引中删除文档。

        参数：
            name (str): Elasticsearch索引的名称。
            body (dict): 用于查找要删除的文档的搜索体。

        返回：
            None
        """
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        self.es.delete_by_query(index=name, query=body,refresh=True)
            

    def delete_by_id(self, name, id) -> dict:
        """
        通过ID在Elasticsearch中删除文档。
        
        参数：
            name (str): Elasticsearch索引的名称。
            id (str): 要删除的文档的ID。
        
        返回：
            dict: 删除操作的结果。
        """
        if id == '' or name == '':
            raise TypeError("params cannot be empty")
        
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        return self.es.delete(index=name, id=id,refresh=True)


    def find_by_id(self, name, id) -> dict:
        """
        通过ID在Elasticsearch中查找文档。

        参数：
            name (str): Elasticsearch索引的名称。
            id (str): 文档的ID。

        返回：
            dict: 文档的详细信息。
        """
        if id == '' or name == '':
            raise TypeError("params cannot be empty")
        
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        return self.es.get(index=name, id=id)
           

    def find_by_body(self, name, body) -> list:
        """
        通过给定的body在Elasticsearch中搜索并返回结果。

        参数：
            name (str): Elasticsearch索引的名称。
            body (dict): 搜索的body。

        返回：
            list: 搜索响应的结果列表。

        该函数使用Elasticsearch的search API执行搜索操作，并将所有的结果都return回去。
        """
        if name == '':
            raise TypeError("index cannot be empty")
        
        if body == {}:
            raise KeyError("body cannot be empty")
        
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        response = self.es.search(index=name, body=body)
        return response['hits']['hits']
         

    def find_by_body_nopaging(self, name, body) -> list:
        """
        通过给定的body在Elasticsearch中搜索并返回结果，且不分页。

        参数：
            name (str): Elasticsearch索引的名称。
            body (dict): 搜索的body。

        返回：
            list: 搜索响应的结果列表。

        该函数使用Elasticsearch的search API执行搜索操作，并使用scroll API来获取所有的结果。
        """
        if name == '':
            raise TypeError("index cannot be empty")
        
        if body == {}:
            raise KeyError("body cannot be empty")
        
        if not self.es.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")
        
        response = self.es.search(index=name, scroll='1m', body=body)
        # 获取 scroll_id 和初始结果
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        # 处理初始结果
        all_hits = hits
        # 循环获取剩余结果
        while len(hits) > 0:
            response = self.es.scroll(scroll_id=scroll_id, scroll='1m')
            hits = response['hits']['hits']
            all_hits.extend(hits)
        # 清除 scroll
        self.es.clear_scroll(scroll_id=scroll_id)
        return all_hits


    def find_and_create_index(self, yaml_key, mapping) -> str:
        """
        通过name从配置文件中获取对应的index_name，然后判断index是否存在，不存在则创建，最后返回index_name。

        参数:
            name (str): 在配置文件中配置的index name。
            mapping (dict): index的mapping。

        返回:
            str: 创建的index_name。
        """
        if yaml_key == '':
            raise TypeError("yaml_key cannot be empty")
        
        index_name = ElasticUtil.elastic_config.get_value(yaml_key)
        if not self.es.indices.exists(index=index_name) and mapping is not None:
            self.es.indices.create(index=index_name, body=mapping)
        return index_name
   
    
    def find_by_sql(self, sql, fetch_size=100) -> list:
        """
        执行Elasticsearch的SQL查询。

        参数：
            sql (str): Elasticsearch的SQL语句。
            fetch_size (int): 一次从Elasticsearch获取的文档数量。

        返回：
            list: JSON字符串列表，每个字符串表示一个文档。

        该函数执行Elasticsearch的SQL查询，并将结果以JSON字符串列表的形式返回。
        """
        return self.es.sql.query(format="json", query=sql, fetch_size=fetch_size)


    def update(self, name, data, id) -> dict:
        """
        更新Elasticsearch中的文档。

        参数:
            name (str): Elasticsearch索引的名称。
            data (dict): 包含更新字段及其新值的数据字典。
            id (str): 要更新的文档的ID。

        返回:
            dict: 更新操作的结果。

        该函数在指定的Elasticsearch索引中通过文档ID更新文档。返回更新操作的结果。
        """
        return self.es.update(index=name, id=id, body=data)
            