"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/persistence/elastic_util.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2025-09-02 13:44:52
"""

import threading
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

elastic_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')

class ElasticUtil:

    _instances = {}  # 单例实例
    _lock = threading.Lock()  # 线程锁

    def __new__(cls, **kwargs):
        config_key = str(kwargs) if kwargs else 'default'
        with cls._lock:
            if config_key not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[config_key] = instance
            return cls._instances[config_key]

    def __init__(self, **kwargs) -> None:
        if hasattr(self, '_conn'):  # 防止重复初始化
            return

        self._init_kwargs = kwargs
        self._get_connection()
        
        # 添加心跳检测
        self._last_health_check = time.time()
        self._health_check_interval = 60

    def _get_connection(self):
        try:
            if self._init_kwargs:
                self._conn = Elasticsearch(
                    hosts=self._init_kwargs["host"],
                    basic_auth=(self._init_kwargs["username"], self._init_kwargs["password"]),
                    max_retries=self._init_kwargs["max_retries"],
                    connections_per_node=min(50, os.cpu_count() * 4),
                    request_timeout=self._init_kwargs["timeout"],
                    # 连接优化参数
                    retry_on_timeout=True,
                    sniff_on_start=False,  # 单节点无需嗅探
                    sniff_on_connection_fail=False
                )
            else:
                self._conn = Elasticsearch(
                    hosts=YamlUtil(elastic_cnf).get_value('persistence.elastic.host'),
                    basic_auth=(
                        YamlUtil(elastic_cnf).get_value('persistence.elastic.username'),
                        YamlUtil(elastic_cnf).get_value('persistence.elastic.password')
                    ),
                    max_retries=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.max_retries')),
                    connections_per_node=min(50, os.cpu_count() * 4),
                    request_timeout=int(YamlUtil(elastic_cnf).get_value('persistence.elastic.timeout')),
                    # 连接优化参数
                    retry_on_timeout=True,
                    sniff_on_start=False,  # 单节点无需嗅探
                    sniff_on_connection_fail=False
                )
        except Exception as e:
            logger.error(f"Create ES connection failed: {str(e)}")
            raise
    
    def _check_health(self):
        """执行健康检查"""
        now = time.time()
        if now - self._last_health_check > self._health_check_interval:
            try:
                # 双重检查：集群健康状态 + 实际查询测试
                health = self._conn.cluster.health()
                if health['status'] not in ('green', 'yellow'):
                    raise ConnectionError(f"Cluster status: {health['status']}")
                self._last_health_check = now
            except Exception as e:
                logger.warning(f"Elasticsearch health check failed: {str(e)}")
                try:
                    # 先关闭旧连接
                    if hasattr(self, '_conn'):
                        self._conn.close()
                    # 重建连接
                    self._get_connection()
                except Exception as reconnect_error:
                    logger.critical(f"Reconnect failed: {str(reconnect_error)}")
                finally:
                    self._last_health_check = time.time()

    def insert(self, name, data) -> dict:
        self._check_health()

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.index(index=name, body=data,refresh=True)
        return response["_shards"]["successful"], response['_id']
    
    def insert_idx(self, name, id, data) -> dict:
        self._check_health()

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.index(index=name,id=id, body=data,refresh=True)
        return response["_shards"]["successful"], response['_id']

    def batch_insert(self, name, datas) -> int:
        self._check_health()

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        if not all(isinstance(doc, dict) for doc in datas):
            raise TypeError("All elements in datas must be of dict type")

        actions = [
            {
                "_index": name,
                "_source": doc
            }
            for doc in datas
        ]

        response = bulk(self._conn, actions)
        self.refresh_index(name)
        return response[0]

    def refresh_index(self, name) -> None:
        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        self._conn.indices.refresh(index=name)

    def delete_by_body(self, name, body) -> dict:
        self._check_health()
        
        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        return self._conn.delete_by_query(index=name, query=body, refresh=True)

    def delete_by_id(self, name, id) -> dict:
        self._check_health()
        
        if id == '' or name == '':
            raise TypeError("params cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        return self._conn.delete(index=name, id=id, refresh=True)

    def find_by_id(self, name, id) -> dict:
        self._check_health()
        
        if id == '' or name == '':
            raise TypeError("params cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        return self._conn.get(index=name, id=id)

    def find_by_body(self, name, body) -> list:
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if body == {}:
            raise KeyError("body cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.search(index=name, body=body)
        return response['hits']['hits']
    
    def find_by_body_analysis(self, name, body) -> list:
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if body == {}:
            raise KeyError("body cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        return self._conn.search(index=name, body=body)

    def find_by_body_nopaging(self, name, body) -> list:
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if body == {}:
            raise KeyError("body cannot be empty")

        if not self._conn.indices.exists(index=name):
            raise Exception(f"Index {name} does not exist")

        response = self._conn.search(index=name, scroll='1m', body=body)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        all_hits = hits
        while len(hits) > 0:
            response = self._conn.scroll(scroll_id=scroll_id, scroll='1m')
            hits = response['hits']['hits']
            all_hits.extend(hits)
        self._conn.clear_scroll(scroll_id=scroll_id)
        return all_hits

    def create_index(self, name, mapping) -> None:
        self._check_health()
        
        if name == '':
            raise TypeError("index cannot be empty")

        if not self._conn.indices.exists(index=name) and mapping is not None:
            self._conn.indices.create(index=name, body=mapping)

    def find_by_sql(self, sql, fetch_size=100) -> list:
        self._check_health()
        return self._conn.sql.query(format="json", query=sql, fetch_size=fetch_size)

    def update(self, name: str, data: dict, id: str) -> dict:
        self._check_health()
        return self._conn.update(index=name, id=id, body=data, refresh=True)
    
    def update_by_query(self, name: str,data:dict) -> dict:
        self._check_health()
        return self._conn.update_by_query(index=name, body=data,conflicts="proceed", refresh=True, wait_for_completion=False)

    def __del__(self):
        self._conn.close()
