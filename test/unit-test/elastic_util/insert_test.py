"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/insert_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:30
"""

import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 


class TestElasticUtilInsert(unittest.TestCase):

    def setUp(self):
        """
        初始化ElasticUtil实例。
        此函数在每次测试之前运行，用于初始化测试中使用的ElasticUtil实例。
        """
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.data = {'key': 'value'}
        self.elastic.es.indices.create(index=self.index_name)
        
    def tearDown(self):
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)

    def test_insert_success(self):
        """
        测试向Elasticsearch索引中插入文档是否成功。
        该测试函数向Elasticsearch索引中插入一个文档，然后使用get API来检查文档是否插入成功。
        """
        _,insert_id = self.elastic.insert(self.index_name, self.data)
        # 检查数据是否插入成功
        result = self.elastic.es.get(index=self.index_name, id=insert_id)
        self.assertEqual(result['_source'], self.data)

    def test_insert_failure_index_not_exists(self):
        """
        测试向不存在的Elasticsearch索引中插入文档是否失败。
        该测试函数尝试向不存在的Elasticsearch索引中插入一个文档，并检查是否抛出异常。
        """
        name = 'non_existent_index'
        self.elastic.es.options(ignore_status=404).indices.delete(index=name)  # 删除索引
        with self.assertRaises(Exception):
            self.elastic.insert(name, self.data)

    def test_insert_failure_elasticsearch_connection_error(self):
        """
        测试Elasticsearch连接出错时插入文档是否失败。
        该测试函数模拟Elasticsearch连接错误，然后尝试向Elasticsearch索引中插入一个文档，并检查是否抛出异常。
        """
        original_es = self.elastic.es
        self.elastic.es = None
        with self.assertRaises(Exception):
            self.elastic.insert(self.index_name, self.data)
        self.elastic.es = original_es

    def test_insert_failure_data_format_error(self):
        """
        测试插入格式错误的数据时是否抛出异常。
        该测试函数尝试插入一个无效格式的数据到Elasticsearch索引中，并检查是否抛出异常。
        """
        data = 'invalid data'
        with self.assertRaises(Exception):
            self.elastic.insert(self.index_name, data)


if __name__ == '__main__':
    unittest.main()