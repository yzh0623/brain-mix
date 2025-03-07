"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/find_by_body_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:49
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil  # type: ignore

class TestElasticUtilFindByBody(unittest.TestCase):
    
    def setUp(self):
        """
        初始化ElasticUtil实例和测试数据。
        该函数在每次测试之前运行，用于初始化测试中使用的ElasticUtil实例，以及设置测试所需的索引名称、查询体和数据。
        """
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.body = {'query': {'match_all': {}}}
        self.data = {'key': 'value'}

    def test_index_exists_and_hits_not_empty(self):
        """
        测试存在的Elasticsearch索引中检索到的文档是否非空。
        该测试函数先删除并创建一个Elasticsearch索引，插入一个文档，
        然后通过find_by_body方法检索文档，并检查返回结果的长度是否为1。
        """
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.indices.create(index=self.index_name)
        _, self.id = self.elastic.insert(self.index_name, self.data)
        
        self.elastic.es.indices.refresh(index=self.index_name)
        
        result = self.elastic.find_by_body(self.index_name, self.body)
        self.assertEqual(len(result), 1)

    def test_index_exists_and_hits_empty(self):
        """
        测试存在的Elasticsearch索引中检索到的文档是否为空。
        该测试函数先删除并创建一个Elasticsearch索引，然后
        通过find_by_body方法检索文档，并检查返回结果的长度是否为0。
        """
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.indices.create(index=self.index_name)
        result = self.elastic.find_by_body(self.index_name, self.body)
        self.assertEqual(len(result), 0)

    def test_index_not_exists(self):
        """
        测试不存在的Elasticsearch索引中检索文档是否抛出异常。
        该测试函数尝试从不存在的Elasticsearch索引中检索文档，并检查是否抛出异常。
        """
        name = 'non_existent_index'
        with self.assertRaises(Exception):
            self.elastic.find_by_body(name, self.body)

    def test_body_empty(self):
        """
        测试find_by_body方法的body参数为空时是否抛出KeyError异常。
        该测试函数尝试使用一个空的body来调用find_by_body方法，并检查是否抛出KeyError异常。
        """
        body = {}
        with self.assertRaises(KeyError):
            self.elastic.find_by_body(self.index_name, body)


if __name__ == '__main__':
    unittest.main()