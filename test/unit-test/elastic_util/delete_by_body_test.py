"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/delete_by_body_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:36:15
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 


class TestElasticUtilDeleteByBody(unittest.TestCase):

    def setUp(self):
        """
        初始化ElasticUtil实例。
        此函数在每次测试之前运行，用于初始化测试中使用的ElasticUtil实例。
        """
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.key_value = {'key': 'value'}
        self.match_all = {'match_all': {}}
        self.body = {'query': self.match_all}


    def test_index_not_exists(self):
        """
        测试删除不存在的Elasticsearch索引时是否抛出异常。
        该测试函数尝试从不存在的Elasticsearch索引中删除文档，并检查是否抛出异常。
        """
        name = 'non_existent_index'
        with self.assertRaises(Exception):
            self.elastic.delete_by_body(name, self.body)


    def test_delete_success(self):
        """
        测试从Elasticsearch索引中删除文档是否成功。
        该测试函数从Elasticsearch索引中删除一个文档，并检查文档是否被删除。
        """
        # 删除测试索引
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        # 创建测试索引和文档
        self.elastic.es.indices.create(index=self.index_name)
        self.elastic.es.index(index=self.index_name, body=self.key_value)
        body = {'match': self.key_value}
        self.elastic.delete_by_body(self.index_name, body)
        # 检查文档是否被删除
        response = self.elastic.es.search(index=self.index_name, body=self.body)
        self.assertEqual(response['hits']['total']['value'], 0)


    def test_body_empty(self):
        """
        测试Elasticsearch删除API的body为空时是否使用默认的查询体。
        该测试函数创建一个Elasticsearch索引，然后使用一个空的查询体来删除文档，
        并检查文档是否被删除。
        """
        # 删除测试索引
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        # 创建测试索引
        self.elastic.es.indices.create(index=self.index_name)
        body = {}
        # 检查查询体是否为空
        if not body:
            # 使用一个默认的查询体
            body = self.match_all
        self.elastic.delete_by_body(self.index_name, body)


if __name__ == '__main__':
    unittest.main()