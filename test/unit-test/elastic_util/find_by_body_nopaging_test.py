"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/find_by_body_nopaging_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:59
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 

class TestElasticUtilFindByBodyNoPaging(unittest.TestCase):

    def setUp(self):
        """
        setUp方法在每个测试函数执行前调用，以设置测试环境。
        该方法创建了一个Elasticsearch实例，并设置了要测试的Elasticsearch索引名称和查询体。
        """
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.body = {'query': {'match_all': {}}}
        self.elastic.es.options(ignore_status=400).indices.create(index=self.index_name)

    def tearDown(self):
        """
        tearDown方法在每个测试函数执行完毕后调用，以清理测试环境。
        该方法删除了在setUp方法中创建的Elasticsearch索引，以避免对其他测试造成影响。
        """
        self.elastic.es.options(ignore_status=[400,404]).indices.delete(index=self.index_name)

    def test_index_name_empty(self):
        """
        测试find_by_body_nopaging方法的index_name参数为空时是否抛出TypeError异常。
        该测试函数尝试使用一个空的index_name来调用find_by_body_nopaging方法，并检查是否抛出TypeError异常。
        """
        with self.assertRaises(TypeError):
            self.elastic.find_by_body_nopaging('', {})

    def test_body_empty(self):
        """
        测试find_by_body_nopaging方法的body参数为空时是否抛出KeyError异常。
        该测试函数尝试使用一个空的body来调用find_by_body_nopaging方法，并检查是否抛出KeyError异常。
        """
        with self.assertRaises(KeyError):
            self.elastic.find_by_body_nopaging(self.index_name, {})

    def test_index_not_exists(self):
        """
        测试在索引不存在时，find_by_body_nopaging方法是否抛出异常。
        该测试函数删除一个Elasticsearch索引，然后尝试从该索引中
        检索文档，并检查是否抛出异常。
        """
        self.elastic.es.indices.delete(index=self.index_name)
        with self.assertRaises(Exception):
            self.elastic.find_by_body_nopaging(self.index_name, self.body)

    def test_search_success(self):
        """
        测试find_by_body_nopaging方法在搜索结果非空时的返回值。
        该测试函数使用find_by_body_nopaging方法来搜索一个存在的文档，并检查返回结果的
        长度是否为1。
        """
        self.elastic.es.index(index=self.index_name, body={'key': 'value'})
        self.elastic.es.indices.refresh(index=self.index_name)
        result = self.elastic.find_by_body_nopaging(self.index_name, self.body)
        self.assertEqual(len(result), 1)

    def test_search_result_empty(self):
        """
        测试find_by_body_nopaging方法在搜索结果为空时的返回值。
        该测试函数使用find_by_body_nopaging方法来搜索一个不存在的文档，并检查返回结果是否为空。
        """
        result = self.elastic.find_by_body_nopaging(self.index_name, self.body)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()