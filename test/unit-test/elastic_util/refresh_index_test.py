"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/refresh_index_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:22
"""

import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 


class TestElasticUtilRefreshIndex(unittest.TestCase):
    def setUp(self):
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.not_index_name = 'not_exists_index'
        
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.not_index_name)
        
        self.elastic.es.indices.create(index=self.index_name)

    def test_refresh_index_success(self):
        """
        测试刷新现有索引是否成功。
        此测试用例刷新索引并检查是否未引发异常。
        """
        self.elastic.refresh_index(self.index_name)

    def test_refresh_index_not_exists(self):
        """
        测试刷新不存在的索引时是否抛出异常。
        该测试函数尝试刷新不存在的索引，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.refresh_index('not_exists_index')

    def test_refresh_index_empty_name(self):
        """
        测试刷新索引时索引名称为空时是否抛出异常。
        该测试函数尝试刷新索引时索引名称为空，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.refresh_index('')

if __name__ == '__main__':
    unittest.main()