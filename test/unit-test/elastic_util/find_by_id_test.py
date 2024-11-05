"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/find_by_id_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:43
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil  # type: ignore


class TestElasticUtilFindById(unittest.TestCase):

    def setUp(self):
        """
        初始化ElasticUtil实例。
        该函数在每次测试之前运行，用于初始化测试中使用的ElasticUtil实例。
        """
        self.elastic = ElasticUtil()

        self.index_name = 'test_index'
        self.data = {'key': 'value'}

        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.indices.create(index=self.index_name)
        _, self.id = self.elastic.insert(self.index_name, self.data)

    def test_find_by_id_success(self):
        """
        测试查找存在的文档是否成功。
        该测试函数尝试通过ID来查找一个存在的文档，并检查查找结果是否与原数据相同。
        """
        result = self.elastic.find_by_id(self.index_name, self.id)
        self.assertEqual(result["_source"], self.data)

    def test_find_by_id_index_not_exists(self):
        """
        测试索引不存在时的查找。
        该测试函数尝试从不存在的索引中通过ID来查找一个文档，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.find_by_id('not_exists_index', self.id)

    def test_find_by_id_id_empty(self):
        """
        测试ID为空时的查找。
        该测试函数尝试使用一个空的ID来查找一个文档，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.find_by_id(self.index_name, '')

    def test_find_by_id_index_name_empty(self):
        """
        测试索引名称为空时的查找。
        该测试函数尝试使用一个空的索引名称来查找一个文档，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.find_by_id('', self.id)


if __name__ == '__main__':
    unittest.main()
