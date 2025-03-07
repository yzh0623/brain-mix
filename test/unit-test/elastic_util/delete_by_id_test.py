"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/delete_by_id_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:36:11
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 

class TestElasticUtilDeleteById(unittest.TestCase):

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
        _,self.id = self.elastic.insert(self.index_name, self.data)
        

    def test_delete_success(self):
        """
        测试删除文档是否成功。
        该测试函数删除一个文档，然后检查文档是否被删除。
        """
        self.elastic.es.indices.refresh(index=self.index_name)
        # 删除数据
        result = self.elastic.delete_by_id(self.index_name, self.id)
        self.assertEqual(result['result'], 'deleted')


    def test_index_not_exists(self):
        """
        测试删除不存在的索引时是否抛出异常。
        该测试函数尝试删除不存在的索引，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.delete_by_id('not_exists_index', self.id)


    def test_id_empty(self):
        """
        测试删除空ID时是否抛出异常。
        该测试函数尝试删除空ID，并检查是否抛出异常。
        """
        with self.assertRaises(TypeError):
            self.elastic.delete_by_id(self.index_name, '')


    def test_index_name_empty(self):
        """
        测试删除空索引名时是否抛出异常。
        该测试函数尝试删除一个空索引名，并检查是否抛出异常。
        """
        with self.assertRaises(TypeError):
            self.elastic.delete_by_id('', self.id)


if __name__ == '__main__':
    unittest.main()