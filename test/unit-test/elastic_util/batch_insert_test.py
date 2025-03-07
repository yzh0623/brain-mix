"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/batch_insert_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:36:19
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 


class TestElasticUtilBatchInsert(unittest.TestCase):
    def setUp(self):
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.elastic.es.indices.create(index=self.index_name)
        
    def tearDown(self):
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)  

    def test_batch_insert_success(self):
        """
        测试批量插入文档到Elasticsearch索引成功的用例。
        该测试函数批量插入2个文档到Elasticsearch索引中，并检查插入是否成功。
        """
        datas = [
            {'key': 'value'}, 
            {'key2': 'value2'}, 
            {'key3': 'value3'}, 
            {'key4': 'value4'}, 
            {'key5': 'value5'}, 
            {'key6': 'value6'}, 
            {'key7': 'value7'}
            ]
        self.elastic.batch_insert(self.index_name, datas)

    def test_batch_insert_index_not_exists(self):
        """
        测试批量插入文档到不存在的Elasticsearch索引失败的用例。
        该测试函数尝试批量插入2个文档到不存在的Elasticsearch索引中，并检查是否抛出异常。
        """
        name = 'not_exists_index'
        self.elastic.es.options(ignore_status=404).indices.delete(index=name)  
        datas = [{'key': 'value'}, {'key2': 'value2'}]
        with self.assertRaises(Exception):
            self.elastic.batch_insert(name, datas)

    def test_batch_insert_empty_datas(self):
        """
        测试批量插入空文档列表的用例。
        该测试函数尝试批量插入一个空的文档列表到Elasticsearch索引中，
        用于确保在没有数据插入时不会抛出异常。
        """
        datas = []
        self.elastic.batch_insert(self.index_name, datas)

    def test_batch_insert_datas_contain_non_dict(self):
        """
        测试批量插入的文档列表中包含非字典对象时是否抛出异常。
        该测试函数尝试批量插入包含非字典对象的文档列表到Elasticsearch索引中，
        并检查是否抛出TypeError异常。
        """
        datas = [{'key': 'value'}, 'not_dict']
        with self.assertRaises(TypeError):
            self.elastic.batch_insert(self.index_name, datas)

if __name__ == '__main__':
    unittest.main()