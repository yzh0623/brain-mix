"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/find_and_create_index_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:36:06
"""
import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 

class TestFindAndCreateIndex(unittest.TestCase):
    def setUp(self):
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        self.test_name = 'es.test-index'
        self.test_name_comp = self.elastic.elastic_config.get_value(self.test_name)
        self.mapping = {
                "mappings": {
                    "properties": {
                        "overcon": {"type": "text"}
                    }
                }
            }

    def test_index_exists(self):
        # 模拟索引存在
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.test_name_comp)
        result = self.elastic.find_and_create_index(self.test_name,self.mapping)
        self.assertEqual(result, self.test_name_comp)

    def test_index_not_exists_and_mapping_not_empty(self):
        # 模拟索引不存在且mapping不为空
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.test_name_comp)
        result = self.elastic.find_and_create_index(self.test_name, self.mapping)
        self.assertEqual(result, self.test_name_comp)
        self.assertFalse(self.elastic.es.indices.exists(index=self.index_name))

    def test_index_not_exists_and_mapping_empty(self):
        # 模拟索引不存在且mapping为空
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name)
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.test_name_comp)
        result = self.elastic.find_and_create_index(self.test_name, None)
        self.assertEqual(result, self.test_name_comp)
        self.assertFalse(self.elastic.es.indices.exists(index=self.index_name))

    def test_name_empty(self):
        # 测试name为空时，抛出异常
        with self.assertRaises(Exception):
            self.elastic.find_and_create_index('', self.mapping)


if __name__ == '__main__':
    unittest.main()