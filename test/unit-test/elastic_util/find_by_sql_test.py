"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/test/unit-test/elastic_util/find_by_sql_test.py
Author: yuanzhenhui
Date: 2024-11-05 08:04:51
LastEditTime: 2024-11-05 08:35:37
"""

import unittest

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from elastic_util import ElasticUtil # type: ignore 

class TestElasticUtilFindBySql(unittest.TestCase):

    def setUp(self):
        """
        初始化ElasticUtil实例和测试数据。
        该函数在每次测试之前运行，用于初始化测试中使用的ElasticUtil实例，以及设置测试所需的索引名称。
        """
        self.elastic = ElasticUtil()
        self.index_name = 'test_index'
        
        self.elastic.es.options(ignore_status=404).indices.delete(index=self.index_name) 
        self.elastic.es.indices.create(index=self.index_name)
        

    def test_01_find_by_sql_success(self):
        """
        测试find_by_sql方法是否能正确地执行SQL语句。
        该测试函数插入一个文档，然后使用find_by_sql方法来检索该文档，
        并检查返回结果的长度是否为1。
        """
        self.elastic.es.index(index=self.index_name, body={"key": "value"})
        self.elastic.es.indices.refresh(index=self.index_name)
        # 执行测试
        result = self.elastic.find_by_sql(f"SELECT * FROM {self.index_name}")
        # 断言结果是否正确
        self.assertEqual(len(result["rows"]), 1)
        
    def test_02_find_by_sql_fetch_size(self):
        """
        测试find_by_sql方法的fetch_size参数是否生效。
        该测试函数插入10个文档，然后使用fetch_size参数为5来检索文档，
        并检查返回结果的长度是否为5。
        """
        for i in range(10):
            self.elastic.es.index(index=self.index_name, body={"key": f"value_{i}"})
        self.elastic.es.indices.refresh(index=self.index_name)
        # 执行测试
        result = self.elastic.find_by_sql(f"SELECT * FROM {self.index_name}", fetch_size=5)
        # 断言结果是否正确
        self.assertEqual(len(result["rows"]), 5)

    def test_03_find_by_sql_empty_sql(self):
        """
        测试SQL语句为空时是否抛出异常。
        该测试函数尝试执行一个空的SQL查询，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.find_by_sql("")

    def test_04_find_by_sql_sql_error(self):
        """
        测试SQL语法错误时是否抛出异常。
        该测试函数尝试执行一个无效的SQL查询，并检查是否抛出异常。
        """
        with self.assertRaises(Exception):
            self.elastic.find_by_sql("SELECT * FROM invalid_index")

if __name__ == '__main__':
    unittest.main()