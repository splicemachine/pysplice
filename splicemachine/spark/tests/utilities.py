"""
Copyright 2018 Splice Machine, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

class Assertions:    
    @staticmethod
    def basic_df_schema_assertions(out, event, schemaTableName):
        """
        Make the standard set of assertions for a function
        that takes in a dataframe (optional), and a schema table
        name
        """
        splitted = schemaTableName.split('.')
        assert out['event'] == event # validate scala function called
        assert out['schemaTableName'] == schemaTableName # validate argument 
        assert out['schemaName'] == splitted[0]
        assert out['tableName'] == splitted[1]

    @staticmethod
    def query_assertions(out, event, query):
        """
        Make the standard set of assertions for a function
        that takes in a query (SQL)
        """
        assert out['event'] == event # validate scala function
        assert out['query_string'] == query # validate SQL query
    
    @staticmethod
    def dict_assertions(assertions):
        """
        Assert that all keys match values in dictionary
        """
        print(list(assertions.keys()))
        print(list(assertions.values()))
        assert list(assertions.keys()) == list(assertions.values())
