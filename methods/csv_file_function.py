# -*- coding: utf-8 -*- 

# import pandas as pd
import os
from methods.return_file_list import files_list

"""
    这里的函数中的column_name_str指的是csv中的列名字符串

"""

def is_column_exist(column_name_str,column_name):  #判断列名是否存在
    fields = column_name_str.split(',')
    if column_name in fields:
        return True
    else:
        return False

def column_position(column_name_str,column_name):  #返回列名所在的列序号
    fields = column_name_str.split(',')
    if is_column_exist(column_name_str,column_name):
        position = fields.index(column_name)
        return position

def column_value(line_str , column_name_str , column_name):
    fields = line_str.split(',')
    pos = column_position(column_name_str,column_name)
    col_value = fields[pos]
    return col_value 

    

        

