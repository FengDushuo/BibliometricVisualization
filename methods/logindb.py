#!/usr/bin/env python
# coding=utf-8
import pymysql
from methods.setdb import logconnect    #已整合进setdb.py中
 
loginconnect = logconnect    #连接数据库服务器
    
loginconn = loginconnect.cursor()        #创建操作游标
#你需要一个游标 来实现对数据库的操作相当于一条线索

loginconn.execute("CREATE TABLE IF NOT EXISTS users (id int(11) PRIMARY KEY ,username VARCHAR(20),password VARCHAR(40),email VARCHAR(40))")
 
# #查看
# conn.execute("SELECT * FROM user")    #选择查看自带的user这个表  (若要查看自己的数据库中的表先use XX再查看)
# rows = conn.fetchall()                #fetchall(): 接收全部的返回结果行，若没有则返回的是表的内容个数 int型
# for i in rows:
#     print(i)
 
# #创建表
# conn.execute("drop database if exists login_database")   #如果login_database数据库存在则删除
# conn.execute("create database login_database")   #新创建一个数据库
# conn.execute("use login_database")         #选择login_database这个数据库
# # sql 中的内容为创建一个名为users的表
# sql = """create table users(username VARCHAR(20),password BIGINT)"""  #()中的参数可以自行设置
# conn.execute("drop table if exists users") # 如果表存在则删除
# conn.execute(sql)   # 创建表
 
# sql = "INSERT INTO users (username, password) VALUES ('fengdushuo', 12345678)"
# conn.execute(sql)



# 删除
# conn.execute("drop table new_table")
 
# conn.close()           #   关闭游标连接
# connect.close()        #   关闭数据库服务器连接 释放内存
 