import pymysql
from methods.setdb import logconnect


def select_table(table, column, condition, value ):
    loginconnect = logconnect    #连接数据库服务器
        
    loginconn = loginconnect.cursor()        #创建操作游标
    sql = "select " + column + " from " + table + " where " + condition + "='" + value + "'"
    loginconn.execute(sql)
    lines = loginconn.fetchall()
    return lines

def select_columns(table, column ):
    loginconnect = logconnect   #连接数据库服务器
        
    loginconn = loginconnect.cursor()        #创建操作游标
    sql = "select " + column + " from " + table
    loginconn.execute(sql)
    lines = loginconn.fetchall()
    return lines

def select_maxid(table):
    loginconnect = logconnect    #连接数据库服务器
        
    loginconn = loginconnect.cursor()        #创建操作游标
    sql = "select MAX(id) as maxid from "+table
    loginconn.execute(sql)
    lines=loginconn.fetchall()
    if lines[0][0]:
        return lines[0][0]
    else:
        return 0

def select_dics(table,column):
    loginconnect = logconnect    #连接数据库服务器
        
    loginconn = loginconnect.cursor()        #创建操作游标
    sql = "select " + column + " from " + table
    loginconn.execute(sql)
    desc = loginconn.description  # 获取字段的描述，默认获取数据库字段名称，重新定义时通过AS关键重新命名即可
    data_dict = [dict(zip([col[0] for col in desc], row)) for row in loginconn.fetchall()]  # 列表表达式把数据组装起来
    return data_dict

    
