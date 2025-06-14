from methods.readdb import *
from methods.setdb import logconnect
import pymysql



def insertinfo(table,username,password,email):
	loginconnect = logconnect    #连接数据库服务器
		
	loginconn = loginconnect.cursor()        #创建操作游标
	print("start insert data into mysql!")
	id=select_maxid(table)+1
	sql_insert="INSERT INTO "+table+"(id,username,password,email) VALUES("+str(id)+",'"+username+"','"+password+"','"+email+"')"
	try:
	    loginconn.execute(sql_insert)
	    #提交
		loginconnect.commit()
	except Exception as e:
	    #错误回滚
	    loginconnect.rollback() 


def updateinfo(table,username,password,email):
	loginconnect = logconnect    #连接数据库服务器
		
	loginconn = loginconnect.cursor()        #创建操作游标
	print("start insert data into mysql!")
	sql_insert="UPDATE "+table+" SET "+"username='"+username+"',password='"+password+"',email='"+email+"' WHERE email='"+email+"';"
	try:
	    loginconn.execute(sql_insert)
	    #提交
		loginconnect.commit()
	except Exception as e:
	    #错误回滚
	    loginconnect.rollback() 