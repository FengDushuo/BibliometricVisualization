import pymysql

mysqluser="root"
mysqlpassword="19971130"   #这里填自己的mysql帐号和密码
logindatabase="login_db"
hostsrc="localhost"

conn = pymysql.connect(host=hostsrc,port=3306,user=mysqluser,password=mysqlpassword,charset='utf8')
cursor = conn.cursor()
createloginsql="CREATE DATABASE IF NOT EXISTS "+logindatabase
cursor.execute(createloginsql)


logconnect = pymysql.connect(    #连接数据库服务器
    user=mysqluser,              #本地mysql用户名
    password=mysqlpassword,          #本地MySQL密码
    host=hostsrc,
    port=3306,
    db=logindatabase,
    charset="utf8"
)

loginconnect = logconnect                #连接数据库服务器
    
loginconn = loginconnect.cursor()        #创建操作游标
                                         #你需要一个游标 来实现对数据库的操作相当于一条线索

loginconn.execute("CREATE TABLE IF NOT EXISTS users (id int(11) PRIMARY KEY ,username VARCHAR(20),password VARCHAR(40),email VARCHAR(40))")
