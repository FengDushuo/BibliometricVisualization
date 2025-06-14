#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from email.header import Header

my_sendername = "***"       # 发件人账号名 例如“dsfeng”
my_sender="***"    # 发件人邮箱账号 例如“1224556@163.com”
my_pass = "***"            # 发件人邮箱密码 例如“123456”
mail_host = "***"         # SMTP服务器 例如“stmp.163.com”

def send_email(message,subject,to_address):
    ret=True
    try:
        msg=MIMEText(message,'html','utf-8')
        msg['From']=formataddr([my_sendername,my_sender])   # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To']=formataddr(to_address)                   # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject']= Header(subject, 'utf-8').encode()  # 邮件的主题
 
        server=smtplib.SMTP_SSL("smtp.163.com", 465)        # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)                   # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender,[to_address[1],"***"],msg.as_string())       # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except smtplib.SMTPException as e:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        print(e)
        ret=False
    return ret

# 测试sendemail 
# token="khsadlkhdkajdhkajsdbanmdbadabhdjasdhsmnj"
# message ="""<h1>找回密码</h1>点击下面的链接重置密码<a href="http://127.0.0.1:8000/modify?token="""+token+"""">http://127.0.0.1:8000/modify?token="""+token
# ret=send_email(message,"修改密码",["fengdushuo","1369162653@qq.com"])
# if ret:
#     print("邮件发送成功")
# else:
#     print("邮件发送失败")
