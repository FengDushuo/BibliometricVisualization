import tornado.web
import tornado.escape
import methods.readdb as mrd
import methods.writedb as mwd
from handlers.base import BaseHandler
from methods.sendemail import send_email
import os
import time
from hashlib import sha1

    
token = sha1(bytes("%s%s" % (os.urandom(16),time.time()),encoding="utf-8")).hexdigest()
User_Info={}

class SendEmailHandler(BaseHandler):    #继承base.py中的类BaseHandler
    def get(self):
        self.render("forgetpwd.html",type_="forgetpwd")

    def post(self):
        domain=self.get_argument("domain",default=None)
        email = self.get_argument("email",default=None)
        dbinfo = mrd.select_dics("users","*")
        usrs,pwds,emails=zip(*map(lambda x: (x["username"],x["password"],x["email"]),dbinfo))
        return_data = {}
        if email=="":
            return_data = {"info": "no_email_here!"}
            self.write( return_data)
        else:
            if email in emails:
                user_infos = mrd.select_table(table="users",column="*",condition="email",value=email)
                username = user_infos[0][1]
                User_Info={"username":username,"email":email,"token":token}
                message = "<h1>找回密码</h1>点击下面的链接重置密码<a href=\"http://"+domain+":8000/modify\">http://"+domain+":8000/modify</a>"
                
                if send_email(message,"密码找回",[username,email]):
                    return_data = {"info": "success_send_email"}
                    self.write(return_data)
            else:
                return_data = {"info": "no_email_exist"}
                self.write(return_data)
