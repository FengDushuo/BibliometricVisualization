import tornado.web
import tornado.escape
import methods.readdb as mrd
import methods.writedb as mwd
from handlers.base import BaseHandler
from handlers.forgetpwd import *

class ModifyHandler(BaseHandler):
    def get(self):
        # token = self.get_argument("token",default=None)
        # for user in USER_INFO.values():
        #     if User_Info["token"] == token:
        self.render("modify.html")

    def post(self):
        username = self.get_argument("username",default=None)
        password = self.get_argument("password",default=None)
        repassword = self.get_argument("repassword",default=None)
        email = self.get_argument("email",default=None)
        user_infos = mrd.select_table(table="users",column="*",condition="username",value=username)
        return_data = {}
        if username=="" or email=="" or password=="" or repassword=="":
            return_data = {"info":"do_not_full!"}
            self.write(return_data)
        else:
            if username in user_infos[0]:
                if email !=user_infos[0][3]:
                    return_data = {"info":"username_email_unfit!"}
                    self.write(return_data)
                else: 
                    if repassword !=password:
                        return_data = {"info":"repassword_is_false!"}
                        self.write(return_data)
                    else:
                        return_data = {"info":"success!"}
                        mwd.updateinfo("users",username,password,email)
                        self.write(return_data)
            else:
                return_data = {"info":"username_not_exist!"}
                self.write(return_data)
                   