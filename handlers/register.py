import tornado.escape
import methods.readdb as mrd
import methods.writedb as mwd
from handlers.base import BaseHandler

class RegisterHandler(BaseHandler): 
    def get(self):
        self.render("register.html", type_="register")

    def post(self):
        username = self.get_argument("rusername",default=None)
        password = self.get_argument("rpassword",default=None)
        repassword = self.get_argument("rrepassword",default=None)
        email = self.get_argument("remail",default=None)
        accept = self.get_argument("accept",default=None)
        dbinfo = mrd.select_dics("users","*")
        if dbinfo == []:
            usrs = []
            pwds = []
            emails = []
        else:
            usrs,pwds,emails=zip(*map(lambda x: (x["username"],x["password"],x["email"]),dbinfo))
        return_data = {}
        if username=="" or password=="" or repassword=="" or email=="":
            return_data = {"info":"do_not_full!"}
            self.write(return_data)
        else:
            if usrs == []:
                if accept=="true":
                    mwd.insertinfo("users",username,password,email)
                    return_data = {"info":"success!"}
                    self.write(return_data)
                else:
                    return_data = {"info":"not_checked!"}
                    self.write(return_data)
            elif username in usrs:
                return_data = {"info":"user_is_exist!"}
                self.write(return_data)
            else:
                if email in emails:
                    return_data = {"info":"email_has_been_used!"}
                    self.write(return_data)
                else:
                    if repassword !=password:
                        return_data = {"info":"repassword_is_false!"}
                        self.write(return_data)
                    else:
                        if accept=="true":
                            return_data = {"info":"success!"}
                            mwd.insertinfo("users",username,password,email)
                            self.write(return_data)
                        else:
                            return_data = {"info":"not_checked!"}
                            self.write(return_data)

