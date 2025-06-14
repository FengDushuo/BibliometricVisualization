import tornado.web
import tornado.escape
# import methods.readdb as mrd
from handlers.base import BaseHandler
import handlers.config
import handlers.inputdata

class WoskeywordHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        #username = self.get_argument("user")
        username = tornado.escape.json_decode(self.current_user)
        # user_infos = mrd.select_table(table="users",column="*",condition="username",value=username)
        user_infos=[[99,username,"123456","123456@11.com"]]
        self.render("woskeyword.html", user = user_infos[0],global_data_json=handlers.config.global_data_json,global_data_labels=handlers.config.global_data_labels)