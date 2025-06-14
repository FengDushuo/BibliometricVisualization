import tornado.web
import tornado.escape
# import methods.readdb as mrd
from handlers.base import BaseHandler
import handlers.config
import handlers.inputdata
import os
import json

class WoshistoryHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        #username = self.get_argument("user")
        username = tornado.escape.json_decode(self.current_user)
        # user_infos = mrd.select_table(table="users",column="*",condition="username",value=username)
        user_infos=[[99,username,"123456","123456@11.com"]]
        # 假设 history 存储路径为如下路径（可根据 config 设置）
        history_dir = "static/data/history"
        print(history_dir)
        if os.path.exists(history_dir):
            history_files = sorted([
                f for f in os.listdir(history_dir)
                if f.endswith('.json')
            ])
        else:
            history_files = []

        self.render("woshistory.html", user = user_infos[0],global_data_json=handlers.config.global_data_json,global_data_labels=handlers.config.global_data_labels,history_files=history_files)

    @tornado.web.authenticated
    def post(self):
        filename = self.get_argument("filename")
        history_path= 'static/data/history/'+filename

        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            handlers.config.global_data_json = json.dumps(json_data)
            handlers.config.global_data_labels = json_data["column_name_set"]
            handlers.config.global_data_path = json_data["input_path"]
            handlers.config.global_data_output = json_data["data_path"]

            self.write({
                "status": "success",
                "message": "History loaded",
                "redirect": "/woscountry"
            })
        else:
            self.write({
                "status": "error",
                "message": f"File {filename} not found"
            })
