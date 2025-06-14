#!/usr/bin/env python
# coding=utf-8
import os
import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from handlers.index import IndexHandler    
from handlers.user import UserHandler
# from handlers.forgetpwd import SendEmailHandler
# from handlers.modify import ModifyHandler
# from handlers.register import RegisterHandler
from handlers.wosfunction import WosfunctionHandler
from handlers.woscountry import WoscountryHandler
from handlers.wosjournal import WosjournalHandler
from handlers.wostimeline import WostimelineHandler
from handlers.woskeyword import WoskeywordHandler
from handlers.inputdata import InputdataHandler,Inputdata_ipsHandler
from handlers.inputdata_operate import UploadFileHandler
from handlers.knowledge import KnowledgeHandler
from handlers.vosviewer import VosviewerHandler
from handlers.woshistory import WoshistoryHandler

define("port", default = 8000, help = "run on the given port", type = int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', IndexHandler),          #index既是登录页面也是首页面
            (r'/user', UserHandler),
            (r"/upload_file",UploadFileHandler),
            (r"/inputdata",InputdataHandler),
            (r"/inputdata_ips",Inputdata_ipsHandler),
            (r"/vosviewer",VosviewerHandler),
            (r"/wosfunction",WosfunctionHandler),
            (r"/woscountry",WoscountryHandler),
            (r"/woskeyword",WoskeywordHandler),
            (r"/wostimeline",WostimelineHandler),
            (r"/woshistory",WoshistoryHandler),
            (r"/wosjournal",WosjournalHandler),
            (r"/knowledge",KnowledgeHandler),
        ]
        settings = dict(           
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret = "bZJc2sWbQLKos6GkHn/VB9oXwQt8S0R0kRvJ5/xJ89E=",
            xsrf_cookies = False,
            login_url = '/',
        )
        tornado.web.Application.__init__(self, handlers, **settings)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)

    print("Development server is running at http://localhost:%s" % options.port)
    print("Quit the server with Control-C")

    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()