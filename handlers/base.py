import tornado.web

class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        return self.get_secure_cookie("user")

    # def set_default_headers(self):
    #     origin_url = self.request.headers.get('Origin')
    #     self.set_header("Access-Control-Allow-Origin", '*')
    #     self.set_header("Access-Control-Allow-Credentials", "true")
    #     self.set_header("Access-Control-Allow-Headers", "x-requested-with,token")
    #     self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    #     self.set_header("Access-Control-Max-Age", 1000)
    #     self.set_header("Content-type", "application/json")

    # def options(self):
    #     # 返回方法1
    #     self.set_status(200) # 这里的状态码一定要设置200，建议
    #     self.finish()