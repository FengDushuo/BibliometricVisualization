import tornado.escape
# import methods.readdb as mrd
from handlers.base import BaseHandler
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options, parse_config_file
from tornado.web import Application, RequestHandler
import os


class UploadFileHandler(BaseHandler):    #继承base.py中的类BaseHandler
    def get(self):
        target_filename = self.get_argument('filename',default=None)  # 获取上传文件的文件名
        task = self.get_argument('task_id',default=None)              # 获取文件的唯一标识符
        chunk = 0                                        # 分片序号
        with open("upload/%s" % target_filename, 'wb') as target_file:  # 创建新文件
            while True:
                try:
                    filename = 'upload/%s%d' % (task, chunk)
                    source_file = open(filename, 'rb')                    # 按序打开每个分片
                    target_file.write(source_file.read())                 # 读取分片内容写入新文件
                    source_file.close()
                except IOError:
                    break
                chunk += 1
                os.remove(filename)                     # 删除该分片，节约空间
        

    def post(self):
        upload_file = self.request.files['file']
        task = self.get_argument('task_id',default=None)          # 获取文件唯一标识符
        chunk = self.get_argument('chunk', 0)        # 获取该分片在所有分片中的序号
        filename = '%s%s' % (task, chunk)            # 构成该分片唯一标识符
        with open("upload/%s" % filename, 'wb') as part_file:  # 保存分片到本地
            part_file.write(upload_file[0]["body"])
                
        

# class IpsoperateHandler(BaseHandler):    #继承base.py中的类BaseHandler
