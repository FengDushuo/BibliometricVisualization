# -*- coding: utf-8 -*- 

import os

# 返回file_dir目录下的以file_Suffix为后缀的文件名列表
def files_list(file_dir , file_Suffix):   
    filelist = []
    for root, dirs, files in os.walk(file_dir):  
        dirs = dirs         
        for file in files:  
            if os.path.splitext(file)[1] == '.'+file_Suffix:   
                filelist.append(os.path.join(root, file))  
    return filelist  

#在file_dir目录下新建目录path
def mkdir(path,file_dir): 
    now_path = os.path.dirname(os.path.abspath(__file__))
    to_mkdir = now_path + "/" + file_dir + "/" + path
    isExists = os.path.exists(to_mkdir)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.mkdir(to_mkdir)
        print(to_mkdir + " 创建成功")
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(to_mkdir + " 目录已存在")
        return False

#在file_dir目录下新建目录path
def absolutepath_mkdir(path,file_dir): 
    to_mkdir = file_dir + "\\" + path
    isExists = os.path.exists(to_mkdir)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.mkdir(to_mkdir)
        print(to_mkdir + " 创建成功")
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(to_mkdir + " 目录已存在")
        return False

#列出rootdir文件夹下所有文件(不含子目录)
def list_all_files(rootdir):
    files_list = []
    df_list = os.listdir(rootdir)               
    for i in range(0,len(df_list)):
        path = os.path.join(rootdir,df_list[i])
        if os.path.isfile(path):
            files_list.append(path)
    return files_list  

