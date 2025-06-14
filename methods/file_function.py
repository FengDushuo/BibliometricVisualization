import os

data_root_dir = ""

def dir_dir_list(data_root_dir):
    dir_list = os.listdir(data_root_dir)
    for dir in dir_list:
        if os.path.isfile(os.path.join(data_root_dir,dir)):
            dir_list.remove(dir)
    return dir_list

def get_all_path(dir_root_path):
    rootdir = dir_root_path
    path_list = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        #print(com_path)
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    return path_list

# def all_file_path(file_dir,file_type):   
#     file_list = []   
#     for root, dirs, files in os.walk(file_dir):  
#         for file in files:  
#             if os.path.splitext(file)[1] == '.'+file_type:  
#                 file_list.append(os.path.join(root, file))  
#     return file_list  

def all_file_name(file_dir):
    file_list = os.listdir(file_dir)
    File_list = []
    for file in file_list:
        filename = file.split('.')[0]
        File_list.append(filename)
    return File_list

