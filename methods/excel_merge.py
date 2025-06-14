

import os,time
import pandas as pd
import xlsxwriter
start_time=time.time()
dir = r'D:/a_work/1-phD\project/7-bibliometric/data-sports-injury-medline-26307-20241115' #设置工作路径

#新建列表，存放每个文件数据框（每一个excel读取后存放在数据框,依次读取多个相同结构的Excel文件并创建DataFrame）
DFs = []

for root, dirs, files in os.walk(dir):  #第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    for file in files:
        file_path=os.path.join(root,file)  #将路径名和文件名组合成一个完整路径
        df = pd.read_excel(file_path) #excel转换成DataFrame
        DFs.append(df)
 #合并所有数据，将多个DataFrame合并为一个
alldata = pd.concat(DFs)  #sort='False'

# alldata.to_csv(r'D:\python脚本\csv合并结果.csv',sep=',',index = False,encoding="gbk")
alldata.to_excel(r'D:/a_work/1-phD\project/7-bibliometric/data-sports-injury-medline-26307-20241115/sports-injury-medline-26307-20241115.xlsx',index = False,engine='xlsxwriter')
end_time=time.time()
times=round(end_time-start_time,2)
print('合并完成，耗时{}秒'.format(times))
