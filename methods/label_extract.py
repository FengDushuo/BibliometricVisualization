# -*- coding: utf-8 -*-

import os
from methods.return_file_list import files_list
from methods.return_file_list import absolutepath_mkdir
from methods.csv_file_function import column_value
from methods.csv_file_function import is_column_exist
from methods.csv_file_function import column_position
from functools import reduce


def extract_two_labels_not_quchong(openfile,outputfiled,firstlabel,secondlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    twolabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            twolabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val})
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        # output_write.write(firstlabel+"," + secondlabel)
        # output_write.write('\n')
        for item in twolabellist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path

def extract_three_labels_not_quchong(openfile,outputfiled,firstlabel,secondlabel,thirdlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    thirdlabel_name = thirdlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    threelabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            thirdlabel_val = column_value(line,column_name_str,thirdlabel_name).rstrip()
            threelabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val,thirdlabel_name:thirdlabel_val})
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + thirdlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        # output_write.write(firstlabel+"," + secondlabel + "," + thirdlabel)
        # output_write.write('\n')
        for item in threelabellist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write(",")
            output_write.write(item[thirdlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path

#字典列表去重
def list_dict_duplicate_removal(data_list):
    run_function = lambda x, y: x if y in x else x + [y]
    return reduce(run_function, [[], ] + data_list)


def extract_two_labels_quchong(openfile,outputfiled,firstlabel,secondlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    twolabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            twolabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val})
        output_datalist = list_dict_duplicate_removal(twolabellist)
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        # output_write.write(firstlabel+"," + secondlabel)
        # output_write.write('\n')
        for item in output_datalist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path

def extract_three_labels_quchong(openfile,outputfiled,firstlabel,secondlabel,thirdlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    thirdlabel_name = thirdlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    threelabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            thirdlabel_val = column_value(line,column_name_str,thirdlabel_name).rstrip()
            threelabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val,thirdlabel_name:thirdlabel_val})
        output_datalist = list_dict_duplicate_removal(threelabellist)
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + thirdlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        # output_write.write(firstlabel+"," + secondlabel + "," + thirdlabel)
        # output_write.write('\n')
        for item in output_datalist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write(",")
            output_write.write(item[thirdlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path
    

def twolabel_timestamp_reset(openfile,outputfiled,firstlabel,secondlabel): #firstlabel为timestamp
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    twolabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            twolabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val})
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        output_write.write("date"+"," + "value")
        output_write.write('\n')
        for item in twolabellist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path
      

def extract_three_labels_quchong_nolabel(openfile,outputfiled,firstlabel,secondlabel,thirdlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    thirdlabel_name = thirdlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    threelabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")

        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            thirdlabel_val = column_value(line,column_name_str,thirdlabel_name).rstrip()
            threelabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val,thirdlabel_name:thirdlabel_val})
        output_datalist = list_dict_duplicate_removal(threelabellist)
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + thirdlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        for item in output_datalist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write(",")
            output_write.write(item[thirdlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path

def extract_three_labels_not_quchong_with_timestamp(openfile,outputfiled,firstlabel,secondlabel,thirdlabel):
    result_field = outputfiled
    firstlabel_name = firstlabel.replace(" ","")
    secondlabel_name = secondlabel.replace(" ","")
    thirdlabel_name = thirdlabel.replace(" ","")
    file_to_read_path = openfile

    base=os.path.basename(file_to_read_path)        #命名输出文件所在文件夹
    basepath = base.split('.')[0]
    absolutepath_mkdir(basepath,result_field)

    threelabellist = []
    with open(file_to_read_path, 'r' , encoding='gb18030') as f:
        lines =  f.readlines()
        column_name_str = lines[0]
        column_name_str = column_name_str.replace(" ","")
        column_name_str = column_name_str.replace("\n","")
        # firstlabel_position = column_position(column_name_str,firstlabel_name)
        # secondlabel_position = column_position(column_name_str,secondlabel_name)

        for line in lines[1:]:
            firstlabel_val = column_value(line,column_name_str,firstlabel_name).rstrip()
            secondlabel_val = column_value(line,column_name_str,secondlabel_name).rstrip()
            thirdlabel_val = column_value(line,column_name_str,thirdlabel_name).rstrip()
            threelabellist.append({firstlabel_name:firstlabel_val,secondlabel_name:secondlabel_val,thirdlabel_name:thirdlabel_val})
        output_path = result_field + "\\"+ basepath + "\\" + firstlabel_name + secondlabel_name + thirdlabel_name + ".csv"
        output_write = open(output_path, 'a+' , newline="")
        output_write.write("timestamp" + "," + "label1" + "," + "label2")
        output_write.write('\n')
        for item in threelabellist:
            output_write.write(item[firstlabel_name])
            output_write.write(",")
            output_write.write(item[secondlabel_name])
            output_write.write(",")
            output_write.write(item[thirdlabel_name])
            output_write.write('\n')
        output_write.close()
    return output_path


