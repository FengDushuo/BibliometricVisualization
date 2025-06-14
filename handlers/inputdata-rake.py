import tornado.web
import tornado.escape
import methods.return_file_list as return_file_list
from handlers.base import BaseHandler
import pandas as pd
from flashgeotext.geotext import GeoText
import json
import operator
import time
from collections import Counter
import handlers.config
from multi_rake import Rake
from pyecharts.charts import WordCloud
from pyecharts import options as opts
import re
from datetime import datetime

def read_xlsx_colums(filename):
    # 打开Excel文件
    data = pd.read_excel(filename,sheet_name=0)
    columns = data.columns.tolist()
    return columns

def get_area(input_text):
    geotext = GeoText()
    dic = geotext.extract(input_text=input_text)  # 得到city，country
    # data = json.dumps(dic, indent=4)
    # print(data)	# data是list类型
    country = dic['countries']
    countries_lst = []
    if country:
        country_lst = list(country.values())
        for i in country_lst:
            co = str(Counter(list(i['found_as'])).most_common(1)[0][0])
            countries_lst.append(co)
    else:
        print('没有识别到country信息')
    return countries_lst	# 返回提取到的city，country

# 定义一个函数，该函数接收一个字符串列表并返回一个字典，记录每个字符串和其出现次数
def count_strings_in_list(string_list):
    # 使用字典来记录每个字符串和其出现次数
    count_dict = {}
    # 遍历字符串列表中的每个字符串
    for string in string_list:
        # 如果字符串已经在字典中，增加其计数
        if string in count_dict:
            count_dict[string] += 1
        # 否则，将字符串添加到字典中，并设置计数为1
        else:
            count_dict[string] = 1
    # 返回记录了字符串和出现次数的字典
    return count_dict

def output_all_files(filename,research_label,key_label,research_list,country_list,journal_list,timeline_list,labels,column_name_only_list,output_path):
    # 读取Excel文件
    if "Abstract" in labels:
        add_labels=labels
    else:
        add_labels=labels.append("Abstract")
    df = pd.read_excel(filename, usecols=add_labels, keep_default_na=False)    
    # 将数据转换为字典，并指定orient参数为records，表示每行转换为一个记录
    data_dict = df.to_dict(orient='records')
    #生成研究领域下的 country json
    country_jsons={"Others":[]}
    for item in data_dict:
        flag=0
        for country in country_list:
            #addresses_country = get_area(str(item["Addresses"]))
            if country in str(item["Addresses"]):
                if country != "US":
                    if country != "Taiwan":
                        if country in country_jsons.keys():
                            country_jsons[country].append(item)
                            flag=1
                        else:
                            country_jsons[country] = [item]
                            flag=1
                    else:
                        if "China" in country_jsons.keys():
                            country_jsons["China"].append(item)
                            flag=1
                        else:
                            country_jsons["China"] = [item]
                            flag=1
                else:
                    if "USA" in country_jsons.keys():
                        country_jsons["USA"].append(item)
                        flag=1
                    else:
                        country_jsons["USA"] = [item]
                        flag=1
                    
        if flag == 0:
            country_jsons["Others"].append(item)
    #生成研究领域下的 country 文件
    country_count_list=[]
    for key, value in country_jsons.items(): 
        research_area_json = {"Others":[]}  
        country_outfile = key+".json"
        country_jsonFile = open(output_path + country_outfile, "w+", encoding='utf-8')
        country_js = json.dumps(sorted(country_jsons[key], key=operator.itemgetter(key_label),reverse=True))
        #print(key)
        count_dict={}
        count_dict["country"]=key
        count_dict["infected"]=len(country_jsons[key])
        country_count_list.append(count_dict)
        # print(len(country_jsons[key]))

        country_jsonFile.write(country_js)
        country_jsonFile.close()
        for item_re in country_jsons[key]:
            flag_re=0
            for research in research_list:
                if research in str(item_re[research_label]).replace("&","and"):
                    if research in research_area_json.keys():
                        research_area_json[research].append(item_re)
                        flag_re=1
                    else:
                        research_area_json[research] = [item_re]
                        flag_re=1
            if flag_re == 0:
                research_area_json["Others"].append(item_re)
        for re_key, re_value in research_area_json.items():
            research_outfile = key + "-" + re_key+".json"
            research_jsonFile = open(output_path + research_outfile, "w+", encoding='utf-8')
            research_js = json.dumps(sorted(research_area_json[re_key], key=operator.itemgetter(key_label),reverse=True))
            research_jsonFile.write(research_js)
            research_jsonFile.close()
    count_outfile = "count_country.json"
    count_jsonFile = open(output_path + count_outfile, "w+", encoding='utf-8')
    count_js = json.dumps(sorted(country_count_list, key=operator.itemgetter("infected"),reverse=True))
    count_jsonFile.write(count_js)
    count_jsonFile.close()   
    # # 将字典转换为JSON格式
    json_data = json.dumps(data_dict)
    with open(output_path + 'all_data.json', 'w') as f:
        f.write(json_data)
    f.close()
    #生成研究领域下的 journal json
    journal_jsons={"Others":[]}
    for item in data_dict:
        flag=0
        for journal in journal_list:
            if journal in str(item["Source Title"]).replace("&","and").replace("/","or"):
                journal=journal.replace(":","").replace(".","").replace(",","")[:80]
                if journal in journal_jsons.keys():
                    journal_jsons[journal].append(item)
                    flag=1
                else:
                    journal_jsons[journal] = [item]
                    flag=1
        if flag == 0:
            journal_jsons["Others"].append(item)
    #生成研究领域下的 journal 文件
    journal_count_list=[]
    for key, value in journal_jsons.items(): 
        research_area_json = {"Others":[]}  
        journal_outfile = key.replace("&","and").replace("/","or")+".json"
        journal_jsonFile = open(output_path + journal_outfile, "w+", encoding='utf-8')
        journal_js = json.dumps(sorted(journal_jsons[key], key=operator.itemgetter(key_label),reverse=True))
        #print(key)
        count_dict={}
        count_dict["journal"]=key
        count_dict["infected"]=len(journal_jsons[key])
        journal_count_list.append(count_dict)
        # print(len(journal_jsons[key]))

        journal_jsonFile.write(journal_js)
        journal_jsonFile.close()
        for item_jo in journal_jsons[key]:
            flag_jo=0
            for research in research_list:
                if research in str(item_jo[research_label]).replace("&","and"):
                    if research in research_area_json:
                        research_area_json[research].append(item_jo)
                        flag_jo=1
                    else:
                        research_area_json[research] = [item_jo]
                        flag_jo=1
            if flag_jo == 0:
                research_area_json["Others"].append(item_jo)
        for re_key, re_value in research_area_json.items():
            research_outfile = key.replace("&","and") + "-" + re_key+".json"
            research_jsonFile = open(output_path + research_outfile, "w+", encoding='utf-8')
            research_js = json.dumps(sorted(research_area_json[re_key], key=operator.itemgetter(key_label),reverse=True))
            research_jsonFile.write(research_js)
            research_jsonFile.close()
    count_outfile = "count_journal.json"
    count_jsonFile = open(output_path + count_outfile, "w+", encoding='utf-8')
    count_js = json.dumps(sorted(journal_count_list, key=operator.itemgetter("infected"),reverse=True))
    count_jsonFile.write(count_js)
    count_jsonFile.close()

    #生成研究领域下的 timeline json
    timeline_jsons={"Others":[]}
    timeline_abstract_json={"Others":[]}
    for item in data_dict:
        flag=0
        for timeline in timeline_list:
            if str(timeline) in str(item["Publication Year"]):
                if str(timeline) in timeline_jsons:
                    timeline_jsons[str(timeline)].append(item)
                    timeline_abstract_json[str(timeline)].append(item["Abstract"])
                    flag=1
                else:
                    timeline_jsons[str(timeline)] = [item]
                    timeline_abstract_json[str(timeline)] = [item["Abstract"]]
                    flag=1
        if flag == 0:
            timeline_jsons["Others"].append(item)
            timeline_abstract_json["Others"].append(item["Abstract"])
    #生成研究领域下的 timeline 文件
    timeline_count_list=[]
    for key, value in timeline_jsons.items(): 
        research_area_json = {"Others":[]}  
        research_area_keyword_json = {"Others":[]} 
        timeline_outfile = key+".json"
        timeline_jsonFile = open(output_path + timeline_outfile, "w+", encoding='utf-8')
        timeline_js = json.dumps(sorted(timeline_jsons[key], key=operator.itemgetter(key_label),reverse=True))
        #print(key)
        count_dict={}
        count_dict["timeline"]=key
        count_dict["infected"]=len(timeline_jsons[key])
        timeline_count_list.append(count_dict)
        # print(len(timeline_jsons[key]))
        timeline_jsonFile.write(timeline_js)
        timeline_jsonFile.close()
        for item_jo in timeline_jsons[key]:
            flag_jo=0
            for research in research_list:
                if research in str(item_jo[research_label]).replace("&","and"):
                    if research in research_area_json:
                        research_area_json[research].append(item_jo)
                        research_area_keyword_json[research].append(item_jo["Abstract"])
                        flag_jo=1
                    else:
                        research_area_keyword_json[research]=[item_jo["Abstract"]]
                        research_area_json[research] = [item_jo]
                        flag_jo=1
            if flag_jo == 0:
                research_area_json["Others"].append(item_jo)
                research_area_keyword_json["Others"].append(item_jo["Abstract"])
        for re_key, re_value in research_area_json.items():
            research_outfile = key + "-" + re_key+".json"
            research_jsonFile = open(output_path + research_outfile, "w+", encoding='utf-8')
            research_js = json.dumps(sorted(research_area_json[re_key], key=operator.itemgetter(key_label),reverse=True))
            research_jsonFile.write(research_js)
            research_jsonFile.close()
    count_outfile = "count_timeline.json"
    count_jsonFile = open(output_path + count_outfile, "w+", encoding='utf-8')
    count_js = json.dumps(sorted(timeline_count_list, key=operator.itemgetter("timeline"),reverse=True))
    count_jsonFile.write(count_js)
    count_jsonFile.close()

    #生成词云
    rake = Rake()
    for key, value in timeline_abstract_json.items():
        abstract_list=timeline_abstract_json[key]
        abstract_str=' '.join(abstract_list)        
        # 统计词频
        keywords = rake.apply(abstract_str)
        top30_keywords=keywords[:30]
        # 提取关键词列表中的词语
        # keywords_list = [kw[0] for kw in top20_keywords]
        # 用正则表达式找到每个关键词在原文中出现的次数
        word_frequencies = []
        for keyword in top30_keywords:
            # 使用正则匹配关键词的出现次数，忽略大小写
            # count = len(re.findall(re.escape(keyword), abstract_str))
            word_dict={}
            word_dict["keyword"]=keyword[0]
            word_dict["count"] = keyword[1]
            word_frequencies.append(word_dict)
        # wordcloud_data = [(word, freq) for word, freq in word_frequencies.items()]
        research_area_json = {"Others":[]}  
        timeline_keyword_outfile = key+"-keyword.json"
        timeline_keyword_jsonFile = open(output_path + timeline_keyword_outfile, "w+", encoding='utf-8')
        timeline_keyword_js = json.dumps(word_frequencies)
        timeline_keyword_jsonFile.write(timeline_keyword_js)
        timeline_keyword_jsonFile.close()
        for re_key, re_value in research_area_keyword_json.items():
            abstract_list=research_area_keyword_json[re_key]
            abstract_str=' '.join(abstract_list)        
            # 统计词频
            keywords = rake.apply(abstract_str)
            top30_keywords=keywords[:30]
            # 提取关键词列表中的词语
            # keywords_list = [kw[0] for kw in top20_keywords]
            # 用正则表达式找到每个关键词在原文中出现的次数
            word_frequencies = []
            for keyword in top30_keywords:
                # 使用正则匹配关键词的出现次数，忽略大小写
                # count = len(re.findall(re.escape(keyword), abstract_str))
                word_dict={}
                word_dict["keyword"]=keyword[0]
                word_dict["count"] = keyword[1]
                word_frequencies.append(word_dict)
            research_outfile = key + "-" + re_key+"-keyword.json"
            research_jsonFile = open(output_path + research_outfile, "w+", encoding='utf-8')
            research_js = json.dumps(word_frequencies)
            research_jsonFile.write(research_js)
            research_jsonFile.close()    


class InputdataHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        #username = self.get_argument("user")
        username = tornado.escape.json_decode(self.current_user)
        # user_infos = mrd.select_table(table="users",column="*",condition="username",value=username)
        user_infos=[[99,username,"123456","123456@11.com"]]
        self.render("inputdata.html", user = user_infos[0])

class Inputdata_ipsHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        #username = self.get_argument("user")
        filename = self.get_query_argument("filename")
        csvopenfilepath = "upload/"+filename
        xlsx_colums=read_xlsx_colums(csvopenfilepath)
        username = tornado.escape.json_decode(self.current_user)
        user_infos=[[99,username,"123456","123456@11.com"]]
        handlers.config.global_data_path = filename
        self.render("inputdata_ips.html", user = user_infos[0],xlsx_colums = xlsx_colums,filename=filename)
        

    def post(self):
        username = tornado.escape.json_decode(self.current_user)
        user_infos=[[99,username,"123456","123456@11.com"]]

        #读取用户选择的label
        custom_want = self.request.body_arguments
        custom_want_str=custom_want['text'][0].decode('utf-8')+custom_want['text'][1].decode('utf-8')+custom_want['text'][2].decode('utf-8')+custom_want['text'][3].decode('utf-8')
        #print(custom_want['text'])
        #读取上传的文件
        filename = self.get_query_argument("filename")
        xlsopenfilepath = "upload/"+filename
        #print(xlsopenfilepath)
        data_input = pd.read_excel(xlsopenfilepath,sheet_name=0)
        #用户选择的标签列表
        column_name_only_list = [custom_want['text'][0].decode('utf-8'),custom_want['text'][1].decode('utf-8'),custom_want['text'][2].decode('utf-8'),custom_want['text'][3].decode('utf-8')]
        column_name_list=[custom_want['text'][0].decode('utf-8'),custom_want['text'][1].decode('utf-8'),custom_want['text'][2].decode('utf-8'),custom_want['text'][3].decode('utf-8'),"Article Title","Author Full Names","Since 2013 Usage Count","Addresses","Source Title","Research Areas","Author Keywords","Publication Year"]
        column_name_set=list(set(column_name_list))
        #print(column_name_set)
        #读取地区信息
        column_addresses_list=str(data_input["Addresses"].to_list())
        column_addresses_str = ''.join(column_addresses_list)
        addresses_country = get_area(column_addresses_str)
        # print(addresses_country)
        # print(addresses_country)
        # geotext = GeoText()
        # addresses_country_dict = geotext.extract(input_text=column_addresses_str)['countries']
        # #print(addresses_country_dict)
        # addresses_country_count_dict={}
        # for key, value in addresses_country_dict.items():
        #     addresses_country_count_dict[key]=value['count']
        # addresses_country_sorted_dict = sorted(addresses_country_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        # country_list=[addresses_country_sorted_dict[i][0] for i in range(len(addresses_country_sorted_dict))]
        #print(addresses_country_sorted_dict)
        #读取领域信息
        column_research_list=data_input["Research Areas"].to_list()
        column_research_split_list = [item.split('; ') if isinstance(item, str) else [] for item in column_research_list]
        one_dimensional_list = [item.strip() for sublist in column_research_split_list for item in sublist]
        #print(one_dimensional_list)
        column_research_counts = count_strings_in_list(one_dimensional_list)
        column_research_sorted_dict = sorted(column_research_counts.items(), key=operator.itemgetter(1), reverse=True)
        column_research_sorted_dict_8 = column_research_sorted_dict[:7]
        research_list=[column_research_sorted_dict_8[i][0].replace("&","and") for i in range(7)]
        research_list.append("Others")
        #print(column_research_sorted_dict_8)
        #读取期刊信息
        column_journal_list = data_input["Source Title"].to_list()
        #print(column_journal_list)
        column_journal_counts = count_strings_in_list(column_journal_list)
        column_journal_dict = sorted(column_journal_counts.items(), key=operator.itemgetter(1), reverse=True)
        journal_list = list(set(column_journal_list))
        #print(column_journal_dict)
        journal_dict_list = [column_journal_dict[i][0].replace("&","and").replace("/","or") for i in range(len(column_journal_dict))]

        #读取时间信息
        column_time_list = data_input["Publication Year"].to_list()
        column_time_counts = count_strings_in_list(column_time_list)
        column_time_dict = sorted(column_time_counts.items(), key=operator.itemgetter(0), reverse=False)
        column_time_dict_2 = sorted(column_time_counts.items(), key=operator.itemgetter(1), reverse=True)
        time_list = [column_time_dict[i][0] for i in range(len(column_time_dict))]
        time_list_2 = [column_time_dict_2[i][0] for i in range(len(column_time_dict_2))]
        #print(column_time_dict)
        #新建目录，命名filename+time
        show_data_dir=filename+str(time.time())
        return_file_list.mkdir(show_data_dir,"../static/data")
        output_path= 'static/data/'+ show_data_dir + '/'
        #def output_country_files(filename,research_label,key_label,research_list,country_list,labels,output_path)
        #输出城市文件
        output_all_files(xlsopenfilepath,"Research Areas","Since 2013 Usage Count",research_list,addresses_country,journal_dict_list,time_list,column_name_set,column_name_only_list,output_path)
        handlers.config.global_data_output = output_path

        country_filename=addresses_country[0]
        time_filename=time_list_2[0]
        journal_filename=journal_dict_list[0]
        #print(country_filename,time_filename,journal_filename)
        
        handlers.config.global_data_json = json.dumps({
            "data_path":"/"+output_path,
            "input_path":xlsopenfilepath,
            "countries":addresses_country[:40],
            "researches":research_list[:40],
            "journals":journal_dict_list[:40],
            "timelines":time_list,
            "column_name_set":column_name_set,
            "colors_in_circle":["#65B252","#3896ED","#F36EA7","#9454E6","#FF8800","#EB7E6A","#FFD135","#6A53EC"],
            "colors_dataq":["#66CCCC","#FF99CC","#FF9999","#99CC66","#5151A2","#FF9900","#FF6600","#CCFF66"]
        })
        return_file_list.mkdir("history","../static/data")
        history_path= 'static/data/history/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_filename = f"global_data_{timestamp}.json"
        history_filepath = history_path+history_filename
        with open(history_filepath, 'w', encoding='utf-8') as f:
            f.write(handlers.config.global_data_json)


        handlers.config.global_data_labels=column_name_set

        
        self.render("woscountry.html", user = user_infos[0], global_data_json=handlers.config.global_data_json,global_data_labels=handlers.config.global_data_labels)




