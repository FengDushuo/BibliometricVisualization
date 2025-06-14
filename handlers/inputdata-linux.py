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
from keybert import KeyBERT

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
    #rake = Rake()
    # kw_model = KeyBERT(model='all-MiniLM-L6-v2')  # 可选择其他模型
    # for key, value in timeline_abstract_json.items():
    #     abstract_list=timeline_abstract_json[key]
    #     abstract_str=' '.join(abstract_list)        
    #     # 统计词频
    #     #keywords = rake.apply(abstract_str)
    #     #top30_keywords=keywords[:30]
    #     keywords = kw_model.extract_keywords(
    #         abstract_str, 
    #         keyphrase_ngram_range=(1, 3),  # 可选择(1, 3)获取短语
    #         stop_words='english',
    #         use_maxsum=True,  # 更丰富分散性关键词
    #         nr_candidates=60,
    #         top_n=30
    #     )

    #     # 提取关键词列表中的词语
    #     # keywords_list = [kw[0] for kw in top20_keywords]
    #     # 用正则表达式找到每个关键词在原文中出现的次数
    #     word_frequencies = []
    #     # for keyword in top30_keywords:
    #     #     # 使用正则匹配关键词的出现次数，忽略大小写
    #     #     # count = len(re.findall(re.escape(keyword), abstract_str))
    #     #     word_dict={}
    #     #     word_dict["keyword"]=keyword[0]
    #     #     word_dict["count"] = keyword[1]
    #     #     word_frequencies.append(word_dict)
    #     for keyword, score in keywords:
    #         word_dict = {"keyword": keyword, "count": round(score * 100)}  # scale score for word cloud
    #         word_frequencies.append(word_dict)
    #     # wordcloud_data = [(word, freq) for word, freq in word_frequencies.items()]
    #     research_area_json = {"Others":[]}  
    #     timeline_keyword_outfile = key+"-keyword.json"
    #     timeline_keyword_jsonFile = open(output_path + timeline_keyword_outfile, "w+", encoding='utf-8')
    #     timeline_keyword_js = json.dumps(word_frequencies)
    #     timeline_keyword_jsonFile.write(timeline_keyword_js)
    #     timeline_keyword_jsonFile.close()
    #     for re_key, re_value in research_area_keyword_json.items():
    #         abstract_list=research_area_keyword_json[re_key]
    #         abstract_str=' '.join(abstract_list)        
    #         # 统计词频
    #         keywords = kw_model.extract_keywords(abstract_str, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True, nr_candidates=60, top_n=30)

    #         # keywords = rake.apply(abstract_str)
    #         # top30_keywords=keywords[:30]
    #         # 提取关键词列表中的词语
    #         # keywords_list = [kw[0] for kw in top20_keywords]
    #         # 用正则表达式找到每个关键词在原文中出现的次数
    #         word_frequencies = []
    #         for keyword, score in keywords:
    #             word_dict = {"keyword": keyword, "count": round(score * 100)}  # scale score for word cloud
    #             word_frequencies.append(word_dict)
    #         research_outfile = key + "-" + re_key+"-keyword.json"
    #         research_jsonFile = open(output_path + research_outfile, "w+", encoding='utf-8')
    #         research_js = json.dumps(word_frequencies)
    #         research_jsonFile.write(research_js)
    #         research_jsonFile.close()   

    # 使用更轻量模型
    kw_model = KeyBERT(model='paraphrase-MiniLM-L3-v2')

    def extract_keywords_fast(text, model, max_len=3000, top_n=15):
        if not text:
            return []
        text = text.strip()[:max_len]
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_maxsum=False,
            use_mmr=False,
            top_n=top_n
        )
        return [{"keyword": kw, "count": round(score * 100)} for kw, score in keywords]

    # 生成 timeline 关键词
    for timeline_key, abstract_list in timeline_abstract_json.items():
        timeline_abstract_str = ' '.join(abstract_list)
        timeline_keywords = extract_keywords_fast(timeline_abstract_str, kw_model)

        with open(os.path.join(output_path, f"{timeline_key}-keyword.json"), 'w', encoding='utf-8') as f:
            json.dump(timeline_keywords, f)

        # 研究领域下的关键词
        for research_key, abs_list in research_area_keyword_json.items():
            research_abstract_str = ' '.join(abs_list)
            research_keywords = extract_keywords_fast(research_abstract_str, kw_model)

            with open(os.path.join(output_path, f"{timeline_key}-{research_key}-keyword.json"), 'w', encoding='utf-8') as f:
                json.dump(research_keywords, f)
 


def process_wos_excel_file(filepath, label_columns):
    """
    直接处理本地 WoS Excel 文件，执行数据分析和关键词生成。

    :param filepath: Excel 文件路径（.xlsx）
    :param label_columns: 用户选择的 4 个自定义列（列表）
    :return: 输出路径
    """
    import os

    # 读取 Excel
    data_input = pd.read_excel(filepath, sheet_name=0)

    # 全部列（包含固定列）
    column_name_list = label_columns + [
        "Article Title", "Author Full Names", "Since 2013 Usage Count",
        "Addresses", "Source Title", "Research Areas", 
        "Author Keywords", "Publication Year"
    ]
    column_name_set = list(set(column_name_list))

    # 地址处理
    column_addresses_str = ''.join(str(data_input["Addresses"].to_list()))
    addresses_country = get_area(column_addresses_str)

    # 研究领域处理
    research_list = extract_top_research_fields(data_input["Research Areas"], top_n=7)

    # 期刊处理
    journal_list_raw = data_input["Source Title"].to_list()
    journal_list = list(set(journal_list_raw))

    # 时间线处理
    time_list = sorted(data_input["Publication Year"].unique().tolist())

    # 生成输出目录
    show_data_dir = os.path.basename(filepath).replace(".xlsx", "") + "_" + str(int(time.time()))
    output_path = f'{show_data_dir}/'
    os.makedirs(output_path, exist_ok=True)

    # 调用主处理函数（即你已有的）
    output_all_files(
        filename=filepath,
        research_label="Research Areas",
        key_label="Since 2013 Usage Count",
        research_list=research_list,
        country_list=addresses_country,
        journal_list=journal_list,
        timeline_list=time_list,
        labels=column_name_set,
        column_name_only_list=label_columns,
        output_path=output_path
    )

    return output_path

def extract_top_research_fields(research_series, top_n=7):
    split_list = [item.split('; ') if isinstance(item, str) else [] for item in research_series]
    flat_list = [item.strip().replace("&", "and") for sublist in split_list for item in sublist]
    counts = count_strings_in_list(flat_list)
    sorted_fields = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_fields[:top_n]] + ["Others"]


if __name__ == '__main__':
    excel_path = 'sports-injury-medline-22866-20241115.xlsx'  # 替换成你的文件路径
    selected_labels = ['DOI', 'Publication Year', 'Authors', 'Abstract']  # 替换成你的四个字段
    result_path = process_wos_excel_file(excel_path, selected_labels)
    print("数据处理完成，结果输出目录：", result_path)



