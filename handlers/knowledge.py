import tornado.web
import tornado.escape
import os
import handlers.config
import handlers.inputdata
# import methods.readdb as mrd
from handlers.base import BaseHandler
import json
import random
import handlers.config
import pandas as pd
import joblib
import numpy as np
import matplotlib as plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import itertools
import seaborn as sns
import pmdarima as pm
from methods.model_train_windows import run_models,do_predict,loadAbstractFile,getWord2Vec,getSentenceVec
from methods.model_train_bert_linux import SklearnBertClassifier
import nltk
from nltk.corpus import stopwords
from PIL import Image

def read_xlsx_colums(filename):
    # 打开Excel文件
    data = pd.read_excel(filename,sheet_name=0)
    columns = data.columns.tolist()
    return columns

plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False

def ARIMA(file,output,outfilename):
    # Data loading
    df = pd.read_excel(file)
    data = df.copy()
    # Aggregating data by 'Publication Year' and 'Times Cited'
    count_by_years = data['Publication Year'].value_counts().sort_index()
    times_cited_by_year = data.groupby('Publication Year')['Since 2013 Usage Count'].sum()
    # Creating a continuous range of years to handle missing values
    full_year_range = pd.Index(range(data['Publication Year'].min(), data['Publication Year'].max() + 1))
    # Reindex both time series to fill missing years with 0
    count_by_years = count_by_years.reindex(full_year_range, fill_value=0)[:-1]
    times_cited_by_year = times_cited_by_year.reindex(full_year_range, fill_value=0)[:-1]
    # Splitting dataset (85% train, 15% test)
    train_num = int(count_by_years.shape[0] * 0.85)
    count_train = count_by_years.iloc[:train_num]
    count_test = count_by_years.iloc[train_num:]
    times_cited_train = times_cited_by_year.iloc[:train_num]
    times_cited_test = times_cited_by_year.iloc[train_num:]
    model = pm.auto_arima(count_by_years, start_p=1, start_q=1,
                           max_p=8, max_q=8, m=1,
                           start_P=0, seasonal=False,
                           max_d=3, trace=True,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False)
    # forecast = model.predict(10)#预测未来10年的数据
    # print(forecast)
    model2 = pm.auto_arima(times_cited_by_year, start_p=1, start_q=1,
                           max_p=8, max_q=8, m=1,
                           start_P=0, seasonal=False,
                           max_d=3, trace=True,
                           information_criterion='aic',
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False)
    # Forecasting future values for specific years (e.g., 2026–2035)
    # Ensure index is numerical, using the years directly
    last_year = count_by_years.index[-1]
    future_years = pd.Index(range(last_year + 1, last_year + 21))  # Forecast for 10 more years
    last_year2 = times_cited_by_year.index[-1]
    future_years2 = pd.Index(range(last_year2 + 1, last_year2 + 21))
    # Predict future values (forecast 10 values)
    predict_future = model.predict(20)
    predict_future2 = model2.predict(20)
    # Prepare prediction values for visualization
    PredicValue = pd.Series(predict_future, index=future_years)
    PredicValue2 = pd.Series(predict_future2, index=future_years2)
    # Final plot with both training and predicted values
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1,2,1)
    count_by_years.plot(ax=ax, label='History Data')
    PredicValue.plot(ax=ax, label='Predictions')
    plt.title("Count History and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    ax2 = fig.add_subplot(1,2,2)
    times_cited_by_year.plot(ax=ax2, label='History Data')
    PredicValue2.plot(ax=ax2, label='Predictions')
    plt.title("Times Cited History and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.savefig(output + outfilename, dpi = 600)

class KnowledgeHandler(BaseHandler):
    @tornado.web.authenticated
    def get(self):
        filename = handlers.config.global_data_path
        if "upload/" not in filename:
            filename = "upload/"+filename
        csvopenfilepath = filename
        xlsx_colums=read_xlsx_colums(csvopenfilepath)
        #username = self.get_argument("user")
        username = tornado.escape.json_decode(self.current_user)
        # user_infos = mrd.select_table(table="users",column="*",condition="username",value=username)
        user_infos=[[99,username,"123456","123456@11.com"]]
        self.render("knowledge.html", user = user_infos[0],xlsx_colums = xlsx_colums)

    def post(self):
        username = tornado.escape.json_decode(self.current_user)
        input_file = handlers.config.global_data_path
        output_base_path = handlers.config.global_data_output.replace('/static','static')
        # 构建输出路径
        show_predict_file = os.path.join(output_base_path, 'paper-predict.jpg')
        show_algorithm_file = os.path.join(output_base_path, "algorithm_performance.jpg")
        success_json_file = os.path.join(output_base_path, 'success_list.json')
        # 获取当前工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 判断文件是否存在，不存在则执行生成
        if not os.path.exists(show_predict_file):
            ARIMA(input_file, output_base_path, 'paper-predict.jpg')
        if not os.path.exists(show_algorithm_file):
            image = Image.open(os.path.join(current_dir, 'models', 'ROCAUC-performance.jpg'))
            image.save(show_algorithm_file)
        
        if not os.path.exists(success_json_file): 
            all_words, target, dictionary = loadAbstractFile(input_file)
            #训练Word2Vec模型
                
            word2vec_model = getWord2Vec(all_words,current_dir+'/models',"word2vec_model") 
            data = pd.read_excel(input_file)
            data_cy = data.copy()
            comments = data_cy.iloc[-int(0.20*len(data_cy)):,]
            mlp_model = joblib.load(os.path.join(current_dir, 'models', 'mlp_model.pkl')) 
            success_list=[]
            for i in range(len(comments)):
                data = comments.iloc[i]
                X_data = data['Abstract']
                if X_data != ['']:
                    words = nltk.word_tokenize(X_data)
                    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']   #定义标点符号列表
                    cutwords = [word for word in words if word not in interpunctuations]   #去掉标点符号
                    filtered_words = [w for w in cutwords if w not in stopwords.words('english')]  #去掉停用词
                    all_words=[filtered_words]
                    sentences_vector = getSentenceVec(all_words, word2vec_model)       
                    for sentence in sentences_vector:
                        #print(sentence)        
                        res = mlp_model.predict(np.array(sentence).reshape(1, -1))
                        if res==[1]:
                            success_list.append(data)
                            
            # 转换为 DataFrame
            success_df = pd.DataFrame(success_list)
            # 保存为 JSON 文件
            success_df.to_json(handlers.config.global_data_output.replace('/static','static')+'success_list.json', orient='records', force_ascii=False, lines=False, indent=4)     
        else:
            # 如果存在就直接读取 JSON
            success_df = pd.read_json(success_json_file, orient='records')
            success_list = success_df.to_dict(orient='records')
        ret_list=[]
        for item in success_list:
            ret_item =[str(item["Article Title"]),"http://dx.doi.org/"+str(item["DOI"])]
            ret_list.append(ret_item)
        
        user_infos=[[99,username,"123456","123456@11.com"]]
        self.render("knowledge_result.html", user = user_infos[0], show_predict_file = '../'+show_predict_file, show_algorithm_file = '../'+show_algorithm_file,  recommends = ret_list)

