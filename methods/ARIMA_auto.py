import numpy as np
import pandas as pd
import matplotlib as plot
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf  #自相关图
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf    #偏自相关图
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import itertools
import seaborn as sns
import pmdarima as pm


plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False

def ARIMA(file):
    # Data loading
    df = pd.read_excel(file)
    data = df.copy()

    # Aggregating data by 'Publication Year' and 'Times Cited'
    count_by_years = data['Publication Year'].value_counts().sort_index()
    times_cited_by_year = data.groupby('Publication Year')['Since 2013 Usage Count'].sum()

    # Creating a continuous range of years to handle missing values
    full_year_range = pd.Index(range(data['Publication Year'].min(), data['Publication Year'].max() + 1))

    # Reindex both time series to fill missing years with 0
    count_by_years = count_by_years.reindex(full_year_range, fill_value=0)[:-3]
    times_cited_by_year = times_cited_by_year.reindex(full_year_range, fill_value=0)[:-5]

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
    count_by_years.plot(ax=ax, label='Training Data')
    PredicValue.plot(ax=ax, label='Predictions')
    plt.title("Count Training and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)

    ax2 = fig.add_subplot(1,2,2)
    times_cited_by_year.plot(ax=ax2, label='Training Data')
    PredicValue2.plot(ax=ax2, label='Predictions')
    plt.title("Times Cited Training and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.savefig('./output_data/data-time-predict.jpg', dpi = 600)
    plt.show()




ARIMA("D:/a_work/1-phD/project/3-bibliometric/data-sports-injury-medline-26307-20241115/sports-injury-medline-22866-20241115.xlsx")