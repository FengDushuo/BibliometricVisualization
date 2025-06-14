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


plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False

def ARIMA(file,outname):
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
    times_cited_by_year = times_cited_by_year.reindex(full_year_range, fill_value=0)[:-3]

    # Differencing the data
    diff1_count_by_years = count_by_years.diff(1).iloc[1:].fillna(0)
    diff2_count_by_years = diff1_count_by_years.diff(1).iloc[1:].fillna(0)
    diff1_times_cited_by_year = times_cited_by_year.diff(1).iloc[1:].fillna(0)
    diff2_times_cited_by_year = diff1_times_cited_by_year.diff(1).iloc[1:].fillna(0)

    # ADF Test for stationarity
    adf_count_by_years = ADF(count_by_years.tolist())
    adf_diff1_count_by_years = ADF(diff1_count_by_years.tolist())
    adf_diff2_count_by_years = ADF(diff2_count_by_years.tolist())
    adf_times_cited_by_year = ADF(times_cited_by_year.tolist())
    adf_diff1_times_cited_by_year = ADF(diff1_times_cited_by_year.tolist())
    adf_diff2_times_cited_by_year = ADF(diff2_times_cited_by_year.tolist())

    # Printing ADF test results
    print('ADF test for original count by years: ', adf_count_by_years)
    print('ADF test for 1st diff count by years: ', adf_diff1_count_by_years)
    print('ADF test for 2nd diff count by years: ', adf_diff2_count_by_years)

    # Splitting dataset (85% train, 15% test)
    train_num = int(count_by_years.shape[0] * 0.85)
    count_train = count_by_years.iloc[:train_num]
    count_test = count_by_years.iloc[train_num:]
    times_cited_train = times_cited_by_year.iloc[:train_num]
    times_cited_test = times_cited_by_year.iloc[train_num:]
    #查看训练集的时间序列与数据(只包含训练集)
    plt.figure(figsize=(16,6))

    plt.subplot(2,3,1)
    plt.plot(count_by_years, 'black')
    plt.title("Paper Count by Publication Year", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('Paper Count', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()

    plt.subplot(2,3,2)
    plt.plot(diff1_count_by_years, 'black')
    plt.title("1st order difference of Paper Count", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('1 diff', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()

    plt.subplot(2,3,3)
    plt.plot(diff2_count_by_years, 'black')
    plt.title("2nd order difference of Paper Count", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('2 diff', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()

    plt.subplot(2,3,4)
    plt.plot(times_cited_by_year, 'black')
    plt.title("Times Cited by Publication Year", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('Times Cited', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()

    plt.subplot(2,3,5)
    plt.plot(diff1_times_cited_by_year, 'black')
    plt.title("1st order difference of Times Cited", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('1 diff', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()

    plt.subplot(2,3,6)
    plt.plot(diff2_times_cited_by_year, 'black')
    plt.title("2nd order difference of Times Cited", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('2 diff', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('Publication Year', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    #plt.legend(prop={'family' : 'Times New Roman', 'size'   : 26})
    # plt.savefig('./stationClocks/' + station + '.ps', dpi = 200)
    plt.tight_layout()
    plt.savefig('./output_data/' + outname + '-data-diff.jpg', dpi = 600)
    plt.show()
    
    #绘制
    fig = plt.figure(figsize=(16,12))

    ax1 = fig.add_subplot(221)
    fig = sm.graphics.tsa.plot_acf(count_by_years, lags=20,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
    plt.title("Count Autocorrelation", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()

    ax2 = fig.add_subplot(222)
    fig = sm.graphics.tsa.plot_pacf(count_by_years, lags=14, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    plt.title("Count Partial Autocorrelation", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()

    ax3 = fig.add_subplot(223)
    fig = sm.graphics.tsa.plot_acf(times_cited_by_year, lags=20,ax=ax3)
    ax3.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
    plt.title("Times Cited Autocorrelation", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.ylabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.xlabel('', fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()

    ax4 = fig.add_subplot(224)
    fig = sm.graphics.tsa.plot_pacf(times_cited_by_year, lags=14, ax=ax4)
    ax4.xaxis.set_ticks_position('bottom')
    plt.title("Times Cited Partial Autocorrelation", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()
    plt.savefig('./output_data/' + outname + '-data-acf.jpg', dpi = 600)
    plt.show()
    

    #确定pq的取值范围
    p_min = 0
    d_min = 1
    q_min = 0
    p_max = 5
    d_max = 1
    q_max = 5

    #Initialize a DataFrame to store the results,，以BIC准则
    results_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    results_aic_2 = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

    for p,d,q in itertools.product(range(p_min,p_max+1),
                                range(d_min,d_max+1),
                                range(q_min,q_max+1)):
        if p==0 and d==0 and q==0:
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            results_aic_2.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
        try:
            model = sm.tsa.ARIMA(count_train, order=(p, d, q))
            results = model.fit()
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic

            model_2 = sm.tsa.ARIMA(times_cited_train, order=(p, d, q))
            results_2 = model_2.fit()
            results_aic_2.loc['AR{}'.format(p), 'MA{}'.format(q)] = results_2.aic
        except:
            continue
    #得到结果后进行浮点型转换
    results_aic = results_aic[results_aic.columns].astype(float)
    results_aic_2 = results_aic_2[results_aic_2.columns].astype(float)
    #绘制热力图
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax1 = sns.heatmap(results_aic,
                    mask=results_aic.isnull(),
                    ax=ax1,
                    annot=True,
                    fmt='.2f',
                    cmap="Purples"
                    )
    plt.title("Count AIC", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()

    ax2 = fig.add_subplot(122)
    ax2 = sns.heatmap(results_aic_2,
                    mask=results_aic_2.isnull(),
                    ax=ax2,
                    annot=True,
                    fmt='.2f',
                    cmap="Purples"
                    )
    plt.title("Times Cited AIC", fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    fig.tight_layout()
    plt.savefig('./output_data/' + outname + '-data-aic.jpg', dpi = 600)
    plt.show()
    

    print(results_aic.stack().idxmin())
    print(results_aic_2.stack().idxmin())

    train_results = sm.tsa.arma_order_select_ic(count_train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)
    train_results_2 = sm.tsa.arma_order_select_ic(times_cited_train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

    print('Count AIC', train_results.aic_min_order)
    print('Count BIC', train_results.bic_min_order)
    print('Times Cited AIC', train_results_2.aic_min_order)
    print('Times Cited BIC', train_results_2.bic_min_order)

    # Model selection using AIC
    d = 1
    d2 = 1              # These can be determined through grid search
    p = train_results.aic_min_order[0]
    q = train_results.aic_min_order[1]
    p2 = train_results_2.aic_min_order[0]
    q2 = train_results_2.aic_min_order[1]

    # ARIMA model fitting
    model = sm.tsa.ARIMA(count_train, order=(p, d, q))
    results = model.fit()
    model2 = sm.tsa.ARIMA(times_cited_train, order=(p2, d2, q2))
    results2 = model2.fit()
    # Residuals
    resid = results.resid
    resid2 = results2.resid
    # Plot ACF of residuals
    fig=plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1,2,1)
    sm.graphics.tsa.plot_acf(resid, lags=15, ax=ax)
    plt.title("Count ACF of residuals" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)

    ax2 = fig.add_subplot(1,2,2)
    sm.graphics.tsa.plot_acf(resid2, lags=15, ax=ax2)
    plt.title("Times Cited ACF of residuals" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.savefig('./output_data/' + outname + '-data-acf-resid.jpg', dpi = 600)
    plt.show()
    # Model prediction
    predict_sunspots = results.predict(dynamic=False)
    print(predict_sunspots)
    predict_sunspots2 = results2.predict(dynamic=False)

    # Plot training data vs predictions
    plt.figure(figsize=(18, 6))
    plt.subplot(1,2,1)
    plt.plot(count_train, label='Training Data')
    plt.plot(predict_sunspots, label='Predictions')
    plt.title("Count Training and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})

    plt.subplot(1,2,2)
    plt.plot(times_cited_train, label='Training Data')
    plt.plot(predict_sunspots2, label='Predictions')
    plt.title("Times Cited Training and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
    plt.savefig('./output_data/' + outname + '-data-train.jpg', dpi = 600)
    plt.show()


    # Forecasting future values for specific years (e.g., 2026–2035)
    # Ensure index is numerical, using the years directly
    last_year = count_by_years.index[-1]
    future_years = pd.Index(range(last_year + 1, last_year + 21))  # Forecast for 10 more years
    last_year2 = times_cited_by_year.index[-1]
    future_years2 = pd.Index(range(last_year2 + 1, last_year2 + 21))
    # Predict future values (forecast 10 values)
    predict_future = results.forecast(steps=20)
    predict_future2 = results2.forecast(steps=20)
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
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})

    ax2 = fig.add_subplot(1,2,2)
    times_cited_by_year.plot(ax=ax2, label='History Data')
    PredicValue2.plot(ax=ax2, label='Predictions')
    plt.title("Times Cited History and Prediction" , fontdict={'family' : 'Times New Roman', 'size'   : 16})
    plt.yticks(fontproperties = 'Times New Roman', size = 16)
    plt.xticks(fontproperties = 'Times New Roman', size = 16)
    plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
    plt.savefig('./output_data/' + outname + '-data-predict.jpg', dpi = 600)
    plt.show()
    

ARIMA("D:/a_work/1-phD/project/3-bibliometric/data-sports-injury-medline-26307-20241115/sports-injury-medline-22866-20241115.xlsx","sports-injury-medline-22866-20241115")