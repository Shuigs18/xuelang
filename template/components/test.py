import pandas 
import os
import numpy as np
import sys
sys.path.append('components')
from utils.config import cfg
from tqdm import tqdm
import pickle
import json

def inference():
    # 读取、分析数据集(/code/components/data)
    testing_dataset_path = './components/data'
    
    #由于部分样本sensor检测结果的缺失，使用第一个station得到的sample的列表是全的
    sample_list = os.listdir(os.path.join(testing_dataset_path,'P1000','Report_P1000_Time'))

    # 模型预测
    ids = []
    results = []

    # 选择的特征（传感器）列表
    with open('./components/feature_list.json','r',encoding='utf8') as f:
        feature_list = json.load(f)
        station_list = feature_list.keys()

    for sample_name in tqdm(sample_list):
        sample = []
        
        for station in station_list:
            station_path = os.path.join(testing_dataset_path,station)
            for sensor in feature_list[station]:

                sample_path = os.path.join(station_path,sensor,sample_name)
                # 如果当前传感器采样结果缺失，或者采样结果中无值，则利用训练样本在当前传感器下的均值和方差生成特征
                if not os.path.exists(sample_path):
                    means_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_mean.txt')) # 此处改成缺失值处理方案
                    vars_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_var.txt')) # 此处改成缺失值处理方案
                    sample_onesensor_ = np.array([np.random.normal(loc=means_[0],scale=vars_[0],size=1),np.random.normal(loc=means_[1],scale=vars_[1],size=1),np.random.normal(loc=means_[2],scale=vars_[2],size=1)]).reshape(-1)
                else:
                    sample_onesensor_csv = pandas.read_csv(sample_path)
                    if len(sample_onesensor_csv)==0:
                        means_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_mean.txt')) # 此处改成缺失值处理方案
                        vars_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_var.txt')) # 此处改成缺失值处理方案
                        sample_onesensor_ = np.array([np.random.normal(loc=means_[0],scale=vars_[0],size=1),np.random.normal(loc=means_[1],scale=vars_[1],size=1),np.random.normal(loc=means_[2],scale=vars_[2],size=1)]).reshape(-1)   
                    else:
                        sample_onesensor_array = np.array(sample_onesensor_csv)
                        sample_onesensor_ = np.array([sample_onesensor_array.min(),sample_onesensor_array.mean(),sample_onesensor_array.max()]).reshape(-1)
                # 此处可以输入处理样本每个传感器得到的检测结果,比如得到衍生特征等
                sample.append(np.array(sample_onesensor_))

        sample_ = np.array(sample).reshape(1,-1) # 样本特征，50个传感器（提取了每个传感器采样结果中的最小值、均值、最大值，共计150维特征）
        # 归一化
        # scaler = pickle.load(open('./zscore_scaler96.34146341463415.pkl', 'rb'))

        xgb = pickle.load(open(cfg.save_path, 'rb')) 
        res = xgb.predict(sample_)
        
        # 输出预测文件
        ids.append(sample_name)
        results.append(res[0])
    return ids,results

    


