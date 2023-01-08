import json
from sklearn import preprocessing
import os
import numpy as np
from utils.config import cfg
import random



def get_training_data(mid_path,label):
    sample_list = os.listdir(os.path.join(mid_path,'P1000','Report_P1000_Time'))
    random.shuffle(sample_list)

    with open(cfg.feature_list_path,'r',encoding='utf8') as f:
        feature_list = json.load(f)
        
    station_list = feature_list.keys()
    samples = []
    labels = []
    # for sample_name in sample_list:
    for i in range(min(len(sample_list),80)):
        sample_name = sample_list[i]
        sample = []
        for station in station_list:
            station_path = os.path.join(mid_path,station)
            for sensor in feature_list[station]:
                sample_path = os.path.join(mid_path,sensor,sample_name)
                if not os.path.exists(sample_path):
                    means_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_mean.txt')) # 此处改成缺失值处理方案
                    vars_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_var.txt')) # 此处改成缺失值处理方案
                    sample_onesensor = [np.random.normal(loc=means_[0],scale=vars_[0],size=1),np.random.normal(loc=means_[1],scale=vars_[1],size=1),np.random.normal(loc=means_[2],scale=vars_[2],size=1)]
                else:
                    sample_onesensor = np.load(sample_path)
                    if len(sample_onesensor)<=0:
                        means_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_mean.txt')) # 此处改成缺失值处理方案
                        vars_ = np.loadtxt(os.path.join(cfg.fs_record_path,station,sensor+'_var.txt')) # 此处改成缺失值处理方案
                        sample_onesensor = [np.random.normal(loc=means_[0],scale=vars_[0],size=1),np.random.normal(loc=means_[1],scale=vars_[1],size=1),np.random.normal(loc=means_[2],scale=vars_[2],size=1)]            
                # print(type(sample_onesensor))
                sample.append(np.array(sample_onesensor))
        sample_ = np.array(sample).reshape(-1)
        samples.append(sample_)
        labels.append(label)
    
    return samples,labels