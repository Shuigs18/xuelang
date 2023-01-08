import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
from utils.config import cfg
import sys

def get_max_mean_min_for_each_sample(ori_path,tar_path):
    try:
        station_list = os.listdir(ori_path)
    except:
        print('Please change the paths of training_ok_path and training_ng_path in components/utils/config.py.')
        sys.exit(1)
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)

    for station in station_list:
        # print(station)
        for sensor in os.listdir(os.path.join(ori_path,station)):
            print(station,sensor)
            sensor_save_path = os.path.join(tar_path,station,sensor)
            if not os.path.exists(sensor_save_path):
                os.makedirs(sensor_save_path)
            for sample_name in tqdm(os.listdir(os.path.join(ori_path,station,sensor))):
                
                sample_path = os.path.join(ori_path,station,sensor,sample_name)
                sample_save_path = os.path.join(sensor_save_path,sample_name.replace('.csv','.npy'))
                if not os.path.exists(sample_save_path):
                    sample_onesensor_csv = pd.read_csv(sample_path)
                    sample_onesensor_array = np.array(sample_onesensor_csv)
                    if len(sample_onesensor_array)==0:
                        sample_ = float('nan')
                    else:
                        sample_ = [sample_onesensor_array.max(),sample_onesensor_array.mean(),sample_onesensor_array.min()]
                        
                    np.save(sample_save_path,sample_)
                    # np.save(sample_save_path,sample_)
            
def feature_extraction():
    training_ok_path = cfg.training_ok_path
    training_ng_path = cfg.training_ng_path
    training_ok_path_mod = cfg.training_ok_path_mod
    training_ng_path_mod = cfg.training_ng_path_mod
    get_max_mean_min_for_each_sample(training_ok_path,training_ok_path_mod)
    get_max_mean_min_for_each_sample(training_ng_path,training_ng_path_mod)