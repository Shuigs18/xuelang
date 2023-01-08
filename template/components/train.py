import pandas as pd
import os
import numpy as np
import sys
sys.path.append('components')
from utils.feature_extraction import feature_extraction
from utils.feature_selection import feature_selection
from utils.get_training_data import get_training_data

import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

 
from utils.config import cfg
import json

#在utils文件夹下utils。py文件中修改训练集的路径

#特征提取,处理后的特征保存下来，如果没有值记做nan
print('feature extraction')
feature_extraction()

#特征选择
print('feature selection')
feature_selection()


# 得到训练样本(含特征补全)
print('training')
samples_ok,labels_ok = get_training_data(cfg.training_ok_path_mod,0)
samples_ng,labels_ng= get_training_data(cfg.training_ng_path_mod,1)
samples = samples_ok+samples_ng
labels = labels_ok+labels_ng

# 标准化
# zscore_scaler = preprocessing.StandardScaler()
# print(samples)
# zscore_scaler.fit(samples)
# samples_scaler = zscore_scaler.transform(samples)

X_train,X_test,y_train,y_test = train_test_split(samples,labels,test_size=0.2,random_state=12343)

print(sum(y_train))

# 训练模型
model = xgb.XGBClassifier(max_depth=8,learning_rate=0.1,n_estimators=50,objective='binary:logistic')
# model = xgb.XGBClassifier(max_depth=100,learning_rate=0.01,n_estimators=200,objective='binary:logitraw')

model.fit(X_train,y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)
 
#计算准确率
f1score = f1_score(y_test,y_pred)
print(y_test)
print(y_pred)
print('f1score:%2.f%%'%(f1score*100))

# save_path  = './components/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
pickle.dump(model,open(cfg.save_path,'wb'))
# pickle.dump(zscore_scaler, open('./zscore_scaler'+str(f1score*100)+'.pkl','wb'))
