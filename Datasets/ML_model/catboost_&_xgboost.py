# -*- coding: utf-8 -*-
"""catboost_&_XGBoost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1606pFhDhViYIdKyjxFb8yuuPMM6a0Wll
"""

import numpy as np
import pandas as pd
!pip install catboost

import time
t1 = time.perf_counter()

from google.colab import drive
drive.mount('/content/gdrive')

df=pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/ArchitectureAgnostic_Kepler_data_original3.csv')
print(df.columns)

#df1=df.drop(['Sr No.','Benchmark','Kernel Name','C/M','Dwarf Type'],axis=1)
df1=df.drop(['Sr No.','Benchmark','Kernel Name'],axis=1)
#df1=df1.dropna(axis=0,how='any')
#df1=df1.drop(['No_of_cmpinst(PTX Analysis)'
 #           ,'Compute Instructions (simulation)','Compute Latency','No_of_globinst','Gobal Instructions (simulation)','Global Latency',
  #         'No_of_sharinst','Shared Latency','Miscelleneous ','Miscelleneous Instructions (simulation)','No_of_blocks(Grid Size)'
   #         ,'Threads_pbl (Block Size)','Shared Instructions (simulation)','Miscelleneous Latency','Total Instructions','Warp Schedulers issue cycles'
    #        ,'Warp Schedulers issue cycles','Number of SMs activated','Global Load','Global Store','Branch','Number of Waves'],axis=1)

#df1=df1.drop(['Shared Instructions (simulation)','Number of SMs activated','Number of Waves','Compute Instructions (simulation)','Gobal Instructions (simulation)','Total Instructions','Total Threads','Miscelleneous Latency','Miscelleneous Instructions (simulation)','Average Miscelleneous Latency','Average Shared Latency','Average Compute Latency','Average Global Latency'],axis=1)
#df1=df1[df1['avg_shar_lat']!='#DIV/0!']
#df1=df1[df1['avg_shar_lat']!=' ']
#df1=df1[df1['avg_glob_lat']!='#DIV/0!']
#df1=df1[df1['avg_misc_lat']!='#DIV/0!']
#df1=df1[df1['avg_comp_lat']!='#DIV/0!']

#df1=df1[df1['avg_shar_lat']!='#REF!']

#df1=df1[df1['Average Shared Latency']!='#DIV/0!']
#df1=df1[df1['Average Shared Latency']!=' ']
#df1=df1[df1['Average Global Latency']!='#DIV/0!']
#df1=df1[df1['Average Miscelleneous Latency']!='#DIV/0!']
#df1=df1[df1['Average Compute Latency']!='#DIV/0!']

#df1=df1[df1['Average Shared Latency']!='#REF!']
df1

y_train=df1['Power']
y_train=np.array(y_train).astype('float32')
y_train.shape

df1

x_train

x_train=df1.iloc[:,0:9]
x_train=np.array(x_train).astype('float32')
x_train

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(x_train)
print(X_scale.shape)

##splitting the data for training and validation test
from sklearn.utils import shuffle
X_scale,y_train = shuffle(X_scale, y_train, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, y_train, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from catboost import CatBoostRegressor
regr = CatBoostRegressor()
params = {'iterations': [200,300,400,500], 'depth': [6,7,8,9], 'loss_function': ['RMSE', 'MAE'],  'random_seed': [7],           'logging_level': ['Silent']         }

from xgboost import XGBRegressor
regr = XGBRegressor()

#from sklearn.model_selection import GridSearchCV
#CV_rfc = GridSearchCV(estimator=regr, param_grid=params, cv= 5)
#CV_rfc.fit(X_scale,y_train)

#CV_rfc.best_params_

#rfc1 = CatBoostRegressor(depth=8, iterations=400, loss_function='RMSE', logging_level='Silent' ,random_seed=7)
rfc1=XGBRegressor(colsample_bytree=0.7,learning_rate=0.05,n_estimators=450,max_depth=9,min_child_weight=3,silent=1,subsample=0.7 )

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rfc1, X_scale, y_train, cv=5)
print(scores)

rfc1.fit(X_scale, y_train)
rfc1.feature_importances_

print(scores.mean())

t2 = time.perf_counter()
print('time taken to run:',t2-t1)