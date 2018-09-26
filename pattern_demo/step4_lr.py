# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:59:56 2018

@author: LingXue
"""
import pandas as pd
import numpy as np

feature_path = r'./data/pairs_features.txt'
data_path = r'./data/train_15.csv'

data = pd.DataFrame()


with open(feature_path,'r',encoding = 'utf-8') as f:
     length = int(f.readline())
     line = f.readline()
     while line:
          temp = pd.Series([0]* length)
          for k in line.split():
               temp[int(k.split(':')[0])] =1
          data = data.append(temp,ignore_index=True)
          line = f.readline()
     f.close()

data_y = pd.read_csv(data_path).pop('res')

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size = 0.3)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit (x_train, y_train)
reg.coef_
LinearRegression_y_pred = reg.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.10f"
      % mean_squared_error(y_test, LinearRegression_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.10f' % r2_score(y_test, LinearRegression_y_pred)) 


from sklearn.linear_model import LogisticRegression

# 选择模型
cls = LogisticRegression()
# 把数据交给模型训练
cls.fit(x_train, y_train)
LogisticRegression_y_pred = cls.predict(x_test)

#print("Coefficients:%s, intercept %s"%(cls.coef_,cls.intercept_))
#print("Residual sum of squares: %.2f"% np.mean((cls.predict(x_test) - y_test) ** 2))
#print('Score: %.2f' % cls.score(x_test, y_test)) 
print("Mean squared error: %.10f"
      % mean_squared_error(y_test, LogisticRegression_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.10f' % r2_score(y_test, LogisticRegression_y_pred))  

