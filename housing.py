import pandas as pd
import numpy as np
import pickle

df=pd.read_excel('d:/dataset/housing.xlsx')
df.head()
df=df.astype(int)

X=df.drop(columns=['Price']).values
y=df['Price'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/5,random_state=0)

from sklearn.linear_model import LinearRegression
lrm = LinearRegression()
lrm.fit(X_train,y_train)

y_pred = lrm.predict(X_test)
print(y_pred)

from sklearn import metrics

R2=metrics.r2_score(y_test,y_pred)
MSE=metrics.mean_squared_error(y_test,y_pred)
MAE=metrics.mean_absolute_error(y_test,y_pred)
RMSE=np.sqrt(metrics.mean_squared_error(y_test,y_pred))

print("R2 = ",R2)
print("MSE = ",MSE)
print("MAE =",MAE)
print("RMSE =",RMSE)

pickle.dump(lrm, open("housing.pkl", "wb"))