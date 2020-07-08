# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:27:07 2020

@author: USER
"""
#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt


#uploading the data
data = pd.read_csv("advertising.csv")
data.head()


#visualizing the data

fig ,axs = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#transforming the data

cols_1 = ['TV']
X= data[cols_1]
y= data.Sales

#IMPORT LINEAR REG ALGO

from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)

res = 6.9748214882298925 + 0.05546477*40
res


#create a dataframe with min and max values

X_new =pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds =lr.predict(X_new)
preds

data.plot(kind='scatter', x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth=3)




import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales~TV' , data =data).fit()
lm.conf_int()

#finding pvalues

lm.pvalues

#finding rsqure value

lm.rsquared

#multi linear regression

cols_1 = ['TV','Radio','Newspaper']
X= data[cols_1]
y= data.Sales

lr= LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales~TV+Radio+Newspaper' , data =data).fit()
lm.conf_int()
lm.summary()