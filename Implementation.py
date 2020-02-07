import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
import geopandas as gpd
import os
import folium
from folium.plugins import MarkerCluster
from folium import plugins
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
stats = pd.read_csv('C:\\Users\\jayma\\Desktop\\614Project\\airbnb.csv')
stats.head()

features = ['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']

onehotdf= pd.DataFrame(stats)

train = onehotdf[features]
categorical = train.select_dtypes(exclude = 'number')
numbers = train.select_dtypes(include= 'number')
onehotcategorical = pd.get_dummies(categorical)

train1 = pd.concat([onehotcategorical, numbers], sort = True, axis = 1)

y = train1['price']
x = train1.drop(columns = ['price'], axis=1)
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
from sklearn import model_selection
x_train, x_test , y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=1)
####################################
# Random Forest Regressor
####################################

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100,max_depth = 8)
rf_model = rf_model.fit(x_train,y_train)

y_predictions = rf_model.predict(x_test)
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error

RMSE = sqrt(mean_squared_error(y_test, y_predictions))

print(RMSE)
####################################
# Decision Tree Regressor
####################################

from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor()
dt_model = dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)
RMSE1 = sqrt(mean_squared_error(y_test, y_pred))

print(RMSE1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
lm = LinearRegression()
lm = lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)
y_test = np.array(y_test)
RMSE2 = np.sum((y_test - y_predict)**2)/(y_test.shape[0])
            
print(RMSE2)
#####################################
        # XGBoost or Gradient boosting
#####################################
from sklearn.ensemble import GradientBoostingRegressor

Gb_model = GradientBoostingRegressor(n_estimators = 50, max_depth = 6)
Gb_model = Gb_model.fit(x_train, y_train)
y_pre = Gb_model.predict(x_test)
RMSEXG = sqrt(mean_squared_error(y_test, y_pre))

print(RMSEXG)