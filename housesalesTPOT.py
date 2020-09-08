import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor


#Reading the data
data = pd.read_csv("kc_house_data.csv")
data.head()

#data transformation
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

# split training/test
X_train , X_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

#modeling
tpot = TPOTRegressor(generations=15, population_size=60, verbosity=3,use_dask=True,warm_start=True)
tpot.fit(X_train, y_train)

#results
print(tpot.score(X_test, y_test))
tpot.export('TpotResults.py')

