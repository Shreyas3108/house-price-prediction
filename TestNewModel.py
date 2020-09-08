import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator


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


# Average CV score on the training set was:-14900804159.015076
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.001, loss="huber", max_depth=10, max_features=0.6000000000000001, min_samples_leaf=10, min_samples_split=19, n_estimators=100, subsample=0.25)),
    RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=3, min_samples_split=8, n_estimators=100)
)

exported_pipeline.fit(X_train, y_train)

#results #accurancy=0.90930017506408
print(exported_pipeline.score(X_test, y_test))


