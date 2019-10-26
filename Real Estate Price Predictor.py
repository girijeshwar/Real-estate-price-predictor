 # Real Estate Price Predictor
import pandas as pd

housing = pd.read_csv("housing.csv")

housing.info()

housing

housing.shape

housing.describe()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# TRAIN TEST SPLITTING

import numpy as np
def split_train_test(data, test_ratio):
   np.random.seed(42)
   shuffled = np.random.permutation(len(data))
   print(shuffled)
   test_set_size = int(len(data) * test_ratio)
   test_indices = shuffled[:test_set_size]
   train_indices = shuffled[test_set_size:]
   return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

print(f"rows in train set: {len(train_set)}\nrows in test set: {len(test_set)}\n")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"rows in train set: {len(train_set)}\nrows in test set: {len(test_set)}\n")

# # looking for correlation
orr_matrix = housing.corr()

corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize = (14,10))

housing.plot(kind='scatter', x="RM", y="MEDV", alpha=0.9)

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# fitting

train_set_featurs=train_set.drop("MEDV", axis=1)
train_set_labels = train_set["MEDV"].copy()

train_set_labels.shape

# creating a pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


housing_tr = my_pipeline.fit_transform(train_set_featurs)

housing_tr.shape

#try diffrent model for better scoring
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(train_set_featurs, train_set_labels)

df=test_set.drop("MEDV", axis=1)
df.shape

model.predict(df)

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
mse = mean_squared_error(train_set_labels, housing_predictions)
rmse = np.sqrt(mse)

rmse


# ## Cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_tr, train_set_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

rmse_scores


# In[30]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


print_scores(rmse_scores)


# Save model

from joblib import dump, load
dump(model, 'real_estate.joblib') 

 ## testing

from joblib import dump, load
import numpy as np
model = load('real_estate.joblib') 
features = np.array([[5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)
