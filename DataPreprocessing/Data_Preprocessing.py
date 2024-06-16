#Importing Libraries
import numpy as np #for arrays
import matplotlib.pyplot as plt #To plot graph
import pandas as pd #to preprocess and import data

#Importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values 
# print(X)
y = dataset.iloc[:, -1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
#fit method will connect the imputer to the matrix of features
# i.e. it will find the missing values and compute the average
# Tip:- Include all the numerical values column to find the missing data 
imputer.fit(X[:, 1:3])
#Replacement of the missing salary using transformer
#Update the data
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)

#Encoding categorical data
#Encoding the independent Variable(here Country(string to vector) and leaving Age and salary)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])] , remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

#Encoding Dependent Variable(Here Purchase because it is in string form)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)
#print(y)

#Splitting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
#Note:- Need to apply transform method only to the test set
# because we can't change the feature scaling applied to train
# set
X_test[:,3:] = sc.transform(X_test[:,3:])
print(X_train)
print(X_test)