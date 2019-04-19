# Working on Titanic dataset to know the no. of survivour 
# Importing Libraries
import numpy as np # for numerical and scientific computing
import pandas as pd # For data manipulation Or for Data structures and analysis
import matplotlib.pyplot as plt # for visualization and plotting graphs

# Importing train dataset in working environment
train_dataset = pd.read_csv('train.csv')
train_dataset.isnull().sum() # to check the missing values sum in specific attributes
train_dataset.head()## Glimpse throught the data
train_dataset.describe()# Data description
# Slicing test dataset in feature matrix and Vector of prediction
# Only taking those attributes which will effect the survivors 
#i.e. Passenger class, Sex, Age and Embarked 
# Removing the attributes which doesn't have relation with survivors.
# I.e. PassengerID, Name, SibSp(No of siblings), Parch(no of parents),Ticket no., Fare,Cabin
X_train = train_dataset.iloc[:, [2,4,5,11]].values# feature of matrix
y_train = train_dataset.iloc[:, 1].values# Vector of prediction
y_train = y_train.reshape(-1,1)# Reshaping data

# Importing test dataset in working environment
test_dataset = pd.read_csv('test.csv')
test_dataset.isnull().sum()# to check the missing values sum in specific attributes
test_dataset.head()# Glimpse through the data
test_dataset.describe()# Data description
# Slicing test dataset in feature matrix and Vector of prediction
# Only taking those attributes which will effect the survivors 
#i.e. Passenger class, Sex, Age and Embarked columns
# Removing the attributes which doesn't have relation with survivors.
# I.e. PassengerID, Name, SibSp(No of siblings), Parch(no of parents),Ticket no., Fare,Cabin
X_test = test_dataset.iloc[:, [1,3,4,10]].values# feature of matrix

# Importing test dataset(y) in working environment
Prediction_dataset = pd.read_csv('gender_submission.csv')
y_test = Prediction_dataset.iloc[:, 1].values# Vector of prediction
y_test = y_test.reshape(-1,1)# Reshaping data

# Using imputer to fill the missing values in Integer attributes on train dataset
from sklearn.preprocessing import Imputer
im = Imputer()
X_train[:, [0,2]] = im.fit_transform(X_train[:, [0,2]])
y_train = im.fit_transform(y_train)

#Creating a new dataframe for categorical values on train dataset
a_train = pd.DataFrame(X_train[:,[1,3]])
# Counting the no. of values in atributes
a_train[0].value_counts() #male
a_train[1].value_counts() #S
# Filling the missing values with most frequent value 
a_train[0] = a_train[0].fillna('male')
a_train[1] = a_train[1].fillna('S')
# Fitting a_train dataframe back to X_train
X_train[:,[1,3]] = a_train
# Label encoding the Categorical values into Integers
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
# Label encoding Sex column
X_train[:, 1] = lab.fit_transform(X_train[:, 1])
lab.classes_
#Label encoding embarked colum
X_train[:, 3] = lab.fit_transform(X_train[:, 3])
lab.classes_
# Using onehot encoder to avoid Dummy variable trap
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,3])
X_train = one.fit_transform(X_train)
X_train = X_train.toarray()

#Using standard scale to get values in standardize scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
#PLotting graph to get insight
plt.plot(X_train,y_train)
plt.show()
plt.xlabel('feature matrix')
plt.ylabel('Vector of prediction')
plt.title('to know about survived person')
`   

# Using imputer to fill the missing values in Integer attributes on test dataset
from sklearn.preprocessing import Imputer
im = Imputer()
X_test[:,[0,2]] = im.fit_transform(X_test[:,[0,2]])

#Creating a new dataframe for categorical values on test dataset
a_test = pd.DataFrame(X_test[:,[1,3]])
# Counting the no. of values in atributes
a_test[0].value_counts() #male
a_test[1].value_counts() #S
# Filling the missing values with most frequent value 
a_test[0] = a_test[0].fillna('male')
a_test[1] = a_test[1].fillna('S')
# Fitting a_train dataframe back to X_test
X_test[:,[1,3]] = a_test

# Label encoding the Categorical values into Integers
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
# Label encoding Sex column
X_test[:, 1] = lab.fit_transform(X_test[:, 1])
lab.classes_
#Label encoding embarked colum
X_test[:, 3] = lab.fit_transform(X_test[:, 3])
lab.classes_
# Using onehot encoder to avoid Dummy variable trap
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,3])
X_test = one.fit_transform(X_test)
X_test = X_test.toarray()

#Using standard scale to get values in standardize scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_test)

#Using Imputer on y_test
from sklearn.preprocessing import Imputer
im = Imputer()
y_test = im.fit_transform(y_test)

#PLotting graph to get insight
plt.plot(X_test,y_test)
plt.show()
plt.xlabel('feature matrix')
plt.ylabel('Vector of prediction')
plt.title('to know about survived person')

# Using different Algoriths
# 1. Logistic Linear regression 
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()#importing algorithm
# Fitting logistic regression on X_train and y_train
log_reg = log_reg.fit(X_train,y_train)

# Creating y prediction object for Logistic Regression
y_predL = log_reg.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmL = confusion_matrix(y_predL,y_test)

#Finding Score on Train and test dataset
log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)

#2. Using K-Nearest Neigbhours Algorithm
from sklearn.neighbors import KNeighborsClassifier # importing algorithm
KNN = KNeighborsClassifier()
# Fitting logistic regression on X_train and y_train
KNN = KNN.fit(X_train,y_train)

# Creating y prediction object for KNN
y_predK = KNN.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmK = confusion_matrix(y_predK,y_test)

KNN.score(X_train,y_train)
KNN.score(X_test,y_test)

#3. Using Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
# Fitting Decision tree algorithm on X_train and y_train
DTC = DTC.fit(X_train,y_train)

# Creating y prediction object for DTC
y_predD = DTC.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmD = confusion_matrix(y_predD,y_test)
# Finding score on Decesion tree Algorith
DTC.score(X_train,y_train)
DTC.score(X_test,y_test)

#4. Using Naive Bayes algorithm
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
# Fitting Naive Bayes algorithm on X_train and y_train
GNB = GNB.fit(X_train,y_train)

# Creating y prediction object for GBN
y_predG = GNB.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmG = confusion_matrix(y_predG,y_test)
# Finding score on Naive bayes algorithm
GNB.score(X_train,y_train)
GNB.score(X_test,y_test)

#5. SVM Algorithm
from sklearn.svm import SVC
SV = SVC()
# Fitting SVM on X_train and y_train
SV = SV.fit(X_train,y_train)

# Creating y prediction object for SVM
y_predS = SV.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmS = confusion_matrix(y_predS,y_test)
# Finding score on SVM algorithm
SV.score(X_train,y_train)
SV.score(X_test,y_test)

#6. Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
# Fitting Random forest on X_train and y_train
RFC = RFC.fit(X_train,y_train)
# Creating y prediction object for Random forest
y_predR = RFC.predict(X_test)

# creating confusion matrix to check how many predicted values are true
from sklearn.metrics import confusion_matrix
cmR = confusion_matrix(y_predR,y_test)
# Finding score on SVM algorithm
RFC.score(X_train,y_train)
RFC.score(X_test,y_test)


































