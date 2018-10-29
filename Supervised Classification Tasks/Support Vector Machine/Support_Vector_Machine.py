#import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

file = 'C:/Users/Chris/Desktop/iris.csv'
df = pd.read_csv(file)

X = df.loc[:, df.columns != 'species'] #select all except columns named "species"
y = df.loc[:, df.columns == 'species'] #select all columns named "species"

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25) #hold 25% of data as test and use 75% to train

from sklearn import svm

#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svm_model = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train) #Defining model with C penalty = 1 default parameter / kernal the type of alg used: linear, poly, rbf etc
svm_model.fit(X_train,y_train) #Training svm model using X_train as features and y_train as solutions

y_pred = svm_model.predict(X_test) #Applying model to X_test features for y_pred outcomes

accuracy = metrics.accuracy_score(y_test,y_pred) #Compares predicted solution to actual solution

print("Accuracy is: " + str(accuracy))