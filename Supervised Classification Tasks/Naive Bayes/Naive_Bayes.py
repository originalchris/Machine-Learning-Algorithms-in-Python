#import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

file = 'C:/Users/Chris/Desktop/iris.csv'
df = pd.read_csv(file)

X = df.loc[:, df.columns != 'species'] #select all except columns named "species"
y = df.loc[:, df.columns == 'species'] #select all columns named "species"

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25) #hold 25% of data as test and use 75% to train

from sklearn.naive_bayes import GaussianNB

#http://scikit-learn.org/stable/modules/naive_bayes.html
nb_model = GaussianNB()
nb_model.fit(X_train,y_train.values.ravel())

y_pred = nb_model.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_pred) #Compares predicted solution to actual solution

print("Accuracy is: " + str(accuracy))