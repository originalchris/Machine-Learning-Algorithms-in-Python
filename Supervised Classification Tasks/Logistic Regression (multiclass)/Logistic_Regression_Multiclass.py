#import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

file = 'C:/Users/Chris/Desktop/iris.csv'
df = pd.read_csv(file)

X = df.loc[:, df.columns != 'species'] #select all except columns named "species"
y = df.loc[:, df.columns == 'species'] #select all columns named "species"

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25) #hold 25% of data as test and use 75% to train

from sklearn.linear_model import LogisticRegression

#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
logit_model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logit_model.fit(X_train,y_train)

y_pred = logit_model.predict(X_test) #Applying model to X_test features for y_pred outcomes

accuracy = metrics.accuracy_score(y_test,y_pred) #Compares predicted solution to actual solution

print("Accuracy is: " + str(accuracy))