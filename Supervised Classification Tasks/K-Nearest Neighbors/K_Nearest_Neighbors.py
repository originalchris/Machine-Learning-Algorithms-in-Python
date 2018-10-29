#import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

file = 'C:/Users/Chris/Desktop/iris.csv'
df = pd.read_csv(file)

X = df.loc[:, df.columns != 'species'] #select all except columns named "species"
y = df.loc[:, df.columns == 'species'] #select all columns named "species"

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25) #hold 25% of data as test and use 75% to train

from sklearn.neighbors import KNeighborsClassifier

#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn_model = KNeighborsClassifier(n_neighbors=5) #Defining knn model with n=5 neighbors parameter 
knn_model.fit(X_train,y_train.values.ravel()) #Training svm model using X_train as features and y_train as solutions

y_pred=knn_model.predict(X_test) #Applying model to X_test features for y_pred outcomes

accuracy = metrics.accuracy_score(y_test,y_pred) #Compares predicted solution to actual solution

print("Accuracy is: " + str(accuracy))

k_range= range (1,10)
scores=[]
for k in k_range :
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train.values.ravel())
    y_pred = knn_model.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

import matplotlib.pyplot as plt    
plt.plot(k_range,scores) #plotting the "elbow curve" for lowest value k for highest accuracy