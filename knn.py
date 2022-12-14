#-------------------------------------------------------------------------
# AUTHOR: Weisheng (Max) Zhang
# FILENAME: knn.py
# SPECIFICATION: Read weather_training.csv and estimate temperature for each data point in weather_test.csv
# FOR: CS 5990- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
df = pd.read_csv('weather_training.csv', sep=',', header=0)
X_training = np.array(df.values)[:,1:-1].astype('f')
y_training = np.array(df.values)[:,-1].astype('f')
# print(X_training)
# print(y_training)
tf = pd.read_csv('weather_test.csv', sep=',', header=0)
X_test = np.array(tf.values)[:,1:-1].astype('f')
y_test = np.array(tf.values)[:,-1].astype('f')
# print(X_test)
# print(y_test)
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here

accuracy = 0
bestK = 0
bestP = 1
bestW = 'uniform'
correct = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            # print("for k = " + str(k) + ", p = " + str(p) + " and w = " + w)
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            #fitting the knn to the data
            #--> add your Python code here
            clf = clf.fit(X_training, y_training)
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                # print("prediction = ")
                # print(prediction)
                # print("real = ")
                # print(y_testSample)
                difference = 100*(abs(prediction - y_testSample)/y_testSample)
                if difference >= - 15 and difference <= 15:
                    correct += 1
                    # print("correct = " + str(correct))
            # print(correct)
            if correct/10 > accuracy:
                accuracy = correct/10
                bestK = k
                bestP = p
                bestW = w
            correct = 0
print("Highest KNN accuracy so far: " + str(accuracy) + ", Parameters: k=" + str(bestK) + ", p=" + str(bestP) + ", w=" + bestW)
            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here





