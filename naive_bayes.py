#-------------------------------------------------------------------------
# AUTHOR: Weisheng (Max) Zhang
# FILENAME: naive_bayes.py
# SPECIFICATION: Read the file weather_training.csv and classify each test instance from weather_test.csv
# FOR: CS 5990- Assignment #3
# TIME SPENT: 20 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]
# print(classes)
# -22, -16, -10, -4, 2, 8, 14, 20, 26, 32, 38)
#reading the training data
#--> add your Python code here
df = pd.read_csv('weather_training.csv', sep=',', header=0)
X_training = np.array(df.values)[:,1:-1].astype('f')
y_training = np.array(df.values)[:,-1].astype('f')
# print(X_training)
# print(y_training)
#update the training class values according to the discretization (11 values only)
#--> add your Python code here
closest = -22
for i in range(len(y_training)):
    for j in range(1, len(classes)):
        if abs(y_training[i] - classes[j]) < abs(y_training[i] - closest):
            closest = classes[j]
    y_training[i] = closest
    closest = -22
# print(y_training)
#reading the test data
#--> add your Python code here
tf = pd.read_csv('weather_test.csv', sep=',', header=0)
X_test = np.array(tf.values)[:,1:-1].astype('f')
y_test = np.array(tf.values)[:,-1].astype('f')
# print(y_test)
#update the test class values according to the discretization (11 values only)
#--> add your Python code here
closest = -22
for i in range(len(y_test)):
    for j in range(1, len(classes)):
        if abs(y_test[i] - classes[j]) < abs(y_test[i] - closest):
            closest = classes[j]
    y_test[i] = closest
    closest = -22
# print(y_test)
#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training)

#make the naive_bayes prediction for each test sample and start computing its accuracy
#the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
#to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
#--> add your Python code here

#print the naive_bayes accuracyy
#--> add your Python code here
accuracy = 0
correct = 0
for (x_testSample, y_testSample) in zip(X_test, y_test):
    prediction = clf.predict([x_testSample])
#     print("prediction = ")
#     print(prediction)
#     print("real = ")
#     print(y_testSample)
    difference = 100*(abs(prediction - y_testSample)/y_testSample)
    if difference >= - 15 and difference <= 15:
        correct += 1
        # print("correct = " + str(correct))
# print(correct)
if correct/10 > accuracy:
    accuracy = correct/10
correct = 0
print("naive_bayes accuracy: " + str(accuracy))



