import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
import math
import numpy as np

features_train = []
labels_train = []
features_test = []
labels_test = []

def test(ff):
    print ff
    return ff

#df = pd.read_csv("test/CHK.csv", parse_dates=['Date'])
#df = pd.read_csv("test/NOK.csv", parse_dates=['Date'])
#df = pd.read_csv("test/UAA.csv", parse_dates=['Date'])
df = pd.read_csv("test/VTBR.ME.csv", parse_dates=['Date'])
df = df.dropna(how='any')
#, usecols=['Date', 'Open', 'Close', 'Volume']
#df = df.tail(12)

N = 1

df["Change"] = df["Open"]-df["Close"]
df["Up"] = df["Change"]-0.001 > 0
df["Up"] = df["Up"].astype(float)
df['Weekday'] = df['Date'].dt.weekday
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
#df = df.drop('Date', 1)


#'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Up' / 'Weekday', 'Month', 'Day'

#['Prev_Open', 'Prev_High', 'Prev_Low', 'Prev_Close', 'Prev_Volume', 'Prev_Change', 'Prev_Up', 'Weekday', 'Month', 'Day', 'Open'] = "Up"



#print parse("2012-02-02").weekday()

df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Up', 'Weekday', 'Month', 'Day']]
print df
matrix = df.as_matrix()
lenmatrix = len(matrix)
real_prediction = []

for i in range(1, len(matrix)):
    item = matrix[i-1]
    item[7:10] = matrix[i][7:10]
    item = np.append(item, matrix[i][0])
    print item[0]
    if i < len(matrix) - N:
        features_train.append(item)
        labels_train.append(matrix[i][6])
    elif i != len(matrix)-1:
        features_test.append(item)
        labels_test.append(matrix[i][6])
    else:
        real_prediction = [item]




#print len(features_test)
#print features_test


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(features_train, labels_train)

if N > 1:
    length = len(features_train)
    prediction = clf.predict(features_test)
    print "length_train: {}".format(len(features_train))
    print "length_test: {}".format(len(features_test))
    print prediction
    print labels_test
    accuracy = clf.score(features_test, labels_test)
    print "accuracy: {}".format(accuracy)

print 'UP today??'
print 'Data: {}'.format(real_prediction)
predict = clf.predict(real_prediction)
print 'Prediction: {}'.format(predict)

