# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang
"""
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# make some of the repeated tasks functional
def load_data_and_labels(data_path):
    dataset = pd.read_csv(data_path, header=None)
    X = dataset.iloc[:, 0:-2].values
    label_column = dataset.iloc[:, -2].values
    y = []
    for i in range(len(label_column)):
        # under the new standardized labels, category 16 is normal
        if label_column[i] == 16:
            y.append(0)
        else:
            y.append(1)
    # Convert ist to array
    y = np.array(y)
    return X,y

def encode_data(X):
    x_new = deepcopy(X)
    le = LabelEncoder()
    x_new[:, 1] = le.fit_transform(X[:, 1])
    x_new[:, 2] = le.fit_transform(X[:, 2])
    x_new[:, 3] = le.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3])
    x_new = onehotencoder.fit_transform(x_new).toarray()
    return x_new

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np
# Import argparse to parse command line args
from sys import argv
import argparse

parser = argparse.ArgumentParser(description="FNN for NSL-KDD dataset")
parser.add_argument('--traindata', type=str, action='store')
parser.add_argument('--testdata', type=str, action='store')
parser.add_argument('-s', '--scenario', type=str, action='store')
parser.add_argument('-b', '--batch-size', type=int, action='store')
parser.add_argument('-e', '--epoch', type=int, action='store')
clargs = parser.parse_args(argv[1:])
# Variable Setup
try:
    traindata_path, testdata_path = clargs.traindata, clargs.testdata
except AttributeError as e:
    print(e)
    print('Use --traindata and --testdata to include the train and test data')
    raise

BatchSize = clargs.batch_size if clargs.batch_size else 10
NumEpoch = clargs.epoch if clargs.epoch else 10



# Import dataset.
# Dataset is given in TraningData variable You can replace it with the file 
# path such as “C:\Users\...\dataset.csv’. 
# The file can be a .txt as well. 
# If the dataset file has header, then keep header=0 otherwise use header=none
# reference: https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/

train_data_old, train_label = load_data_and_labels(traindata_path)
test_data_old, test_label = load_data_and_labels(testdata_path)

print(f'train_data.shape = {train_data_old.shape}')
print(f'test_data.shape = {test_data_old.shape}')

# Encoding categorical data (convert letters/words in numbers)
# Reference: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# The following code work without warning in Python 3.6 or older. Newer versions suggest to use ColumnTransformer
'''
train_data = encode_data(train_data)
test_data = encode_data(test_data)
'''
# The following code work Python 3.7 or newer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#OneHotEncode all the categorical data properly
protocol_cats_c1 = len(pd.read_csv('./featureMappingsAll/1.txt', header=None))
type_cats_c2 = len(pd.read_csv('./featureMappingsAll/2.txt', header=None))
state_cats_c3 = len(pd.read_csv('./featureMappingsAll/3.txt', header=None))
colsEncoder = OneHotEncoder(categories=[list(range(protocol_cats_c1)), \
    list(range(type_cats_c2)), list(range(state_cats_c3))])
ct = ColumnTransformer(
    [('one_hot_encoder', colsEncoder, [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
train_data = np.array(ct.fit_transform(train_data_old), dtype=np.float)
test_data = np.array(ct.fit_transform(test_data_old), dtype=np.float)

print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape = {test_data.shape}')
# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is 
# MinMaxScaler. 
# referece: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_data = sc.fit_transform(train_data)  # Scaling to the range [0,1]
test_data = sc.fit_transform(test_data)


########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
# Reference: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
# rectified linear unit activation function relu, reference: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(train_data[0])))

# Adding the second hidden layer, 6 nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer, 1 node, 
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN, 
# Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“, Reference: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# add verbose=0 to turn off the progress report during the training
# To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
classifierHistory = classifier.fit(train_data, train_label, batch_size = BatchSize, epochs = NumEpoch)

# evaluate the keras model for the provided model and dataset
loss, accuracy = classifier.evaluate(train_data, train_label)
print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results
y_pred = classifier.predict(test_data)
y_pred = (y_pred > 0.9)   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
# summarize the first 5 cases
#for i in range(5):
#    print('%s => %d (expected %d)' % (test_data[i].tolist(), y_pred[i], test_label[i]))

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label, y_pred)
'''
res = list(map(lambda x: test_label[x]==y_pred[x], range(len(test_label))))
correct = len(list(filter(lambda x: x, res)))
percentage = correct / len(res)
print(percentage)
'''

print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

########################################
# Part 4 - Visualizing
#######################################

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

# Get the filename of train and test fig
scenario = clargs.scenario if clargs.scenario else ''

# save the scenario results.
with open(f"{scenario}_result.txt", 'w') as f:
    f.write(str(cm))
    f.write('\n')
    acc = (cm[0][0]+cm[1][1]) / (sum(cm[0])+sum(cm[1]))
    f.write(f"Accuracy: {acc}")

# You can plot the accuracy
print('Plot the accuracy')
# Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
# use the command python -c 'import keras; print(keras.__version__)' on MAC or Linux to check Keras' version
plt.plot(classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(f'{scenario}_accuracy_sample.png')

# You can plot history for loss
print('Plot the loss')
plt.plot(classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig(f'{scenario}_loss_sample.png')
