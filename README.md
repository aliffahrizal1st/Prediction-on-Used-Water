# Prediction-on-Used-Water

This project is used to complement my academic process, especially in conducting research on my undergraduate thesis. In this research process aims to find the predictive value of several input values. This input value is obtained from water use data in Batu City.

The water usage data is divided into several features. In this study, I divided it between 1 and 10. This data also has the nature of historical data, with a time span of 1 month for each existing data.
## 1. Import Libraries
These are some of the libraries we will use later in this project.
```
import pandas as pd
import numpy as np
import random as ra
from google.colab import files, drive
drive.mount('/content/gdrive')
```
## 2. Initialization
First, we will initialize some parameters which are the weights and biases of the model. As for additional note the limit value we set in this parameter is between -0.5 and 0.5.
### 2.1 Weight
The weighting of the model uses a matrix form with the size of INxHN (Input Neuron x Hidden Neuron). And for this model because we are using ELM, we only need one matrix (Between Input Layer & Hidden Layer) and for the next we will determine the weight between Hidden Layer & Output Layer.
```
def weight(n_input,n_hidden):
  hidden = []
  for i in range(n_hidden):
    hidden.append([])
    for j in range(n_input):
      rand = ra.uniform(-0.5,0.5)
      hidden[i].append(rand)
  return hidden
```
### 2.2 Bias
Close with the weight, for the bias we only need one matrix with the size of 1xHN (The number of layer x The number of Hidden  Neuron).
```
def bias(n_hidden):
  hidden = []
  for i in range(n_hidden):
    rand = ra.uniform(-0.5,0.5)
    hidden.append(rand)
  return hidden
```
## 3. Normalization
Next is normalization. In this process, we will create all data values with the same length variance by using MinMax Normalization to make the values easier to process.
```
def normalisasi(data):
  data = data.astype(float)
  dataMax = data[0][0]
  dataMin = data[0][0]
  for i in data:
    if (dataMax < np.max(i)):
      dataMax = np.max(i)
    elif (dataMin > np.min(i)):
      dataMin = np.min(i)
  for i in range(len(data)):
    for j in range(len(data[i])):
      data[i][j] = (data[i][j] - dataMin)/(dataMax - dataMin)
  return data,dataMin,dataMax
```
## 4. Splitting Dataset
This method is used to separate the datasets after normalization. The result of this process is that we will use the first n% for the training process and the rest for the testing process.
```
def splittingData(data,split):
  split = int(split * len(data))
  train, test = data[:split], data[split:]
  return train,test
```
## 5. Sigmoid Biner
Sigmoid Biner method is used to help us to decides which value to pass as output and what not to pass especially in Hidden Neuron. 
```
def sigmoid(in_hidden):
  for i in range(len(in_hidden)):
    for j in range(len(in_hidden[i])):
      in_hidden[i][j] = 1.0/(1.0 + np.exp(-in_hidden[i][j]))
  return in_hidden
```
## 6. Moore Penrose
For Moore Penrose method is used for calculating the output after Sigmoid Biner method pass the value that we are going to use.
```
def moorePenrose(out_hidden):
  hPlus = np.dot(np.linalg.inv(np.dot(out_hidden.T,out_hidden)),out_hidden.T)
  return hPlus
```
## 7. Output Weight (Weight Between Hidden Layer & Outpu Layer)
For this method we will calculate the weight of the output layer using the results of Moore Penrose's calculations and target training data.
```
def outputWeight(hPlus,y_train):
  out_weight = np.dot(hPlus,y_train)
  return out_weight
```
## 8. Output From Hidden Layer
After several mathematical functions in this model are determined, then we will package them into one method so that it is more efficient to use.
```
def outputHidden(x_train,tw_hidden,b_hidden):
  in_hidden = np.dot(x_train,tw_hidden)
  for i in range(len(in_hidden)):
    for j in range(len(in_hidden[0])):
      in_hidden[i][j]= in_hidden[i][j] + b_hidden[j]
  out_hidden = sigmoid(in_hidden)
  return in_hidden,out_hidden
```
## 9. Training & Testing Process
After all parts from ELM is finished, then we will begin training & testing process.
### 9.1 Training
```
def training(x_train,y_train,w_hidden,b_hidden):
  in_hidden,out_hidden = outputHidden(x_train,np.transpose(w_hidden),b_hidden)
  hPlus = moorePenrose(out_hidden)
  out_weight = outputWeight(hPlus,y_train)
  return out_weight
```
### 9.2 Testing
```
def testing(x_test,y_test,w_hidden,b_hidden,out_weight):
  in_hidden,out_hidden = outputHidden(x_test,np.transpose(w_hidden),b_hidden)
  predict = np.dot(out_hidden,out_weight)
  return predict
```
## 10. Denormalization
Denormalization is used to return the value that we have normalized before, so that the actual result of the model is obtained.
```
def denormalisasi(data,minimum,maksimum):
  if (isinstance(data[0], list) == False) :
      data = [[i] for i in data]
  hasil = []
  for i in range(len(data)):
    value = (data[i][0]*(maksimum-minimum))+minimum
    hasil.append(value)
  return hasil
```
## 11. Evaluation (Root Means Squared Error)
And for the last process, we will evaluate the model from the results we obtained earlier. This evaluation process uses the calculation of Root Means Squared Error (RMSE) to provide a value for the gap between the results and the actual target.
```
def evaluasi(aktual,prediksi):
  hasil = 0
  for i in range(len(aktual)):
    hasil += (aktual[i]-prediksi[i])**2
  hasil = (hasil/len(prediksi))**.5
  return hasil
```
