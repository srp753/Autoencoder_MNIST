#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:17:42 2017

@author: snigdha
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import timeit


def sigmoid(x):
    e_x = np.exp(-x)
    return (1 / (1 + e_x))

def tanh(x):
    return np.tanh(x)

# Loading the given train and  validation data
string1=[]
string2=[]

f1 = open("digitstrain.txt", 'r')
for line1 in f1:
     string1.append(line1.strip().split(','))
     
f2 = open("digitsvalid.txt", 'r')
for line2 in f2:
     string2.append(line2.strip().split(','))
     
x_train1, y_train1 = [], []
for subl1 in string1:
    x_train1.append(subl1[:-1])
    y_train1.append([subl1[-1]])

x_valid, y_valid = [], []
for subl2 in string2:
    x_valid.append(subl2[:-1])
    y_valid.append([subl2[-1]])
       
x_train1 = np.float64(x_train1)
y_train1 = np.array(y_train1)
x_valid = np.float64(x_valid)
y_valid = np.array(y_valid)

#shuffling the data randomly in order to 
#train the model better
x_train,y_train = shuffle(x_train1,y_train1, random_state=5)


#Initialization of hidden units and number of random 
#initialization seedsand number of epochs
num_epochs = 500
ini = 1
numhid = 500

soft_op_train = np.zeros((3000,784))
soft_op_valid = np.zeros((1000,784))

cross_entr = np.zeros(3000)
cross_entr_valid = np.zeros(1000)

avg_cr_train = np.zeros((num_epochs,ini))
avg_cr_valid = np.zeros((num_epochs,ini))

avg_cl_train = np.zeros((num_epochs,ini))
avg_cl_valid = np.zeros((num_epochs,ini))

avg_class_train = np.zeros(num_epochs)
avg_cren_train = np.zeros(num_epochs)

avg_class_valid = np.zeros(num_epochs)
avg_cren_valid = np.zeros(num_epochs)

x_in = np.zeros((3000,784))
x_inv = np.zeros((1000,784))

start = timeit.default_timer()
for iter1 in range(0,ini):
    
#Initializing the weights and biases    
    w1 = np.random.normal(0, 0.1,(784,numhid))
    b1= 0
    w2 = np.random.normal(0, 0.1,(numhid,784))
    b2= 0
    learn_rate = 0.1
    dw1 = 0
    dw2 = 0 
    db1 = 0
    db2 = 0
    momentum = 0.5
    for iter2 in range(0,num_epochs):
        for i in range(0,3000):
            
            #Forward pass
            x_input = x_train[i,:]
            xinr = x_input.reshape((784,1))
            
            x_in[i,:]= np.transpose(xinr)
            
            #Implementing 30% dropout 
            mask = np.random.binomial(1,0.7,784).reshape(784,1)
            
            xinr_mask = np.multiply(mask,xinr)
    
            a1 = np.matmul(np.transpose(xinr_mask),w1) + b1
            
            h1= sigmoid(a1) 
            h1r = h1.reshape((numhid,1))
    
            a2 = np.matmul(h1,w2) + b2
                
            h2 =sigmoid(a2)
            h2r = h2.reshape((784,1))
            soft_op_train[i,:] = np.transpose(h2r)      

            
            #Back pass
            
            temp4 = np.multiply(h2r,(1-h2r))
            temp5 = np.multiply(temp4,(h2r-xinr))
            deltaw2 = np.transpose(np.matmul(temp5,np.transpose(h1r)))
            deltab2 = np.transpose(temp5)
            
                        
            temp1 = np.multiply(h1r,(1-h1r))
            temp2 = np.matmul(w2,temp5)
            temp3 = np.multiply(temp1,temp2)
            deltaw1 = np.transpose(np.matmul(temp3,np.transpose(xinr_mask)))
            deltab1 = np.transpose(temp3)
            
            w1 = w1 - learn_rate * (deltaw1 + momentum * dw1)
            b1 = b1 - learn_rate * (deltab1 + momentum * db1) 
            
            w2 = w2 - learn_rate * (deltaw2 + momentum * dw2)
            b2 = b2 - learn_rate * (deltab2 + momentum * db2)
            
            dw2 = deltaw2
            db2 = deltab2
            dw1 = deltaw1
            db1 = deltab1
    
#To determine cross entropy error of training set        
        for l in range(0,3000):
            cross_entr[l] = (-1)*np.sum(np.multiply(np.transpose(x_in[l,:]),np.transpose(np.log(soft_op_train[l,:]))))
    
        avg_cr_train[iter2,iter1] = np.sum(cross_entr)/3000


        for j in range(0,1000):
            x_input_v = x_valid[j,:]
            xinr_v = x_input_v.reshape((784,1))
            
            x_inv[j,:] = np.transpose(xinr_v)
    
            a1_v = np.matmul(np.transpose(xinr_v),w1) + b1

            h1_v = sigmoid(a1_v)   
            h1rv = h1_v.reshape((numhid,1))
    
            a2_v = np.matmul(h1_v,w2) + b2
    
        
            h2_v =sigmoid(a2_v)
            h2rv = h2_v.reshape((784,1))
            soft_op_valid[j,:] = np.transpose(h2rv)      
    

        
#To determine cross entropy error of validation set          
        for l in range(0,1000):
            cross_entr_valid[l] = (-1)*np.sum(np.multiply(np.transpose(x_inv[l,:]),np.transpose(np.log(soft_op_valid[l,:]))))
            
        avg_cr_valid[iter2,iter1] = np.sum(cross_entr_valid)/1000 
                      

#averaging errors obtained over all initializations
avg_cren_train = np.mean(avg_cr_train, axis=1)  
avg_cren_valid = np.mean(avg_cr_valid,axis=1) 
  
stop = timeit.default_timer()
print("Time taken",stop-start)
numar = np.arange(0,num_epochs,1)

plt.figure(1)                
plt.plot(numar, avg_cren_train, 'r', label='Avg Cross Entropy Training Error')
plt.plot(numar, avg_cren_valid, 'b', label='Avg Cross Entropy Validation Error')
plt.title('Observation of average cross-entropy error of training,validation and test')
plt.xlabel('Number of epochs')
plt.ylabel('Prediction error')
plt.legend()
plt.show()



    
    