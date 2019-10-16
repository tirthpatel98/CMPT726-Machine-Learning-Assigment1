#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
bias_arr=[True,False]
featureVal=[10,11,12]

for bias in bias_arr:
    weight=[]
    train_err=[]
    test_err=[]
   
    for i in range(7,15):
        targets = values[:,1]
        x = values[:,i]
        
        N_TRAIN = 100
        x_train = x[0:N_TRAIN,:]
        x_test = x[N_TRAIN:,:]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]
       
        
        w,t_err= a1.linear_regression(x_train,t_train,'polynomial', bias,0,3,1)
        pred,te_err= a1.evaluate_regression(x_test,t_test,'polynomial',bias,w,3)
        weight.append(w)
        train_err.append(t_err)
        test_err.append(te_err)    
      
    print("TrainError for bais =",bias,train_err)
    print("TestError for bais =",bias,test_err)
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.8
    
    rects1 = plt.bar(index, train_err, bar_width,alpha=opacity,color='b',label='Training Error')
    
    rects2 = plt.bar(index + bar_width, test_err, bar_width,alpha=opacity,color='g',label='Testing Error')
    
    plt.xlabel('Features')
    plt.ylabel('Error')
    if(bias==False):
        text='Single feature error without bias'
    if(bias==True):
        text='Single feature error with bias'
    plt.title(text)
    plt.xticks(index + bar_width, ('8', '9', '10', '11', '12', '13', '14', '15'))
    plt.legend()
    plt.tight_layout()
    plt.show()
    #input()
for bias in bias_arr:    
    for i in featureVal:
        targets = values[:,1]
        x = values[:,i]
        
        N_TRAIN = 100
        x_train = x[0:N_TRAIN,:]
        x_test = x[N_TRAIN:,:]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]
    
        x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(500,1)
        y_ev1 = np.linspace(np.asscalar(min(t_train)), np.asscalar(max(t_train)), num=500).reshape(500,1)
        
        w,t_err= a1.linear_regression(x_train,t_train,'polynomial', bias,0,3,1)
        
        phi=a1.design_matrix(x_ev,"polynomial",3,bias)
        y_ev = phi@w
        
        fig, ax = plt.subplots()
        plt.plot(x_train,t_train,'g+',label='Train Data')
        plt.plot(x_test,t_test,'b*',label='Test Data')
        plt.plot(x_ev,y_ev,'y.',label='Polynomial Function')
        plt.xlabel('x-value')
        plt.ylabel('y-value')
        plt.title('Visualization of a function and data points of feature '+str(i+1) + ' with bias set to ' +str(bias))
        plt.rcParams["legend.loc"] = 'best'
        plt.legend(numpoints = 1)
        plt.show()
        
