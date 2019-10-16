#!/usr/bin/env python

import assignment1 as a1
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]


normalize_err=[0,1]

for normalize in normalize_err:
    
    if(normalize==1):
        x = a1.normalize_data(x)
    
        N_TRAIN = 100
        x_train = x[0:N_TRAIN,:]
        x_test = x[N_TRAIN:,:]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]

    else:
        
        N_TRAIN = 100
        x_train = x[0:N_TRAIN,:]
        x_test = x[N_TRAIN:,:]
        t_train = targets[0:N_TRAIN]
        t_test = targets[N_TRAIN:]
    
    weight=[]
    train_err=[]
    test_err=[]
    bias_err=[]
    train_dict={}
    test_dict={}
    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions
    for i in range (1,7):
        w,t_err= a1.linear_regression(x_train,t_train,'polynomial',True,0,i,1)
        
        train_dict[i]=t_err
        weight.append(w)
        train_err.append(t_err)
        #print("trainnnnnnnnnnnnnnn",train_err)
        pred,te_err= a1.evaluate_regression(x_test,t_test,'polynomial',True,weight[i-1],i)
        test_dict[i]=te_err
        test_err.append(te_err)
    print("TrainError",train_err)
    print("TestError",test_err)

  
    #print(countries[ans1[0]])
    
    # Produce a plot of results.
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.plot(list(train_dict.keys()), list(train_dict.values()),color='b')
    plt.plot(list(test_dict.keys()), list(test_dict.values()),color='g')
    plt.ylabel('RMS')
    plt.rcParams["legend.loc"] = 'best' 
    plt.legend(["training error","testing error"])
    if(normalize==1):
        plt.title('Fit with polynomials, no regularization with normalization')
    else:
         plt.title('Fit with polynomials, no regularization without normalization')
    plt.xlabel('Polynomial degree')
    plt.show()
  
