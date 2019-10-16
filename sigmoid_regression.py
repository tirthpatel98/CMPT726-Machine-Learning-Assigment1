#!/usr/bin/env python
import assignment1 as a1
import matplotlib.pyplot as plt
import numpy as np
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10]

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
mu_arr=[100,10000]
weight=[]
train_err=[]
test_err=[]
s=2000

w,t_err= a1.linear_regression(x_train,t_train,'sigmoid',True,0,None,mu_arr,s)
weight.append(w)
train_err.append(t_err)
pred,te_err= a1.evaluate_regression(x_test,t_test,'sigmoid',True,w,None,mu_arr,s)
test_err.append(te_err)
#print("Testttttt",te_err)
#print("predddddddd",pred)
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(500,1)

phi=a1.design_matrix(x_ev,"sigmoid",3,True,mu_arr,s)
y_ev = phi@w



plt.plot(x_train,t_train,'g+',label='Train Data')
plt.plot(x_test,t_test,'b*',label='Test Data')
plt.plot(x_ev,y_ev,'y.',label='Polynomial Function')
plt.xlabel('x-values')
plt.ylabel('y-values')
plt.title('Visualization of a sigmoid function for feature 11')
plt.legend(numpoints = 1)
plt.show()