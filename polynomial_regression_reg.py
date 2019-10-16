#!/usr/bin/env python

import assignment1 as a1
import matplotlib.pyplot as plt
import numpy as np

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_trainData = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_trainData = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
lam=[0,0.01,0.1,1, 10, 100,1000,10000]
avg=[]
for lambda_Val in lam:
    val_err_List=[]
    for i in range(1,11):
        if i==1:
            x_validate=x_trainData[10*(i-1):i*10,:]
            t_validate=t_trainData[10*(i-1):i*10,:]                   
            x_train=x_trainData[(i*10):N_TRAIN,:]
            t_train=t_trainData[(i*10):N_TRAIN,:]        
            
        elif i==10:
            x_validate=x_trainData[(10*(i-1)):i*10,:]
            x_train=x_trainData[0:10*(i-1),:]
            t_validate=t_trainData[(10*(i-1)):i*10,:]
            t_train=t_trainData[0:10*(i-1),:]                    
        
        else:
            x_validate=x_trainData[10*(i-1):i*10,:]
            t_validate=t_trainData[10*(i-1):i*10,:]                   
            x_train1=x_trainData[0:10*(i-1),:]   
            x_train2=x_trainData[(i*10):N_TRAIN,:] 
            x_train=np.append(x_train1,x_train2,axis=0)    
             
            t_train1=t_trainData[0:10*(i-1),:]   
            t_train2=t_trainData[(i*10):N_TRAIN,:]  
            t_train=np.append(t_train1,t_train2,axis=0)    
        
        w,t_err= a1.linear_regression(x_train,t_train,'polynomial',True,lambda_Val,2,None,None)
        pred,val_err= a1.evaluate_regression(x_validate,t_validate,'polynomial',True,w,2,None,None)
        #print("trainnnnnnnnnn",t_err)
        #print("testtttttttttt",val_err)
        val_err_List.append(val_err)  
 
        
    sum_of_val_err=sum(val_err_List)
    avg_of_val_err=sum_of_val_err/10
    if lambda_Val!=0:        
        avg.append(avg_of_val_err)
    else:
        avglamzero= avg_of_val_err

del lam[0]    
print("Average",avg)

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()

plt.semilogx(lam,avg)
plt.axhline(y=avglamzero, color='r', linestyle='-')
plt.ylabel('Average Value')
plt.rcParams["legend.loc"] = 'center right' 
plt.legend(["validation error","Validation Error for lambda 0"])
plt.title('Plot of average validation set error versus lambda')
plt.xlabel('Lambda')
plt.show()
