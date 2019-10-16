"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'
    
    
    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    
   
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')
    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)    
   

def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x_train, t_train, basis, bias,reg_lambda=0, degree=1, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis
    
    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """
    
    # Construct the design matrix.
    # Pass the required parameters to this function
    
    phi = design_matrix(x_train,basis,degree,bias,mu,s)   
    #print(x_train.shape)      
    # Learning Coefficients
    if reg_lambda > 0:
        I=np.identity((phi.shape[1]),dtype=int)
        inv = np.linalg.inv((reg_lambda*I)+(phi.T@phi))
        w = inv@(phi.T@t_train) 
        # regularized regression
    else:
        # no regularization 
        w = np.linalg.pinv(phi)@t_train
        
    pred_train=phi@w
    train_err = np.sqrt((np.square(pred_train-t_train)).mean())
    return (w, train_err)



def design_matrix(x,basis=None,degree=1,bias=True,mu=None,s=1):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        ?????

    Returns:
      phi design matrix
    """
    
    if basis == 'polynomial':
        if(degree==1):  
            if bias == True:                                                                    
                x=np.append(np.ones((len(x),1)).astype(int),values=x,axis=1)    
                phi=x
            else:
                pass    
        else:
            newMatrix=x
            for i in range(2,degree+1):
                temp=np.power(x,i)
                newMatrix=np.concatenate((newMatrix,temp),axis=1)
            if bias == True:
                newMatrix=np.append(np.ones((len(newMatrix),1)).astype(int),values=newMatrix,axis=1)
            phi=newMatrix         
        
    elif basis == 'sigmoid':

        for i in mu:
            if(i==mu[0]):
                temp= (x-i)/s
                phi1=1/(1+np.exp(-temp))
                phi=phi1
            else:
                temp= (x-i)/s
                phi1=1/(1+np.exp(-temp))
                phi=np.concatenate((phi,phi1),axis=1)
        phi=np.append(np.ones((len(phi),1)).astype(int),values=phi,axis=1)
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x_test,t_test,basis,bias,w,degree=1,mu=None,s=1):
    """Evaluate linear regression on a dataset.
     
    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      
      """
    
    phi = design_matrix(x_test,basis,degree,bias,mu,s)
    pred_test=phi@w
    # Measure root mean squared error on testing data.
    t_est = pred_test
    #print("deleteeeeeeeeeee",t_est)
    #print(np.shape(t_est))
    err = np.sqrt((np.square(pred_test-t_test)).mean())
    
    

    return (t_est, err)


    
    

load_unicef_data()