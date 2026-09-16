import numpy as np 
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s=sigmoid(x)
    return s*(1-s)

def softmax(x):
    shifted_x=x-np.max(x,axis=1,keepdims=True)
    exp_x=np.exp(shifted_x)
    return exp_x/np.sum(exp_x,axis=1,keepdims=True)

