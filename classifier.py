import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
from utils import *
import pickle

STATES = {"Andhra Pradesh" : 0, "Arunachal Pradesh" : 1, "Assam" : 2, "Bihar" : 3, "Chhattisgarh" : 4,
		  "Goa": 5, "Gujarat" : 6,"Haryana" : 7,"Himachal Pradesh" : 8, "Jammu and Kashmir" : 9, "Jharkhand" : 10, 
		  "Karnataka" : 11, "Kerala" : 12, "Madhya Pradesh" : 13, "Maharashtra" : 14, "Manipur" : 15,"Meghalaya" : 16,
		  "Mizoram" : 17, "Nagaland" : 18, "Odisha" : 19, "Punjab" : 20, "Rajasthan" : 21, "Sikkim" : 22,"Tamil Nadu" : 23,
		  "Telangana" : 24,"Tripura" : 25, "Uttar Pradesh" : 26, "Uttarakhand" : 27, "West Bengal" : 28}

EPC_VENDORS = {"LNT" : 0, "Nagarjuna" : 1, "GMR" : 2, "Tata" : 3, "Gammon" : 4}
data = pd.read_csv("data3.csv")

data['Geography'] = data['Geography'].map(STATES)
data['EPC Vendors'] = data['EPC Vendors'].map(EPC_VENDORS)

X = data.values[:, 1:]
max_values = X.max(axis=0)
X = X / max_values
Y = data.values[:, 0]

print(X.shape)
Y = Y.reshape(len(Y), 1)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

print("Shape of Training Data: {}".format(X_train.shape))
print("Shape of Testing Data: {}".format(X_test.shape))
print("Shape of Training Output: {}".format(y_train.shape))
print("Shape of Testing Output: {}".format(y_test.shape))

layers_dims = [4, 10, 10, 8, 8, 5, 1]

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, number of features)
    Y -- true "label" vector (containing 0 if bid successfull, 1 otherwise), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. 
    parameters = initialize_parameters_deep(layers_dims)
        
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
                
        # Compute cost.
        cost = compute_cost(AL, Y)
            
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
         
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

print("Max value: {}".format(max_values))
parameters = L_layer_model(X_train, y_train, layers_dims, num_iterations = 50000, print_cost = True)
with open('classifier_parameters.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('max_values.pickle', 'wb') as handle:
    max_vals = {"max_values": max_values}
    pickle.dump(max_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)


pred_train = predict(X_train, y_train, parameters)
pred_test = predict(X_test, y_test, parameters)
#out = np.array([3, 24, 25993, 59]).reshape(1, 4)
#print(out.shape)
#out = out / max_values
#print(out.shape)
#print(max_values.shape)
#result = predict(out.T, y_test, parameters)
#print(result)