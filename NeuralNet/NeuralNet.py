# -*- coding: utf-8 -*-

import numpy as np


class NeuralNet:
    def __init__(self, layer_dims):
        self.parameters = {}
        self.L = len(layer_dims)          # number of layers in the network
        np.random.seed(3)
        for l in range(1, self.L):
            ### START CODE HERE ### (≈ 2 lines of code)
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * .01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1) )
            ### END CODE HERE ###
            
            assert(self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(self.parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
    
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
    
        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """
        
        ### START CODE HERE ### (≈ 1 line of code)
        
        Z = np.dot(W, A) + b
        ### END CODE HERE ###
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
            ### END CODE HERE ###
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
            ### END CODE HERE ###
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """
    
        caches = []
        A = X
        L = len(self.parameters) // 2                  # number of layers in the neural network
        
        
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W" + str(l)], self.parameters["b" + str(l)], "relu")
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_activation_forward(A, self.parameters["W" + str(L)], self.parameters["b" + str(L)], "sigmoid")
        caches.append(cache)
        ### END CODE HERE ###
        
        assert(AL.shape == (1,X.shape[1]))
                
        return AL, caches
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)
    
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        ### START CODE HERE ### (≈ 3 lines of code)
        dW = 1./m * np.dot(dZ, A_prev.T)
        self.dZ = dZ
        self.A = A_prev.T
        self.m = m
        db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        ### END CODE HERE ###
        
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    # GRADED FUNCTION: linear_activation_backward

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            ### START CODE HERE ### (≈ 2 lines of code)
            
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        elif activation == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            ### END CODE HERE ###
            
        
        return dA_prev, dW, db
    
    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ... 
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        
        
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
            
            
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
           
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your
        parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(self.parameters) // 2 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        
        
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(L):
            
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    def fit(self, X, Y, learning_rate, num_iter):
        for i in range(num_iter):
            
            AL, caches = self.L_model_forward(X)
            
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            
            self.update_parameters(grads, learning_rate)
            if i % 100 == 0:
                
                print("Cost at iteration " + str(i) + " is :" + str(cost))
    
    
    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).
    
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]
    
        # Compute loss from aL and y.
        
        AL = np.where(AL == 0, 0.001, AL)
        cost = -1./m * np.sum(Y *np.log(AL) + (1-Y)*np.log(1-AL))
        
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost
    
    def predict(self, X):
        A, caches = self.L_model_forward(X)
        y_predict = np.where(A > 0.5, 1, 0)
        
        return y_predict
    
    
    def relu(self, Z):
        R = np.maximum(Z, 0)
        return R, R
        
    def relu_backward(self, dA, z):
        z[z<=0] = 0
        z[z>0] = 1
        z = np.multiply(z, dA)
        return z 
    
    def sigmoid_backward(self, dA, z):
        
        
        dz = (1.0/(1.0+np.exp(-z))) * ((1 - 1.0/(1.0+np.exp(-z))))
        dz = dz * dA
        return dz
        
    def sigmoid(self, z):
        """
        Compute the sigmoid of z
    
        Arguments:
        z -- A scalar or numpy array of any size.
    
        Return:
        s -- sigmoid(z)
        """
    
        
        s = 1.0/(1.0+np.exp(-z))
        
        
        return s, s 