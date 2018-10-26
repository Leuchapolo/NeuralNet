# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(3)
#make num pixels variable replace num_px

num_px = 128
dim = num_px**2 *3



def load_image_data():
    # load image data 
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('../LogisticRegression/input/training_set', batch_size = 1, target_size = (num_px, num_px),  class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('../LogisticRegression/input/test_set', batch_size = 1, target_size = (num_px, num_px),  class_mode = 'binary')
    
    
    #preprocess data into arrays
    X_train = np.ndarray((len(training_set), dim))
    Y_train = np.ndarray((len(training_set), 1))
    X_test = np.ndarray((len(test_set), dim))
    Y_test = np.ndarray((len(test_set), 1))
    
    
    index = 0 
    
    
    
    
    while index < len(training_set):
        tr = training_set[index]
        
        X_train[index] = tr[0].reshape(tr[0].shape[1] * tr[0].shape[2] * tr[0].shape[3])
        Y_train[index] = tr[1][0]
        if index < len(test_set):
            te = test_set[index]
            X_test[index] = te[0].reshape(te[0].shape[1] * te[0].shape[2] * te[0].shape[3])
            Y_test[index] = te[1][0]
        index+=1
        
    
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    return X_train, X_test, Y_train, Y_test


#main program
    

# =============================================================================
# Load data
# =============================================================================

#X_train, X_test, Y_train, Y_test = load_image_data()

from numpy import genfromtxt

my_data = genfromtxt('Iris.csv', delimiter=',', dtype = None).T
X = my_data[1:-1, 1:]
Y = my_data[-1, 1:].reshape(1,150)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_Y = LabelEncoder()
Y[0,:] = labelEncoder_Y.fit_transform(Y[0,:])


oneHotEncoder_Y = OneHotEncoder()
Y = oneHotEncoder_Y.fit_transform(Y.T).toarray().T

X = X.astype(float)
Y = Y[0].astype(float).reshape(1,150)







X_train = X[:,:120]
X_test = X[:,120:]
Y_train = Y[:, :120]
Y_test = Y[:, 120:]




# =============================================================================
# Network dealings

# =============================================================================
 
from NeuralNet import NeuralNet as NN

net = NN.NeuralNet((4, 50, 1))

net.fit(X_train, Y_train,  0.001 , 9300)



Y_prediction_train = net.predict(X_train)
Y_prediction_test = net.predict(X)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y)) * 100))



#picture

#weight_pic  = net.sigmoid((net.w.reshape(64,64, 3)) * 255)
#weight_pic  = (net.w.reshape(64,64, 3)) * 255
#



# =============================================================================
#Image Generator
# =============================================================================


#
#
#net = SingleLayerImageGenerator(num_px)
#net.fit(Y_train, X_train, .01, 1000)
#net.fit(Y_train, X_train, .005, 1000)
#
#weight_pic = net.render().reshape(num_px, num_px, 3)
#
#
#
#
#
## =============================================================================
## Make pretty picture
## =============================================================================
#
#
        


#plt.imshow(weight_pic)


 
    

