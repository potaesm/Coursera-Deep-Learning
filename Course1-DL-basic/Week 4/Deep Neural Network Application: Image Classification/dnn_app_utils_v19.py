#---------------------------------------- Created By Mr.Suthinan Musitmani ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import cv2
import os
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#---------------------------------------- Augmentation Generator Function ----------------------------------------

def image_augmentation(image_height=64, image_width=64, file_type='', base_path='', true_label='true', false_label='false', save_path='', times=5):
    """
    Use for generate the augmentation images
    """
    train_base_path = base_path + 'train_set/'
    test_base_path = base_path + 'test_set/'

    train_save_path = save_path + 'train_set/'
    test_save_path = save_path + 'test_set/'

    true_train_addrs = glob.glob(train_base_path + true_label + '/*.' + file_type)
    false_train_addrs = glob.glob(train_base_path + false_label + '/*.' + file_type)
    true_test_addrs = glob.glob(test_base_path + true_label + '/*.' + file_type)
    false_test_addrs = glob.glob(test_base_path + false_label + '/*.' + file_type)

    if not os.path.exists(train_save_path + true_label):
        os.makedirs(train_save_path + true_label)
    if not os.path.exists(train_save_path + false_label):
        os.makedirs(train_save_path + false_label)
    if not os.path.exists(test_save_path + true_label):
        os.makedirs(test_save_path + true_label)
    if not os.path.exists(test_save_path + false_label):
        os.makedirs(test_save_path + false_label)

    datagen = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

    for i in range(len(true_train_addrs)):
        true_train_img = cv2.imread(true_train_addrs[i])
        true_train_img = cv2.resize(true_train_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
        true_train_img = cv2.cvtColor(true_train_img, cv2.COLOR_BGR2RGB)
        
        true_train_data = img_to_array(true_train_img)
        true_train_samples = expand_dims(true_train_data, 0)

        true_train_it = datagen.flow(true_train_samples, save_to_dir=save_path + 'train_set/' + true_label, save_prefix='aug', save_format=file_type, batch_size=32)
        for j in range(times):
            true_train_it.next()

    for i in range(len(false_train_addrs)):
        false_train_img = cv2.imread(false_train_addrs[i])
        false_train_img = cv2.resize(false_train_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
        false_train_img = cv2.cvtColor(false_train_img, cv2.COLOR_BGR2RGB)
        
        false_train_data = img_to_array(false_train_img)
        false_train_samples = expand_dims(false_train_data, 0)

        false_train_it = datagen.flow(false_train_samples, save_to_dir=save_path + 'train_set/' + false_label, save_prefix='aug', save_format=file_type, batch_size=32)
        for j in range(times):
            false_train_it.next()

    for i in range(len(true_test_addrs)):
        true_test_img = cv2.imread(true_test_addrs[i])
        true_test_img = cv2.resize(true_test_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
        true_test_img = cv2.cvtColor(true_test_img, cv2.COLOR_BGR2RGB)

        true_test_data = img_to_array(true_test_img)
        true_test_samples = expand_dims(true_test_data, 0)

        true_test_it = datagen.flow(true_test_samples, save_to_dir=save_path + 'test_set/' + true_label, save_prefix='aug', save_format=file_type, batch_size=32)
        for j in range(times):
            true_test_it.next()

    for i in range(len(false_test_addrs)):
        false_test_img = cv2.imread(false_test_addrs[i])
        false_test_img = cv2.resize(false_test_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
        false_test_img = cv2.cvtColor(false_test_img, cv2.COLOR_BGR2RGB)

        false_test_data = img_to_array(false_test_img)
        false_test_samples = expand_dims(false_test_data, 0)

        false_test_it = datagen.flow(false_test_samples, save_to_dir=save_path + 'test_set/' + false_label, save_prefix='aug', save_format=file_type, batch_size=32)
        for j in range(times):
            false_test_it.next()

    return

#---------------------------------------- H5 Converter Function ----------------------------------------

def create_h5(image_height=64, image_width=64, image_depth=3, file_type='', base_path='', true_label='true', false_label='false'):
    """
    Use for create the train_dataset.h5 and test_dataset.h5 from the set of images
    """
    train_base_path = base_path + 'train_set/'
    test_base_path = base_path + 'test_set/'

    true_train_addrs = glob.glob(train_base_path + true_label + '/*.' + file_type)
    false_train_addrs = glob.glob(train_base_path + false_label + '/*.' + file_type)

    train_addrs = true_train_addrs + false_train_addrs

    true_test_addrs = glob.glob(test_base_path + true_label + '/*.' + file_type)
    false_test_addrs = glob.glob(test_base_path + false_label + '/*.' + file_type)

    test_addrs = true_test_addrs + false_test_addrs

    true_train_labels = [1 for addr in true_train_addrs]
    false_train_labels = [0 for addr in false_train_addrs]

    train_labels = true_train_labels + false_train_labels

    true_test_labels = [1 for addr in true_test_addrs]
    false_test_labels = [0 for addr in false_test_addrs]

    test_labels = true_test_labels + false_test_labels

    list_classes = [false_label.encode('utf8'), true_label.encode('utf8')]

    train_shape = (len(train_addrs), image_height, image_width, image_depth)
    test_shape = (len(test_addrs), image_height, image_width, image_depth)

    if not os.path.exists('datasets/'):
        os.makedirs('datasets/')

    train_hdf5_path = 'datasets/train_dataset.h5'
    test_hdf5_path = 'datasets/test_dataset.h5'

    with h5py.File(train_hdf5_path, mode='w', libver='latest', swmr=True) as train_hdf5_file:
        train_img_arr = []
        for i in range(len(train_addrs)):
            train_img = cv2.imread(train_addrs[i])
            train_img = cv2.resize(train_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
            train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
            train_img_arr.append(train_img)
        train_hdf5_file.create_dataset("train_set_x", data=train_img_arr)
        train_hdf5_file.create_dataset("train_set_y", data=train_labels)
        train_hdf5_file.create_dataset("list_classes", data=list_classes)

    with h5py.File(test_hdf5_path, mode='w', libver='latest', swmr=True) as test_hdf5_file:
        test_img_arr = []
        for i in range(len(test_addrs)):
            test_img = cv2.imread(test_addrs[i])
            test_img = cv2.resize(test_img, (image_height, image_width), interpolation=cv2.INTER_CUBIC)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            test_img_arr.append(test_img)
        test_hdf5_file.create_dataset("test_set_x", data=test_img_arr)
        test_hdf5_file.create_dataset("test_set_y", data=test_labels)
        test_hdf5_file.create_dataset("list_classes", data=list_classes)

    return

#---------------------------------------- Forward Propagation Functions ----------------------------------------
def tanh(Z):
    """
    Implements the tanh activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of tanh(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.tanh(Z)
    cache = Z
    
    return A, cache

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the ReLU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0, Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def leaky_relu(Z):
    """
    Implement the Leaky ReLU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0.01*Z, Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def linear_function(Z):
    """
    Implements the linear function.
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of linear(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = Z
    cache = Z
    return A, cache

#---------------------------------------- Backward Propagation Functions ----------------------------------------

def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single TANH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = np.tanh(Z)
    dZ = dA * (1 - s ** 2)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def leaky_relu_backward(dA, cache):
    """
    Implement the backward propagation for a single LEAKY RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0.01
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_function_backward(dA, cache):
    """
    Implement the backward propagation for a single LINEAR FUNCTION unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    dZ = dA
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#---------------------------------------- Load Data ----------------------------------------

def load_data(base_path='datasets/'):
    train_dataset = h5py.File(base_path + 'train_dataset.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(base_path + 'test_dataset.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#---------------------------------------- Initialize Parameters ----------------------------------------

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     

#---------------------------------------- Initialize Parameters For L-layer ----------------------------------------

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

#---------------------------------------- Activation Forward ----------------------------------------

def linear_forward(A, W, b):
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
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
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
    
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = tanh(Z)

    elif activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = relu(Z)

    elif activation == "leaky-relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = leaky_relu(Z)
    
    elif activation == "linear":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = linear_function(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#---------------------------------------- Activation Forward For L-layer ----------------------------------------

def L_model_forward(X, parameters, l_activation = "relu", o_activation = "sigmoid"):
    """
    Implement forward propagation for the [LINEAR->L-Activation]*(L-1)->LINEAR->Output-Activation computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = l_activation)
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = o_activation)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

#---------------------------------------- Cost Function Computation ----------------------------------------

def compute_cost(AL, Y):
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
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

#---------------------------------------- Activation Backward ----------------------------------------

def linear_backward(dZ, cache):
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

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
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
    
    if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "leaky-relu":
        dZ = leaky_relu_backward(dA, activation_cache)
    
    elif activation == "linear":
        dZ = linear_function_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

#---------------------------------------- Activation Backward For L-layer ----------------------------------------

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#---------------------------------------- Update Parameters ----------------------------------------

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

#---------------------------------------- Predict ----------------------------------------

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

#---------------------------------------- Print Mislabeled Images ----------------------------------------

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
