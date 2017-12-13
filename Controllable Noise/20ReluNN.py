"""
20 layer MLP ReLU network run on the FashionMNIST and MNIST datasets.
Output: file "accuracies.txt" describing the ratios and resulting training and 
test accuracies given by multiple optimizers including gradient descent. 
Used to replicate Figure 1.b in Section 4.1 of the original paper.

Python version: 3.6.2
Tensorflow version: 1.4.0

Code based on: https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a
Code by: Shenyang Huang & Kaylee Kutschera
"""

# Import libraries
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

# Import Fashion MNIST (downloaded from https://github.com/zalandoresearch/fashion-mnist)
mnist = input_data.read_data_sets('FashionMNIST_data', one_hot=True)
# Replace with below to run on the original MNIST dataset (downloaded from http://yann.lecun.com/exdb/mnist/)
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define trainset and testset
train = mnist.train
test = mnist.test

# Our network is deep: it has 20 hidden layers, with each layer containing 50 hidden units. 
# We use the ReLU activation function
# Network parameters
n_hidden = 50
# 50 hidden units in each hidden layer 
n_input = 784 # Fashion MNIST data input (img shape: 28*28)
n_classes = 10 # Fashion MNIST total classes (0–9 digits)
n_samples = mnist.train.num_examples # Number of examples in training set 

Train_Accuracy = []
Test_Accuracy = []

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
    return X, Y

def initialize_parameters(ramseed):
    '''
    ramseed sets the random seed used for initialization
    Initializes parameters to build a neural network with tensorflow. 
    
    Returns:
    parameters -- a dictionary of tensors containing weights and biases
    '''
    
    # Set random seed for reproducibility
    tf.set_random_seed(ramseed)
    
    # Initialize weights and biases for each layer
    w1 = tf.get_variable("w1", [n_hidden, n_input], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b1 = tf.get_variable("b1", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w2 = tf.get_variable("w2", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b2 = tf.get_variable("b2", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w3 = tf.get_variable("w3", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b3 = tf.get_variable("b3", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w4 = tf.get_variable("w4", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b4 = tf.get_variable("b4", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w5 = tf.get_variable("w5", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b5 = tf.get_variable("b5", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w6 = tf.get_variable("w6", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b6 = tf.get_variable("b6", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w7 = tf.get_variable("w7", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b7 = tf.get_variable("b7", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w8 = tf.get_variable("w8", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b8 = tf.get_variable("b8", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w9 = tf.get_variable("w9", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b9 = tf.get_variable("b9", [n_hidden, 1], initializer=tf.zeros_initializer())

    w10 = tf.get_variable("w10", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b10 = tf.get_variable("b10", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w11 = tf.get_variable("w11", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b11 = tf.get_variable("b11", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w12 = tf.get_variable("w12", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b12 = tf.get_variable("b12", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w13 = tf.get_variable("w13", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b13 = tf.get_variable("b13", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w14 = tf.get_variable("w14", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b14 = tf.get_variable("b14", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w15 = tf.get_variable("w15", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b15 = tf.get_variable("b15", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w16 = tf.get_variable("w16", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b16 = tf.get_variable("b16", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w17 = tf.get_variable("w17", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b17 = tf.get_variable("b17", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w18 = tf.get_variable("w18", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b18 = tf.get_variable("b18", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w19 = tf.get_variable("w19", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b19 = tf.get_variable("b19", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    w20 = tf.get_variable("w20", [n_hidden, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    b20 = tf.get_variable("b20", [n_hidden, 1], initializer=tf.zeros_initializer())
    
    Weight_Output = tf.get_variable("Weight_Output", [n_classes, n_hidden], initializer=tf.contrib.layers.xavier_initializer(seed=ramseed))
    bias_Output = tf.get_variable("bias_Output", [n_classes, 1], initializer=tf.zeros_initializer())
    
    # Store initializations as a dictionary of parameters
    parameters = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2,
        "w3":w3,
        "b3":b3,
        "w4":w4,
        "b4":b4,
        "w5":w5,
        "b5":b5,
        "w6":w6,
        "b6":b6,
        "w7":w7,
        "b7":b7,
        "w8":w8,
        "b8":b8,
        "w9":w9,
        "b9":b9,
        "w10":w10,
        "b10":b10,
        "w11":w11,
        "b11":b11,
        "w12":w12,
        "b12":b12,
        "w13":w13,
        "b13":b13,
        "w14":w14,
        "b14":b14,
        "w15":w15,
        "b15":b15,
        "w16":w16,
        "b16":b16,
        "w17":w17,
        "b17":b17,
        "w18":w18,
        "b18":b18,
        "w19":w19,
        "b19":b19,
        "w20":w20,
        "b20":b20,
        "Weight_Output":Weight_Output,
        "bias_Output":bias_Output
    }
    
    return parameters

def forward_propagation(X, parameters):
    '''
    Arguments:
    X - input dataset placeholder, of shape (input size, number of examples)
    parameters - python dictionary containing the parameters 
    
    Returns:
    output - the output of the last LINEAR unit
    '''
    w1 = parameters['w1']
    b1 = parameters['b1']
    Z1 = tf.add(tf.matmul(w1,X), b1)  
    A1 = tf.nn.relu(Z1)
    
    w2 = parameters['w2']
    b2 = parameters['b2']
    Z2 = tf.add(tf.matmul(w2,A1), b2)  
    A2 = tf.nn.relu(Z2)
    
    w3 = parameters['w3']
    b3 = parameters['b3']
    Z3 = tf.add(tf.matmul(w3,A2), b3)  
    A3 = tf.nn.relu(Z3)
    
    w4 = parameters['w4']
    b4 = parameters['b4']
    Z4 = tf.add(tf.matmul(w4,A3), b4)  
    A4 = tf.nn.relu(Z4)
    
    w5 = parameters['w5']
    b5 = parameters['b5']
    Z5 = tf.add(tf.matmul(w5,A4), b5)  
    A5 = tf.nn.relu(Z5)
    
    w6 = parameters['w6']
    b6 = parameters['b6']
    Z6 = tf.add(tf.matmul(w6,A5), b6)  
    A6 = tf.nn.relu(Z6)
    
    w7 = parameters['w7']
    b7 = parameters['b7']
    Z7 = tf.add(tf.matmul(w7,A6), b7)  
    A7 = tf.nn.relu(Z7)
    
    w8 = parameters['w8']
    b8 = parameters['b8']
    Z8 = tf.add(tf.matmul(w8,A7), b8)  
    A8 = tf.nn.relu(Z8)
    
    w9 = parameters['w9']
    b9 = parameters['b9']
    Z9 = tf.add(tf.matmul(w9,A8), b9)  
    A9 = tf.nn.relu(Z9)
    
    w10 = parameters['w10']
    b10 = parameters['b10']
    Z10 = tf.add(tf.matmul(w10,A9), b10)  
    A10 = tf.nn.relu(Z10)
    
    w11 = parameters['w11']
    b11 = parameters['b11']
    Z11 = tf.add(tf.matmul(w11,A10), b11)  
    A11 = tf.nn.relu(Z11)
    
    w12 = parameters['w12']
    b12 = parameters['b12']
    Z12 = tf.add(tf.matmul(w12,A11), b12)  
    A12 = tf.nn.relu(Z12)
   
    w13 = parameters['w13']
    b13 = parameters['b13']
    Z13 = tf.add(tf.matmul(w13,A12), b13)  
    A13 = tf.nn.relu(Z13)
    
    w14 = parameters['w14']
    b14 = parameters['b14']
    Z14 = tf.add(tf.matmul(w14,A13), b14)  
    A14 = tf.nn.relu(Z14)
    
    w15 = parameters['w15']
    b15 = parameters['b15']
    Z15 = tf.add(tf.matmul(w15,A14), b15)  
    A15 = tf.nn.relu(Z15)
    
    w16 = parameters['w16']
    b16 = parameters['b16']
    Z16 = tf.add(tf.matmul(w16,A15), b16)  
    A16 = tf.nn.relu(Z16)
    
    w17 = parameters['w17']
    b17 = parameters['b17']
    Z17 = tf.add(tf.matmul(w17,A16), b17)  
    A17 = tf.nn.relu(Z17)
    
    w18 = parameters['w18']
    b18 = parameters['b18']
    Z18 = tf.add(tf.matmul(w18,A17), b18)  
    A18 = tf.nn.relu(Z18)
    
    w19 = parameters['w19']
    b19 = parameters['b19']
    Z19 = tf.add(tf.matmul(w19,A18), b19)  
    A19 = tf.nn.relu(Z19)
    
    w20 = parameters['w20']
    b20 = parameters['b20']
    Z20 = tf.add(tf.matmul(w20,A19), b20)  
    A20 = tf.nn.relu(Z20)
    
        
    # Enter the output layer 
    w_last = parameters['Weight_Output']
    b_last = parameters['bias_Output']
    output = tf.add(tf.matmul(w_last,A20), b_last)
    
    return output

def compute_cost(Z, Y):
    '''
    Computes the cost
    
    Arguments:
    Z - output of forward propagation (output of the last LINEAR unit)
    Y - "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    '''
    
    # Get logits (predictions) and labels
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    # Compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model(train, test, ramseed,learning_rate, num_epochs, minibatch_size, optim):
    '''
    Implements the tensorflow neural network.

    Arguments:
    train - training set
    test - test set
    ramseed - random seed
    learning_rate - learning rate of the optimization
    num_epochs - number of epochs of the optimization loop
    minibatch_size - size of a minibatch
    optim - optimizer to use in the network
    
    Returns:
    parameters - parameters learnt by the model
    '''
    
    # Ensure that model can be rerun without overwriting tf variables
    ops.reset_default_graph()
    # For reproducibility
    tf.set_random_seed(ramseed)
    seed = ramseed
    # Get input and output shapes
    (n_x, m) = train.images.T.shape
    n_y = train.labels.T.shape[0]
    
    # Create placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    # Initialize parameters
    parameters = initialize_parameters(ramseed)
    
    # Forward propagation
    Z = forward_propagation(X, parameters)
    # Cost function
    cost = compute_cost(Z, Y)

    
    # Backpropagation 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    if optim == "Adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
    elif optim == "Adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
    elif optim == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    elif optim == "ProximalGradientDescent":
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(cost)
    elif optim == "ProximalAdagrad":
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate).minimize(cost)
    elif optim == "RMSProp":
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Start session to compute Tensorflow graph
    with tf.Session() as sess:
        
        # Run initialization
        sess.run(init)
        
        # Training loop
        for epoch in range(num_epochs):
            
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            
            for i in range(num_minibatches):
                
                # Get next batch of training data and labels
                minibatch_X, minibatch_Y = train.next_batch(minibatch_size)
                
                # Execute optimizer and cost function
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X.T, Y: minibatch_Y.T})
                
                # Update epoch cost
                epoch_cost += minibatch_cost / num_minibatches
        
        # Save parameters
        parameters = sess.run(parameters)
        
        # Calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        
        # Calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: train.images.T, Y: train.labels.T}))
        print ("Test Accuracy:", accuracy.eval({X: test.images.T, Y: test.labels.T}))
        
        # Append training and testing accuracy
        Train_Accuracy.append(accuracy.eval({X: train.images.T, Y: train.labels.T}))
        Test_Accuracy.append(accuracy.eval({X: test.images.T, Y: test.labels.T}))
        
        
        return parameters



def make_hparam_string(learning_rate, batch_size,epoch, optimizer):
    return "learning_rate:%.00E,batch_size:%d, epoch:%d, optimizer:%s" % (learning_rate, batch_size,epoch, optimizer)

optimizers = ["GradientDescent"]
# To run on multiple different optimizers, replace the above line with the following line.
#optimizers = ["GradientDescent", "Adadelta", "Adagrad", "Adam", "ProximalGradientDescent", "ProximalAdagrad", "RMSProp"]

for optimizer in optimizers:    
    learning_rates = [0.05,0.05,0.025,0.04,0.02,0.02,0.01,0.01,0.02,0.005,0.03,0.02,0.06, 0.015,0.011,0.0075]
    batch_sizes = [50,1000,500,1000,500,100,50,100,200,1000,50,50,75,100,100,100]

    ratio = [x/y for x, y in zip(learning_rates, batch_sizes)]
    epoch = 15
    Train_Accuracy = []
    Test_Accuracy = []
    for i in range(len(learning_rates)):
        learning_rate = learning_rates[i]
        batch_size = batch_sizes[i]
        hparam = make_hparam_string(learning_rate, batch_size, epoch, optimizer)
        print('Starting run for %s' % hparam)
        # Run model with settings
        model(train,test,42,learning_rate,epoch,batch_size, optimizer)
    print("Completed")
    print()
    
    
    with open("accuracies.txt", "a") as out:
        out.write("Using optimizer: "+ optimizer+"\n")
        out.write("ratios = " + str(ratio)+"\n")
        out.write("train_accuracy = " + str(Train_Accuracy)+"\n")
        out.write("test_accuracy = " + str(Test_Accuracy)+"\n")
        out.write("-------------------------------------------------------------\n")
