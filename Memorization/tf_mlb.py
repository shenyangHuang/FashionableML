""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Original code:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


from __future__ import print_function
from mlp_loading import flatten, get_mnist
import tensorflow as tf
import numpy as np
from shuffle import shuffle_labels, randomize

def convert_to_onehot(labels):
    one_labels = []
    for y in labels:
        el = [0]
        arr = el * 10
        el = np.array(arr).astype(np.float64)
        arr[int(y)] += 1
        one_labels.append(arr)
    return one_labels
display_step = 1

train, lab, valid, v_labels, test, test_labels = get_mnist()
labels = np.array(convert_to_onehot(randomize(lab)))
v_labels = np.array(convert_to_onehot(v_labels))
train = np.array(flatten(train))
test = np.array(flatten(test))
valid = np.array(flatten(valid))

# trying a subset
labels = labels[:1000]
v_labels = v_labels[:1000]
train = train[:1000]
test = test[:1000]
valid = valid[:1000]

print("Loaded data")


# Parameters
# small ratio
learning_rate = 0.005
training_epochs = 300
batch_size = 800
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = (tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 =  (tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_2 = tf.nn.relu(layer_2)    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

valid_1000 = []
train_1000 = []
cost_1000 = []

# Test model
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = train[i*batch_size:i*batch_size + batch_size]
            batch_y = labels[i*batch_size:i*batch_size + batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            # Should I do it on the entire set? Or just the batch?
            # for now, entire set
            #if i % batch_size/4 == 0:
            #    print(1/4)
        # Display logs per epoch step
            
        if epoch % display_step == 0:
            train_acc = accuracy.eval({X: train, Y: labels})
            valid_acc = accuracy.eval({X: valid, Y: v_labels})
            valid_1000.append(valid_acc)
            train_1000.append(train_acc)
            print(train_acc)
            print(valid_acc)
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            cost_1000.append(avg_cost)

    print("Optimization Finished!")
