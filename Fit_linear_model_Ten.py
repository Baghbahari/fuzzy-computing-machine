####Fitting a linear model and plot the data and the model

import tensorflow as tf
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt

# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name="w")
# Our model of y = a*x + b
y_model = tf.multiply(x, w[0]) + w[1]

# Our error is defined as the square of the differences
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()
xs = []
ys = []

with tf.Session() as session:
    session.run(model)
    writer = tf.summary.FileWriter("output", session.graph)
    for i in range(10000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
	xs.append(x_value)
	ys.append(y_value)
        session.run(train_op, feed_dict={x: x_value, y: y_value})

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
a=w_value[0]
b=w_value[1]
writer.close()
plt.plot(xs,ys)
plt.plot(xs,a*np.array(xs) + b)
plt.show()
call(["tensorboard", "--logdir=output"])
