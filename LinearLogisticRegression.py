#import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    sample_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, sample_per_class)
    Y0 = np.zeros((sample_per_class))

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, sample_per_class)
        Y1 = (ci + 1)* np.ones(sample_per_class)
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
    
    if regression == False:
        class_ind = [Y== class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype = np.float32)
    X, Y = shuffle(X0, Y0)
    return X, Y

np.random.seed(10)
num_classes = 2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0], True)
colors = ['r' if l==0 else 'b' for l in Y[:]]
plt.scatter(X[:, 0], X[:, 1], c = colors)
plt.xlabel("Scaled age (in years)")
plt.ylabel("Tumor size (in cm)")
plt.show()

input_dim = 2
lab_dim = 1
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name = "weight")
b = tf.Variable(tf.zeros([lab_dim]), name = "bias")

output = tf.nn.sigmoid( tf.matmul(input_features, W) + b)
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels)*tf.log(1 - output))
serr = tf.square(input_labels - output)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(serr)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)