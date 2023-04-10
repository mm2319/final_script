import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from keras.optimizers import SGD, Adam
import numpy as np

def loss_f(x, u):
    """
    this function measures the difference between true value of U(t) and ground-truth U (the size of cells)
    x: ground-truth U (the size of cells)
    u: U(t)
    
    """
    x = tf.cast(x, dtype=tf.double)
    u = tf.cast(u,dtype=tf.double)
    return tf.reduce_mean(tf.square(tf.subtract(x , u)))

def train_nonlinear(model, epoch, T, Y):
    optimizer = SGD(learning_rate=1.e-3)
    for epoch in range(epoch):
        print("start of epoch %d" %(epoch,))
        for i in range(len(Y[:,0])):

            # define the polynomial linear regression inputs and ouputs
            x = tf.constant([[T[i]]])
            u = tf.constant([Y[i,0],Y[i,1]])
            # predict the size of cells based on the input
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_variables)
                y = model(x)
                d_loss =loss_f(u, y)

            # calculate the gradients for each parameters
            grads = tape.gradient(d_loss, model.trainable_variables)

            # update each parameters
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

def obtain_nonlinear_data(model, T):
    x_train = []
    y_1_train = []
    y_2_train = []
    for i in range(len(T)):
            x = tf.constant([[T[i]]])
            with tf.GradientTape() as t:
                t.watch(x)
                y = model(x)
            grad = t.jacobian(y, x)
        
            grad = tf.reshape(grad, [2])
            Y = tf.reshape(y, shape = (1,2)).numpy()
            u = Y[0][0]
            v = Y[0][1]
            x = np.array([1, u, v, u**2, v**2, u*v, u**3, v**3])
            x_train.append(x)
            y_1_train.append(grad.numpy()[0])
            y_2_train.append(grad.numpy()[1])
    return x_train, y_1_train, y_2_train