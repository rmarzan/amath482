#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# ## Preprocessing the data

# In[3]:


# use 5,000 training images for validation data
# convert uint8 to floating point numbers by dividing by 255.0
# create validation set
X_valid = X_train_full[:5000] / 255.0
X_train = X_train_full[5000:] / 255.0
X_test = X_test / 255.0

y_valid = y_train_full[:5000]
y_train = y_train_full[5000:]


# # Fully Connected Neural Network

# ### Building the model

# In[4]:


from functools import partial
# (use 'partial' to eliminate redundant lines)
my_dense_layer = partial(tf.keras.layers.Dense, activation="relu")

model = tf.keras.models.Sequential([
    # flattened input layer
    tf.keras.layers.Flatten(input_shape=[28,28]),
    # hidden layer; dense = fully connected
    my_dense_layer(784),
    my_dense_layer(32),
    # output layer; 10 probabilities for 10 classes of items
    my_dense_layer(10, activation="softmax")
])

# learning rate = determines how much to change weights/biases in each iteration; very important
model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=["accuracy"])


# ### Training the model

# In[5]:


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid,y_valid))


# ### RESULTS

# In[6]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)
model.evaluate(X_test,y_test)


# # Convolutional Neural Network

# ### Preprocessing the data

# In[7]:


# Training: 55000x28x28 (TensorFlow wants 55000x28x28x1)
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# ### Building the model

# In[8]:


from functools import partial

my_conv_layer = partial(tf.keras.layers.Conv2D, activation="relu", padding="valid", kernel_initializer="he_uniform")
my_dense_layer = partial(tf.keras.layers.Dense, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))

model = tf.keras.models.Sequential([
    # no need to flatten layer for CNN
    # Conv2D args = # of filters, filter size, zero padding = same (as input size)
    my_conv_layer(32,3,padding="same",input_shape=[28,28,1]),
    tf.keras.layers.AveragePooling2D(2),
    my_conv_layer(64,3),
    tf.keras.layers.MaxPooling2D(2),
    my_conv_layer(128,3),
    
    # must flatten before going into fully connected
    tf.keras.layers.Flatten(),
    # hidden layer: how many neurons; dense = fully connected
    my_dense_layer(32),
    # output layer; 10 probabilities for 10 classes
    my_dense_layer(10, activation="softmax")
])


# decide on options for training
# learning rate = determines how much to change weights/biases in each iteration; very important
model.compile(loss="sparse_categorical_crossentropy",
             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=["accuracy"])


# ### Training the model

# In[9]:


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid,y_valid))


# ### RESULTS

# In[11]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

y_pred = model.predict_classes(X_train)
conf_train = confusion_matrix(y_train, y_pred)
print(conf_train)
model.evaluate(X_test,y_test)


# In[ ]:




