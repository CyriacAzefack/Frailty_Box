#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display

tf.enable_eager_execution()


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


class AE(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.name_model = 'AutoEncoder'
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(nb_features,), dtype='float32'),
                tf.keras.layers.Dense(int(nb_features * 0.5)),
                tf.keras.layers.Dense(int(nb_features * 0.25)),
                tf.keras.layers.Dense(latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(int(nb_features * 0.25)),
                tf.keras.layers.Dense(int(nb_features * 0.5)),
                tf.keras.layers.Dense(nb_features),
            ]
        )

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, z):
        return self.generative_net(z)

    def train(self, optimizer, train_dataset, test_dataset,
              latent_dim, epochs, batch_size, nb_features,
              freq_print=10):

        loss_list = []

        for epoch in range(1, epochs + 1):
            for train_x in train_dataset:
                start_time = time.time()
                gradients, loss = self.compute_gradients(train_x)
                apply_gradients(optimizer, gradients, model.trainable_variables)
                end_time = time.time()

            if epoch % freq_print == 0:

                test_vector_for_generation = []
                encoded_points = []

                loss = tf.keras.metrics.Mean()

                for test_x in test_dataset:
                    loss(self.compute_loss(test_x))
                    z = self.encode(test_x)
                    test_vector_for_generation.append(z)

                    encoded_points = encoded_points + [list(x) for x in z.numpy()]

                loss_value = -loss.result()
                loss_list.append(-loss_value)
                display.clear_output(wait=False)

                print(f"Epoch : {epoch}", end='\r')
                print('Epoch: {}, Test set LOSS: {}, '
                      'time elapse for current epoch {}'.format(epoch,
                                                                loss_value,
                                                                end_time - start_time))

                # Plot Elbo graph
                fig = plt.figure(figsize=(20, 5))
                plt.plot(loss_list, c='black')
                fig.patch.set_facecolor('white')
                plt.yscale('log')
                plt.grid(True)
                plt.show()

        return model

    def compute_loss(self, x):
        z = self.encode(x)
        x_logit = self.decode(z)
        x_logit = tf.cast(x_logit, tf.float32)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        cross_ent_sum = -tf.reduce_sum(cross_ent)
        return -tf.reduce_mean(cross_ent_sum)


# ## Data

# Loading the data
patients = pd.read_csv('../data/data_noise.csv', sep=',', engine='python')
dae_data = np.array(
    patients.iloc[:, 1:-1])  # dismiss id_sejour and id_cluster from the data we'll pass to the autoencoder
clust_dae = patients.iloc[:, -1]
dae_data = dae_data.astype('float32')
print('Done loading dataset, shape=', np.shape(dae_data))

# Loading the data

## TO DO 
# dae_data = 

TRAIN_BUF = int(dae_data.shape[0] * 0.8)
dae_data_train = dae_data[:TRAIN_BUF]
dae_data_test = dae_data[TRAIN_BUF:]

num_examples_to_generate = nb_features = dae_data.shape[1]

batch_size = 1000

epochs = 500
latent_dim = 20
optimizer = tf.keras.optimizers.Adam(1e-4)

train_dataset = tf.data.Dataset.from_tensor_slices(dae_data_train).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(dae_data_test).batch(batch_size)

# Model
model = AE(latent_dim)

# Train
model = model.train(optimizer, train_dataset, test_dataset,
                    latent_dim, epochs, batch_size, nb_features,
                    freq_print=10)
