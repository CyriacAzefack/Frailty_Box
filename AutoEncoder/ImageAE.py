from __future__ import absolute_import, division, print_function, unicode_literals

import glob
# import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import PIL
# import imageio
from IPython import display

from Data_Drift.Drift_v2 import LogClustering

tf.compat.v1.enable_eager_execution()


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


# @tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


# @tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


class CVAE(tf.keras.Model):
    """
    Convolutional Variational AutoEncoder
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.name_model = 'CVAE'
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


if __name__ == '__main__':
    name = 'hh113'
    train_ratio = .9
    input_folder = f"../output/{name}/Daily_Images/"
    list_files = glob.glob(input_folder + '*.png')

    dataset = []
    width = LogClustering.LogClustering.WIDTH
    height = LogClustering.LogClustering.HEIGHT

    for file in list_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        # img = img.reshape(img.shape[0], img.shape[1],1)
        dataset.append(img)

    nb_features = width * height
    dataset = np.asarray(dataset, dtype='float32')
    dataset = dataset.reshape((-1, height, width, 1))

    print('Done loading dataset, shape=', np.shape(dataset))

    train_size = int(len(dataset) * train_ratio)
    train_images = dataset[:train_size]
    test_images = dataset[train_size:]

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    #  Normalizing the images to the range of [0., 1.]
    train_images /= 255
    test_images /= 255

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    TRAIN_BUF = train_size
    TEST_BUF = len(dataset) - TRAIN_BUF
    BATCH_SIZE = 200

    # Shuffle Dataset

    epochs = 200
    latent_dim = 20
    num_examples_to_generate = 16

    optimizer = tf.keras.optimizers.Adam(1e-4)

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim=latent_dim)

    generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_images:
            compute_apply_gradients(model, train_x.reshape(1, 28, 28, 1), optimizer)
        end_time = time.time()

        if epoch % 10 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_images:
                loss(compute_loss(model, test_x.reshape(1, 28, 28, 1)))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            generate_and_save_images(
                model, epoch, random_vector_for_generation)

    # print(model.summary())
    fig, ax = plt.subplots(1, 2)

    x = test_images[0].reshape(1, 28, 28, 1)
    y = model.encode(x)
    z = model.decode(y).numpy().reshape(28, 28)
    print(z.shape)

    images = [x.reshape(28, 28), z]

    #
    images = np.asarray(images)
    images = images.reshape((2, 28, 28))
    for axi, img in zip(ax.flat, images):
        axi.set(xticks=[], yticks=[])

        axi.imshow(img, interpolation='nearest', cmap='gray')

    plt.show()
