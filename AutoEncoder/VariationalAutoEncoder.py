import glob
import logging
from typing import Any

import PIL
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display

# import heva_theme
# plt_heva_theme = pio.templates['heva_theme'].layout.colorway

tf.compat.v1.enable_eager_execution()
logger = logging.getLogger(__name__)


###### AUTO ENCODING ######

def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Compute log normal density
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def apply_gradients(optimizer, gradients, variables):
    """
    Apply gradient to variables
    """
    optimizer.apply_gradients(zip(gradients, variables))


def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def export_gif_file(anim_file, workspace):
    """
    Export pictures to create a GIF (to vosualize encoding process)
    """
    with imageio.get_writer(workspace + anim_file, mode='I') as writer:
        filenames = glob.glob(f'{workspace}image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


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


class AE(tf.keras.Model):
    """
    Class to define an autoencoder (regular or denoising).
    """

    def __init__(self):
        super().__init__()
        self.name_model = 'AE'

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables), loss

    def display_image(self, epoch_no):
        return PIL.Image.open(f'../results/{self.name_model}/{self.name_model}_image_at_epoch_{epoch_no}.png')

    def plot_loss(self, val_elbo_list: list, train_elbo_list: list, ref: str, show: bool = False):
        """
        Plot the loss function (elbo values), for training and validation set.
        Save image in '../results/{self.name_model}/' folder, with 'ref' in filename.

        :param val_elbo_list: list of elbo values of validation set
        :param train_elbo_list: list of elbo values of training set
        :param ref: ref of the iteration
        :param show: If True, plot the graph in notebook.
        """
        # Plot Elbo graph
        list_x = [x + 1 for x in range(len(val_elbo_list))]
        fig = plt.figure(figsize=(20, 5))
        plt.plot(list_x, val_elbo_list, c='red', label='Loss - validation')
        plt.plot(list_x, train_elbo_list, c='black', label='Loss - train')
        fig.patch.set_facecolor('white')
        plt.yscale('log')
        plt.xticks((np.arange(0, max(list_x), step=10)))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../results/{self.name_model}/Loss_{ref}_{self.name_model}.png')

        if show == True:
            plt.show()
        else:
            plt.close()

    def plot_training_images(self, dae_data_validation: Any, x_logit: np.array, epoch: int):
        """
        Plot image of validation input element and encoded/decoded output for comparison.

        :param dae_data_validation: validation data
        :param x_logit: array of decoded values
        :param epoch: the actual epoch
        """
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(dae_data_validation[0:self.num_examples_to_generate], cmap='gray')
        ax1.axis('off')

        ax2.imshow(x_logit[0:self.num_examples_to_generate], cmap='gray')
        ax2.axis('off')
        plt.savefig(f'../results/{self.name_model}/image_at_epoch_{self.name_model}_{epoch}.png')

    def train(self, optimizer: Any, train_dataset: Any, validation_dataset: Any, dae_data_validation: np.array,
              latent_dim: int, epochs: int, batch_size: int, nb_features: int, ref: str,
              early_stop_patience: int = 50, freq_print: int = 10, plot_test: bool = False, show: bool = False) -> Any:
        """
        Train autoencoder

        :param optimizer: tf optimizer
        :param train_dataset: tf dataset
        :param validation_dataset: tf dataset
        :param dae_data_validation: tf dataset
        :param latent_dim: size of the latent space
        :param epochs: number of epochs
        :param batch_size: number of element for each batch
        :param nb_features: number of features of input space
        :param ref: reference for export names
        :param early_stop_patience: number of iteration without improvement in validation set before stopping
        :param freq_print: frequency for image plotting
        :param plot_test: plot test images if True
        :param show: show training results during training if True
        :return: trained AE element

        """

        self.freq_print = freq_print
        self.batch_size = batch_size
        self.nb_features = nb_features

        self.num_examples_to_generate = min(self.batch_size, int(nb_features / 2))

        elbo_list = []
        train_elbo_list = []

        early_stop_test = 0
        loss_before = 1e10

        for epoch in range(1, epochs + 1):

            train_loss = tf.keras.metrics.Mean()

            test_vector_for_generation = []

            for train_x in train_dataset:
                gradients, loss = self.compute_gradients(train_x)
                apply_gradients(optimizer, gradients, self.trainable_variables)

                train_loss(self.compute_loss(train_x))

            loss = tf.keras.metrics.Mean()

            for test_x in validation_dataset:
                loss(self.compute_loss(test_x))

                if self.name_model == 'VAE':
                    mean, logvar = self.encode(test_x)
                    z = self.reparameterize(mean, logvar)
                else:
                    z = self.encode(test_x)

                test_vector_for_generation.append(z)

            elbo = loss.result()
            elbo_list.append(elbo)

            train_elbo = train_loss.result()
            train_elbo_list.append(train_elbo)

            # Early stop
            if elbo > loss_before:
                early_stop_test += 1
                if early_stop_test >= early_stop_patience:
                    print(f'early stop : {epoch}')
                    # load json and create model
                    json_file = open('model_encoder.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    self.inference_net = tf.keras.models.model_from_json(loaded_model_json)
                    self.inference_net.build(input_shape=(batch_size, nb_features))
                    json_file = open('model_decoder.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    self.generative_net = tf.keras.models.model_from_json(loaded_model_json)
                    self.generative_net.build(input_shape=(batch_size, latent_dim))

                    # load weights into new model
                    self.inference_net.load_weights("model_encoder.h5")
                    self.generative_net.load_weights("model_decoder.h5")
                    print("Loaded model from disk")
                    break
            else:
                early_stop_test = 0
                loss_before = elbo
                # serialize model to JSON
                model_json_encoder = self.inference_net.to_json()
                model_json_decoder = self.generative_net.to_json()
                with open("model_encoder.json", "w") as json_file:
                    json_file.write(model_json_encoder)
                with open("model_decoder.json", "w") as json_file:
                    json_file.write(model_json_decoder)
                # serialize weights to HDF5
                self.inference_net.save_weights("model_encoder.h5")
                self.generative_net.save_weights("model_decoder.h5")

            if epoch % freq_print == 0 and epoch > 0:
                display.clear_output(wait=False)
                x_logit = self.decode(test_vector_for_generation, apply_sigmoid=True)

                # Print
                if show == True:
                    print(f'Epoch: {epoch} \nTrain set Loss: {train_elbo}, \nValidation set Loss: {elbo}')

                # Plot image input and encoded/decoded
                if plot_test == True:
                    self.plot_training_images(dae_data_validation, x_logit, epoch)

                # Plot ELBO
                self.plot_loss(elbo_list, train_elbo_list, ref, show=show)

        return self


### VAE

class VAE(AE):
    """
    Class to define a variationnal autoencoder
    """

    def __init__(self, latent_dim, nb_features):
        super().__init__()
        self.latent_dim = latent_dim
        self.name_model = 'VAE'
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(nb_features,), dtype='float32'),
                # tf.keras.layers.Dense(int(nb_features * 0.5), activation='relu'),
                tf.keras.layers.Dense(int(nb_features * 0.25), activation='relu'),
                tf.keras.layers.Dense(4 * latent_dim),
                tf.keras.layers.Dense(latent_dim + latent_dim, activation="relu")
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.InputLayer(input_shape=(2 * latent_dim,)),
                tf.keras.layers.Dense(int(nb_features * 0.25), ),
                # tf.keras.layers.Dense(int(nb_features * 0.5), activation='relu'),
                tf.keras.layers.Dense(nb_features, activation="sigmoid"),
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100))
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        x_logit = tf.cast(x_logit, tf.float32)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

        logpx_z = -tf.reduce_sum(cross_ent)
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)

        logpx_z = tf.cast(logpx_z, tf.float32)
        logqz_x = tf.cast(logqz_x, tf.float32)
        logpz = tf.cast(logpz, tf.float32)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
