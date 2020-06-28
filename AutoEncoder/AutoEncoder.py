import glob
import importlib
import os
import random

import cv2

spam_spec = importlib.util.find_spec("tkinter")
matplotlib_installed = spam_spec is not None
if matplotlib_installed:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def main():
    name = 'hh113'
    train_ratio = .9
    input_folder = f"../output/{name}/tw_images/"
    list_files = glob.glob(input_folder + '*.png')

    dataset = []

    for file in list_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        # img = img.reshape(img.shape[0], img.shape[1],1)
        dataset.append(img)

    dataset = np.asarray(dataset, dtype='float32')

    print('Done loading dataset, shape=', np.shape(dataset))

    # SPLIT TRAIN - TEST
    TRAIN_BUF = int(dataset.shape[0] * train_ratio)
    data_train = dataset[:TRAIN_BUF]
    data_test = dataset[TRAIN_BUF:]

    data_train /= 255.
    data_test /= 255.

    latent_dim = 10
    epochs = 1000
    batch_size = 10

    ## BUILD THE MODEL
    model = AE_Model(input_width=width, input_height=height, latent_dim=latent_dim)

    # Model parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.MeanSquaredError()
    metric = tf.keras.metrics.RootMeanSquaredError()
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    checkpoint_path = f"../output/{name}/AutoEncoder_logs/checkpoint.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    model.compile(optimizer, loss=loss, metrics=[metric])

    if latest_checkpoint:
        model.load_weights(latest_checkpoint)

    model.fit(data_train, data_train, epochs=epochs, batch_size=batch_size, validation_data=(data_test, data_test),
              shuffle=True, callbacks=[es_callback]),  # save_model_callback])

    model.plot_history()

    # TEST THE ENCODE-DECODE OPERATION
    nb_test = 5

    x = np.asarray(random.choices(data_train, k=nb_test)).reshape((nb_test, height, width))

    z = model.predict(x).reshape((nb_test, height, width))
    print(z.shape)

    fig, ax = plt.subplots(nb_test, 2)

    ximg = []
    zimg = []
    for i in range(nb_test):
        ximg.append(cv2.resize(x[i], (1280, 1280), interpolation=cv2.INTER_AREA))
        zimg.append(cv2.resize(z[i], (1280, 1280), interpolation=cv2.INTER_AREA))

    # images = images.reshape((2, 1280, 1280))
    for axi, xi, zi in zip(ax, ximg, zimg):
        axi[0].set(xticks=[], yticks=[])
        axi[1].set(xticks=[], yticks=[])
        axi[0].imshow(xi)
        axi[1].imshow(zi)

    plt.show()

    # CLUSTERING

    tensor_dataset = tf.data.Dataset.from_tensor_slices(dataset).batch(len(dataset))
    encoded_points = []
    for d in tensor_dataset:
        z = model.encode(d)
        encoded_points += [list(x) for x in z.numpy()]

    encoded_points = np.asarray(encoded_points)

    n_clusters = silhouette_plots(encoded_points, display=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(encoded_points)
    silhouette_avg = silhouette_score(encoded_points, clusters)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    encoded_centers = kmeans.cluster_centers_

    tensor_encoded_centers = tf.data.Dataset.from_tensor_slices(encoded_centers).batch(len(encoded_centers))

    decoded_centers = []
    for d in tensor_encoded_centers:
        z = model.decode(d)
        decoded_centers += [list(x) for x in z.numpy()]

    decoded_centers = np.asarray(decoded_centers)

    fig, ax = plt.subplots(1, n_clusters)

    img_centers = []
    for i in range(n_clusters):
        img = decoded_centers[i]
        # img[img >= .2] = 1.
        # img[img < .2] = 0.
        img_centers.append(cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA))

    for axi, img in zip(ax.flat, img_centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(img)

    plt.title("Cluster Centers")

    plt.figure()
    plt.imshow(img_centers[1] - img_centers[0])

    plt.show()

    pass


def silhouette_plots(data, display=True):
    """
    Plot the differents silhoutte values
    :return:
    """

    range_n_clusters = [2, 3, 4, 5, 6]

    X = data

    optimal_n_clusters = 1
    avg_silhouette = 0

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        if silhouette_avg > avg_silhouette:
            avg_silhouette = silhouette_avg
            optimal_n_clusters = n_clusters

        if display:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters

            # Compute the silhouette scores for each sample

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
    if display:
        plt.show()

    print()
    print(f"Choose Number of Clusters : {optimal_n_clusters}")
    # print(f"Choosen Number of PCA Components : {min_n_components}")

    return optimal_n_clusters


class AE_Model(tf.keras.Model):
    def __init__(self, input_width, input_height, latent_dim, name='AE_Model'):
        super().__init__(name=name)
        self.input_width = input_width
        self.input_height = input_height

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.input_height, self.input_width), name='encoder_input'),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(self.input_width * self.input_height * 0.5),
                tf.keras.layers.Dense(self.input_width * self.input_height * 0.1),
                tf.keras.layers.Dense(self.latent_dim * 2),
                tf.keras.layers.Dense(self.latent_dim, activation="relu", name='encoder_output')
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,), name='decoder_input'),
                tf.keras.layers.Dense(self.latent_dim * 2),
                tf.keras.layers.Dense(self.input_width * self.input_height * 0.1),
                # tf.keras.layers.Dense(self.input_width * self.input_height * 0.5),
                tf.keras.layers.Dense(self.input_width * self.input_height, activation="sigmoid"),
                tf.keras.layers.Reshape((self.input_height, self.input_width), name='decoder_output')
            ]
        )

    def call(self, inputs):
        encoded_inputs = self.encoder(inputs)
        reconstructed = self.decoder(encoded_inputs)

        # reconstructed = tf.to_float(reconstructed > 0.5)

        return reconstructed

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def plot_history(self):
        history = self.history.history
        # PLOT THE RESULTS
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        plt.figure()
        plt.plot(history['mean_squared_error'])
        plt.plot(history['val_mean_squared_error'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def get_loss_error(self):
        return self.history.history['loss'][-1], self.history.history['mean_squared_error'][-1]


if __name__ == "__main__":
    main()
