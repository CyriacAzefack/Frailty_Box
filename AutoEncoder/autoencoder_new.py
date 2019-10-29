import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.models import Model

sns.set_style('darkgrid')


def main():
    name = 'hh113'
    train_ratio = .8
    input_folder = f"../output/{name}/tw_images/"
    list_files = glob.glob(input_folder + '*.png')

    dataset = []

    width = height = 0
    for file in list_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        # img = img.reshape(img.shape[0], img.shape[1],1)
        dataset.append(img.flatten())

    dataset = np.asarray(dataset, dtype='float32')

    print('Done loading dataset, shape=', np.shape(dataset))

    # SPLIT TRAIN - TEST
    TRAIN_BUF = int(dataset.shape[0] * train_ratio)
    data_train = dataset[:TRAIN_BUF]
    data_test = dataset[TRAIN_BUF:]

    data_train /= 255.
    data_test /= 255.

    # BUILD THE AUTOENCODER NN
    input_dim = height * width
    encoding_dim = 40
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # TRAIN THE NN
    nb_epoch = 1000
    batch_size = 100

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=False)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history = autoencoder.fit(data_train, data_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True,
                              validation_data=(data_test, data_test),
                              verbose=1,
                              callbacks=[checkpointer, tensorboard]).history
    # autoencoder = load_model('model.h5')

    # PLOT THE RESULTS
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    for j in range(10):
        x = data_test[j].reshape(1, height * width)

        z = autoencoder.predict(x).reshape(height, width)
        print(z.shape)

        fig, ax = plt.subplots(1, 2)

        images = [x.reshape(height, width), z]
        #
        for i in range(2):
            img = images[i]
            img[img >= .2] = 1.
            img[img < .2] = 0.
            images[i] = cv2.resize(img, (1280, 1280), interpolation=cv2.INTER_AREA)
        #
        images = np.asarray(images)
        # images = images.reshape((2, 1280, 1280))
        for axi, img in zip(ax.flat, images):
            axi.set(xticks=[], yticks=[])

            axi.imshow(img)

        plt.show()


if __name__ == "__main__":
    main()
