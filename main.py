from tkinter import Tk, Menu, Label
import matplotlib.pyplot as plt
import numpy as np

from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image


from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import (Dense, Flatten, Reshape, Input, Lambda,
                          BatchNormalization, Dropout, Conv2D, LeakyReLU,
                          Conv2DTranspose)


def dropout_and_batch(x):
    return Dropout(0.3)(BatchNormalization()(x))


def noiser(args):
    global z_mean, z_log_var
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    N = K.random_normal(shape=(batch, hidden_dim), mean=0., stddev=1.0)

    return K.exp(z_log_var / 2) * N + z_mean


def open_image():
    filename = askopenfilename()
    FirstImage = Image.open(filename)

    photo = ImageTk.PhotoImage(FirstImage)
    MainWindows.geometry('{0}x{1}'.format(int(FirstImage.size[0] * 2 + 10), int(FirstImage.size[1])))


# Параметры изображений
IMG_W = 64
IMG_H = 64
CHANNELS = 3

# Латентное пространство
hidden_dim = 32


# Енкодер
input_img = Input((IMG_W, IMG_H, CHANNELS))
x = Conv2D(32, strides=2, kernel_size=5, activation="relu", padding="same")(input_img)
x = BatchNormalization()(x)
x = Conv2D(64, strides=2, kernel_size=5, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(128, strides=2, kernel_size=5, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(256, strides=2, kernel_size=5, activation="relu", padding="same")(x)
x = Flatten()(x)

z_mean = Dense(hidden_dim, name="z_mean")(x)
z_log_var = Dense(hidden_dim, name="z_log_var")(x)

h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean, z_log_var])

# Декодер
input_dec = Input(shape=(hidden_dim,))
d = Dense(4*4*512, activation='relu')(input_dec)
d = dropout_and_batch(d)
d = Reshape(target_shape=(4, 4, 512))(d)
d = Conv2DTranspose(512, strides=2, kernel_size=5, activation="relu", padding="same")(d)
d = dropout_and_batch(d)
d = Conv2DTranspose(256, strides=2, kernel_size=5, activation="relu", padding="same")(d)
d = dropout_and_batch(d)
d = Conv2DTranspose(128, strides=2, kernel_size=5, activation="relu", padding="same")(d)
d = dropout_and_batch(d)
d = Conv2DTranspose(64, strides=2, kernel_size=5, activation="relu", padding="same")(d)
d = dropout_and_batch(d)
d = Conv2DTranspose(32, strides=1, kernel_size=5, activation="relu", padding="same")(d)
d = dropout_and_batch(d)
decoded = Conv2DTranspose(3, strides=1, kernel_size=5, activation="sigmoid", padding="same")(d)

encoder = keras.Model(input_img, h, name='encoder')
decoder = keras.Model(input_dec, decoded, name='decoder')
generator = keras.Model(input_img, decoder(encoder(input_img)), name="generator")

discriminator = keras.Sequential(name="discriminator")
discriminator.add(Conv2D(64, 4, strides=2, padding='same', input_shape=[IMG_W, IMG_H, CHANNELS]))
discriminator.add(LeakyReLU())
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, 4, strides=2, padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(256, 4, strides=2, padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(512, 4, strides=2, padding='same'))
discriminator.add(LeakyReLU())
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation="sigmoid"))

generator.load_weights("C:/Users/Degur/Desktop/HAKATON_2.0/weights/gen_weigths.h5")

MainWindows = Tk()
MainWindows.title('Test NN')
MainWindows.geometry("500x500")


def get_res_NN(img, model):
    img = img.resize((64, 64))
    img = np.array(img) / 255

    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)
    predict = predict.reshape(64, 64, 3)
    plt.imshow(predict)
    plt.show()


def open_image():
    global generator

    filename = askopenfilename()
    img = Image.open(filename)
    img = img.resize((256, 256))

    photo = ImageTk.PhotoImage(img)
    get_res_NN(img, generator)
    view = Label(MainWindows, image=photo)
    view.pack(side="bottom", fill="both")

    MainWindows.mainloop()


menu = Menu(MainWindows)
MainWindows.config(menu=menu)
submenu = Menu(menu, tearoff=False)
menu.add_cascade(label="Меню", menu=submenu)
submenu.add_command(label="Загрузить изображение", command=open_image)
MainWindows.mainloop()