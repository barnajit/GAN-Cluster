from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

import scipy

from scipy.misc import imsave

from PIL import Image
import time

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #Try clustering with 3 cluster
            cluster_idx = np.random.randint(0,3, 1)

            # Sample noise and generate a batch of new images
            #if cluster_idx == 0:
             #   noise = np.random.normal(0, 0.33, (batch_size, self.latent_dim))
            #elif cluster_idx == 1:
             #   noise = np.random.normal(0.33, 0.67, (batch_size, self.latent_dim))
            #else:
             #   noise = np.random.normal(0.67, 1, (batch_size, self.latent_dim))

            noise = self.rand_gen(cluster_idx, batch_size)

            
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, 0.33, 0)
                self.save_imgs(epoch, 0.67, 1)
                self.save_imgs(epoch, 1, 2)

    def save_imgs(self, epoch, cluster_range, cluster_idx):
        r, c = 5, 5
        #noise = np.random.normal(cluster_range - 0.33, cluster_range, (r * c, self.latent_dim))
        noise = self.rand_gen(cluster_idx, r * c)

        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5


        #fig, axs = plt.subplots(r, c)
        #cnt = 0
        #for i in range(r):
            #for j in range(c):
           #     axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
          #      axs[i,j].axis('off')
         #       cnt += 1
        #fig.savefig("mnist_%d.png" % (epoch+cluster_idx))
        #plt.close()
       
        n = 5
        margin = 10
        width = n * 28 + (n - 1) * margin
        height = n * 28 + (n - 1) * margin
        stitched_filters = np.zeros((width, height), dtype=np.uint8)
        k=0
        for i in range(n):
            for j in range(n):
            
               img11 = gen_imgs[k, :,:,0]
               #img1=img11[:,:,np.newaxis]
                       
               stitched_filters[(28 + margin) * i: (28 + margin) * i + 28,
                         (28 + margin) * j: (28 + margin) * j + 28] = img11 * 255
               k=k+1

        imsave('mnist2_%d.png' % (epoch+cluster_idx), stitched_filters[:,:])
  



        
        
    def rand_gen(self,cluster_idx, batch_size):
        noise1 = np.random.normal(0, 0.2, (batch_size, 33))
        noise2 = np.random.normal(0.4, 0.6,(batch_size, 33))
        noise3 = np.random.normal(0.8, 1, (batch_size, 34))
        if cluster_idx == 0:  
          a        = np.concatenate((noise1, noise2), axis=1)
          noise    = np.concatenate((a, noise3), axis=1)
        elif cluster_idx == 1:
          a        = np.concatenate((noise2, noise3), axis=1)
          noise    = np.concatenate((a, noise1), axis=1)
        else:
          a        = np.concatenate((noise3, noise1), axis=1)
          noise    = np.concatenate((a, noise2), axis=1)
        
        return noise



if __name__ == '__main__':
    dcgan = DCGAN() 
    dcgan.train(epochs=40001, batch_size=32, save_interval=1000)
