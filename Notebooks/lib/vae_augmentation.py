import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D ,Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')

# ----------------------VARIATIONAL AUTO ENCODER ----------------------------------------

def sample_z(args):
    """
    Sample from the encoded input.
    
    Parameters:
    args (tuple): Tuple of z_mean and z_log_var, both of shape (batch_size, latent_dim).
    
    Returns:
    numpy.ndarray: Sampled values from the encoded input, of shape (batch_size, latent_dim).
    """
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(z_log_var / 2) * eps

def build_encoder(input_shape, latent_dim):
    """
    Build the encoder model.
    
    Parameters:
    input_shape (tuple): Shape of the input data, of format (height, width, channels).
    latent_dim (int): Dimension of the latent space.
    
    Returns:
    tensorflow.python.keras.engine.training.Model: Encoder model, with input and outputs (z_mean, z_log_var, z).
    tuple: Shape of the intermediate convolutional layer, of format (batch_size, height, width, channels).
    """
    input_data = Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32,3, padding='same', activation='relu')(input_data)
    x = layers.Conv2D(64,3, padding='same', activation='relu')(x)
    conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='latent_mu')(x)   #Mean values of encoded input
    z_log_var = layers.Dense(latent_dim, name='latent_sigma')(x) 
    z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])
    return Model(input_data, [z_mean, z_log_var, z], name='encoder'),conv_shape

def build_decoder(conv_shape, latent_dim):
    """
    Build the decoder model.
    
    Parameters:
    conv_shape (tuple): Shape of the intermediate convolutional layer, of format (batch_size, height, width, channels).
    latent_dim (int): Dimension of the latent space.
    
    Returns:
    tensorflow.python.keras.engine.training.Model: Decoder model, with input and output.
    """
    decoder_input = Input(shape=(latent_dim, ), name='decoder_input')
    x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
    x = Reshape((conv_shape[1], conv_shape[2],conv_shape[3]))(x)
    x = Conv2DTranspose(64,3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32,3, padding='same', activation='sigmoid')(x)
    decoder_output = Conv2DTranspose(1, 3, activation="sigmoid", padding="same", name='decoder_output')(x)
    return Model(decoder_input, decoder_output, name='decoder')


class VAE(keras.Model):
    """
    Class representing the Variational Autoencoder Model.
    
    Parameters
    ----------
    encoder: keras.Model
        The encoder network.
    decoder: keras.Model
        The decoder network.
    kwargs: dict
        Additional keyword arguments passed to the parent class `keras.Model`.
    
    Properties
    ----------
    metrics: list
        List of Keras metrics to track, including the total loss, reconstruction loss, and KL divergence loss.
    
    Methods
    -------
    train_step(data)
        A single training step for the VAE model, computing the total loss as the sum of the reconstruction loss and the KL         divergence loss.
    
    Returns
    -------
    dict
        Dictionary containing the loss values for the total loss, reconstruction loss, and KL divergence loss.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name = "total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data,reconstruction),axis = (1,2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss =  tf.reduce_mean(tf.reduce_sum(kl_loss, axis = 1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }



def aug_data_vae(vae , points):
    """
    Generate augmented data using a variational autoencoder (VAE) model.
    
    Parameters:
    - vae (Model): A Keras model instance of a VAE.
    - points (np.array): A 2D array of shape (n, d) where n is the number of samples and d is the number of features.
    
    Returns:
    - augmented_data (np.array): A 4D array of shape (n, img_shape_y, img_shape_x, 1) where n is the number of samples and           img_shape_y and img_shape_x are the shapes of the images.
    """
    augmented_data = []
    augmented_data_stft = []
    
    # display a n*n 2d manifold of digits
    img_shape_y = 63
    img_shape_x = 251
    
    for i, sample in enumerate(points):
        z_sample =  np.array([sample])
        x_decoded = vae.decoder.predict(z_sample)
        augmented_data.append(x_decoded)
        stft  = x_decoded.reshape(img_shape_y, img_shape_x)
        augmented_data_stft.append(stft)
    
    augmented_data = np.expand_dims(np.array(augmented_data_stft), axis=3)
    
    return augmented_data

