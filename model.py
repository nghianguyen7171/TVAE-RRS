from common import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import torch
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix

from keras.layers import Input, merge, LSTM, Dense, SimpleRNN, Masking, Bidirectional, Dropout, concatenate, Embedding, TimeDistributed, multiply, add, dot, Conv2D
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from keras import regularizers, callbacks
from keras.layers.core import *
from keras.models import *
import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.backend.clear_session()

# params
seed = 42
num_folds = 5
scoring = "roc_auc"
batch_size = 1028

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)

seed_everything(seed)

#Loss
from tensorflow.keras import backend as K

smooth  = 1.
epsilon = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# dice_coef_loss
def dice_coef_multi(y_true, y_pred):
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])

    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_multi

def dice_coef_multi_loss(y_true, y_pred):
    return 1.0 - dice_coef_multi(y_true, y_pred)
# dice_coef_multi_loss

def mean_acc(y_true, y_pred):
    y_true_label = K.argmax(y_true, axis = 1)
    y_pred_label = K.argmax(y_pred, axis = 1)
    cm = tf.math.confusion_matrix(y_true_label, y_pred_label)
    cm_norm = cm / tf.reshape(tf.reduce_sum(cm, axis = 1), (-1, 1))
    zero_pos = tf.where(tf.math.is_nan(cm_norm))
    n_zero   = tf.shape(zero_pos)[0]
    cm_norm  = tf.tensor_scatter_nd_update(cm_norm, zero_pos, tf.zeros(n_zero, dtype=tf.double))
    mean_acc_val = tf.reduce_mean(tf.linalg.diag_part(cm_norm))
    return mean_acc_val

metrics = ["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()]
loss_fn = ["categorical_crossentropy", dice_coef_multi_loss] # "categorical_crossentropy",
optimizer_fn = tf.keras.optimizers.Adam(learning_rate=0.0001)
weights = None



# TVAE 

# Define the Sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Define the VAE model
def build_VAE(input_shape, latent_dim=8):
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(100, activation='tanh', return_sequences=True)(encoder_inputs)
    x = layers.LSTM(50, activation='tanh', return_sequences=True)(x)
    x = layers.LSTM(25, activation='tanh')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    # LSTM Decoder
    decoder_lstm_3 = layers.LSTM(25, activation='tanh', return_sequences=True)
    x_decoded = decoder_lstm_3(tf.expand_dims(z, axis=1))

    decoder_lstm_2 = layers.LSTM(50, activation='tanh', return_sequences=True)
    x_decoded = decoder_lstm_2(x_decoded)

    decoder_lstm_1 = layers.LSTM(100, activation='tanh', return_sequences=True)
    x_decoded = decoder_lstm_1(x_decoded)

    # Clf decoder
    decoder1 = layers.Dense(8, activation='relu')(z)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(64, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(32, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(16, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.1)(decoder1)

    decoder_out = layers.Dense(2, activation='sigmoid')(decoder1)

    # Build the VAE model
    VAE_clf = tf.keras.Model(inputs=encoder_inputs, outputs=decoder_out)

    # VAE loss
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    reconstruction_loss = tf.keras.losses.mean_squared_error(K.flatten(encoder_inputs), K.flatten(x_decoded))
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    VAE_clf.add_loss(vae_loss)

    # Compile the VAE model
    VAE_clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["acc"],
        run_eagerly=True,
    )

    return VAE_clf


# Kwon et al. model¶

def build_kwon_RNN(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(100, activation='tanh', dropout=0.2, return_sequences=True)(inputs)
    x = layers.LSTM(50, activation='tanh', dropout=0.2, return_sequences=True)(x)
    x = layers.LSTM(25, dropout=0.1)(x)
    out = layers.Dense(2, activation='sigmoid')(x)
    model_kwon = tf.keras.Model(inputs=inputs, outputs=out)
    model_kwon.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_kwon

# Attention blocks used in DEWS
def attention_block(inputs_1, num):
    # num is used to label the attention_blocks used in the model

    # Compute eij i.e. scoring function (aka similarity function) using a feed forward neural network
    v1 = Dense(10, use_bias=True)(inputs_1)
    v1_tanh = Activation('relu')(v1)
    e = Dense(1)(v1_tanh)
    e_exp = Lambda(lambda x: K.exp(x))(e)
    sum_a_probs = Lambda(lambda x: 1 / K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(e_exp)
    a_probs = multiply([e_exp, sum_a_probs], name='attention_vec_' + str(num))

    context = multiply([inputs_1, a_probs])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    return context


# Shamount et al
def build_shamount_Att_BiLSTM(input_shape):
    inputs = keras.Input(shape=input_shape)
    enc = Bidirectional(
        LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'),
        'ave')(inputs)
    dec = attention_block(enc, 1)
    dec_out = Dense(5, activation='relu')(dec)
    dec_drop = Dropout(0.2)(dec_out)
    out = Dense(2, activation='sigmoid')(dec_drop)
    model_shamount = tf.keras.Model(inputs=inputs, outputs=out)
    model_shamount.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_shamount


# Define the DCNN model
def build_dcnn(input_shape):
    # Input layer
    input_layer = layers.Input(shape=input_shape)

    # Convolutional layers
    conv1 = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
    maxpool1 = layers.MaxPooling1D(pool_size=2)(conv1)
    conv2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(maxpool1)
    # print(conv2.shape)
    maxpool2 = layers.MaxPooling1D(pool_size=2)(conv2)

    # Flatten layer
    flatten = layers.Flatten()(maxpool2)

    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout1 = layers.Dropout(0.5)(dense1)
    output_layer = layers.Dense(2, activation='sigmoid')(dropout1)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer = optimizer_fn,       
        loss      = loss_fn, 
        metrics   = metrics,
        run_eagerly = True,
    )

    return model