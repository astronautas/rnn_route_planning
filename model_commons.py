from tensorflow.keras.layers import (GRU, LSTM, Activation, Dense, Embedding,
                                     Flatten, TimeDistributed, Lambda)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

import numpy as np

historical = None

def categorical_crossentropy_discounted_loops(yTrue, yPred):
    res = categorical_crossentropy(yTrue, yPred)

    # # with tf.get_default_session().as_default() as df:
    # #     res_eval = res.eval()
    
    # occ_tensor = tf.map_fn(lambda timesteps: tf.map_fn(lambda timestep_pred: 
    #                         timestep_prediction_coordinates[timestep_pred._id], timesteps), yPred)

    # range = res.shape[1]

    # # Add to each sample of a batch, the loss of a loop
    # #occ_tensor = tf.zeros_like(res)
    # # occ_tensor = tf.Variable([], tf.float32)
    # return tf.add(res, occ_tensor)
    historical = yPred
    return res

def softmax_with_temp(x, axis=-1, temperature=1.2):
    return K.softmax(tf.divide(x, temperature))

def get_model(batch_size, timesteps, X_dim, Y_dim, activation):
    model = Sequential()
    model.add(GRU(use_bias=False, units=128, dropout=0.7, recurrent_dropout=0.7, batch_input_shape=(batch_size, timesteps, X_dim), 
                    return_sequences=True, stateful=False))
    model.add(GRU(use_bias=False, units=128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True, stateful=False))
    model.add(GRU(use_bias=False, units=128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, stateful=False))
    model.add(TimeDistributed(Dense(Y_dim, activation=activation)))

    opt = RMSprop(lr=0.0008, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])

    return model