from data_prep import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

class RNN:
    def __init__(self, sonnets, length=40, weightpath=''):
        self.clean_data(sonnets, length)
        
        model = tf.keras.Sequential()
        
        model.add(layers.LSTM(200, input_shape=(self.X.shape[1], self.X.shape[2]))) # NEED INPUT SHAPE AT THIS POINT
        model.add(layers.Dense(self.Y.shape[1], activation='softmax'))
        
        if weightpath != '':
            model.load_weights(weightpath)
            
        model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
                # Loss function to minimize
                loss=keras.losses.CategoricalCrossentropy(),
                # List of metrics to monitor
                metrics=['categorical_crossentropy'])
        
        self.model = model
        
    def clean_data(self, sonnets, length=40):
        x, y, vocab = get_sequences(sonnets)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        print('VOCAB: ' , self.vocab)
        # X needs to be of the form [samples, time steps, features]
        self.X = np.reshape(x, (len(x), length, 1))
        # want to normalize
        self.X = self.X/self.vocab_size
        # one hot encode the outputs
        self.Y = to_categorical(y)
        
    def fit(self, callbacks, batch_size=256, epochs=10):
        # fit the model
        self.model.fit(self.X, self.Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        
    # from keras documentation
    # used to set temperature
    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def predict(self, sequence, temperature=1.0):
        # sequence is characters right now, needs to be integers
        char_to_int = dict((c, i) for i,c in enumerate(self.vocab))
        int_to_char = dict((i,c) for i,c in enumerate(self.vocab))
        
        x = seq_to_ints(sequence, char_to_int)
        x = np.reshape(x, (1, len(sequence), 1))
        x = x / self.vocab_size
        
        predictions = self.model.predict(x, verbose=0)
        temperature_result = self.sample(predictions.flatten(), temperature=temperature)
        # print(predictions)
        result = int_to_char[temperature_result]
        
        return result

def setup_rnn(sonnets, length=40):
    rnn = RNN(sonnets, length, weightpath='weights.hdf5')
    
    # save the weights
    
    # weightpath = 'weights.hdf5'
    # checkpoint = ModelCheckpoint(weightpath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_lst = [checkpoint]
    # rnn.fit(callbacks_lst)

    return rnn
    