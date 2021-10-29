"""

This snippet of code is for the set of functions we will use for the different Recurrent Neural Networks architectures
to try on the different data.

Note this is built on TensorFlow 2 code and may not work on earlier versions.

In this snippet the architecture is built on the sequential "build" of TF.

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, GRU


class build_MLP():

    def __init__(self, n_steps, n_features, n_labels, type="vanilla"):

        self.type = type
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_steps = n_steps

        if type == "vanilla":
            print("Building simple RNN")
            model = self.vanilla()
        elif type == "GRU":
            print("Building GRU RNN")
            model = self.GRU()

        return model

    def vanilla(self, n_units=100, dropout=0.1):

        model = Sequential()

        # Add Simple RNN layer with 100 units
        model.add(SimpleRNN(n_units, input_shape=(self.n_steps, self.n_features), dropout=dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        if self.n_labels > 2:
            print("Multilabel classification model")
            model.add(Dense(self.n_labels, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer='adam')
        else:
            print("Binary Classification model")
            model.add(Dense(1, activation='softmax'))
            model.compile(loss="binary_crossentropy", optimizer='adam')

        print(model.summary())

        return model

    def GRU(self, n_units=100, dropout=0.1, n_layers=5):

        model = Sequential()

        # Add GRU layer with 100 units
        model.add(GRU(n_units, input_shape=(self.n_steps, self.n_features), dropout=dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        if self.n_labels > 2:
            print("Multilabel classification model")
            model.add(Dense(self.n_labels, activation='softmax'))
            model.compile(loss="categorical_crossentropy", optimizer='adam')
        else:
            print("Binary Classification model")
            model.add(Dense(1, activation='softmax'))
            model.compile(loss="binary_crossentropy", optimizer='adam')

        print(model.summary())

        return model