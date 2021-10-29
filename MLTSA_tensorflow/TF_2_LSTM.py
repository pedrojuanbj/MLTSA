"""

This snippet of code is for the set of functions we will use for the LSTM architecture to try on the different data.

Note this is built on TensorFlow 2 code and may not work on earlier versions.

In this snippet the architecture is built on the sequential "build" of TF.

"""
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D



class build_LSTM():

    def __init__(self, n_steps, n_features, n_labels, type="vanilla", n_seq=None):

        self.type = type
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_steps = n_steps

        if type == "vanilla":
            print("Building LSTM")
            clf = self.vanilla()
        elif type == "stacked":
            print("Building Stacked LSTM")
            clf = self.stacked()
        elif type == "bidirectional":
            print("Bidirectional LSTM")
            clf = self.bidirectional()
        elif type == "CNNLSTM":
            print("Convolutional Neural Network - LSTM")
            print("Carefull, this implementation needs an input with the shape of")
            print("[samples, subsequences, timesteps, features]")
            clf = self.CNNLSTM()
        elif type == "ConvLSTM":
            print("Building ConvLSTM ")
            print("Carefull, this implementation needs an input with the shape of")
            print("[samples, timesteps, rows, columns, features]")
            if n_seq == None:
                print("Please specify the n_seq of your input")
            clf = self.ConvLSTM(n_seq)

        self.model = clf

        return

    def vanilla(self, n_units=100, dropout=0.1):

        model = Sequential()

        # Add LSTM layer with 100 units
        model.add(LSTM(n_units, input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        print("Multilabel classification model")
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    def stacked(self, n_units=100, dropout=0.1, n_layers=5):

        model = Sequential()

        # Add LSTM layer with 100 units
        for n in range(n_layers):
            model.add(LSTM(n_units, return_sequences=True, input_shape=(self.n_steps, self.n_features)))
        model.add(LSTM(n_units, input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        print("Multilabel classification model")
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model

    def bidirectional(self, n_units=100, dropout=0.1):
        model = Sequential()

        # Add LSTM layer with 100 units
        model.add(Bidirectional(LSTM(n_units, activation='relu'), input_shape=(self.n_steps, self.n_features)))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        print("Multilabel classification model")
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model

    def CNNLSTM(self, n_units=100, dropout=0.1, pool_size=2, n_interpretations=50):
        model = Sequential()

        # Add LSTM layer with 100 units
        model.add(TimeDistributed(Conv1D(filters=n_interpretations, kernel_size=1, activation='relu'),
                                  input_shape=(None, self.n_steps, self.n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=pool_size)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(n_units, activation='relu'))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        print("Multilabel classification model")
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model

    def ConvLSTM(self, n_seq, dropout=0.1, n_interpretations=50):
        model = Sequential()

        # Add LSTM layer with 100 units
        model.add(ConvLSTM2D(filters=n_interpretations, kernel_size=(1, 2), activation='relu',
                             input_shape=(n_seq, 1, self.n_steps, self.n_features)))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        print("Multilabel classification model")
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model
