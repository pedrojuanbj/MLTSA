"""

This snippet of code is for the set of functions we will use for the different Recurrent Neural Networks architectures
to try on the different data.

Note this is built on TensorFlow 2 code and may not work on earlier versions.

In this snippet the architecture is built on the sequential "build" of TF.

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, GRU


class build_MLP():
"""
Factory class to ensemble RNN models.
"""
    def __init__(self, n_steps, n_features, n_labels, type="vanilla"):
	"""
	initialize the RNN generator class

	:param n_steps: Number of steps of the input data, shape the input layer of the model
	:type n_steps: int
	:param n_features: Number of features of the input data, shape the input layer of the model
	:type n_features: int
	:param n_labels: Number of labels in the input data, shape the output layer of the model
	:type n_labels: int
	:param type: type of the RNN model to be ensembled, optional, default as 'vanilla'
	:type type: str
	"""
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
	"""
	Classic RNN model with one hidden layer

	:param n_units: Number of units of the hidden layer, optional, default as 100
	:type n_units: int
	:param dropout: Dropout rate of the dropout process, optional, default as 0.1
	:type dropout: float
	"""
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
	"""
	Gated recurrent unit RNN model


	:param n_units: Number of units of the hidden layer, optional, default as 100
	:type n_units: int
	:param dropout: Dropout rate of the dropout process, optional, default as 0.1
	:type dropout: float
	:param n_layers: Number of the hidden layer, optional, default as 5 #TODO: Check with pedro, this args is not called in the function here
	:type n_layers: int
	"""
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
