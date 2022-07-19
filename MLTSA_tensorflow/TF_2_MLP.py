"""

This snippet of code is for the set of functions we will use for the basic Multi-Layer Perceptron architecture to try
on the different data.

Note this is built on TensorFlow 2 code and may not work on earlier versions.

In this snippet the architecture is built on the sequential "build" of TF.

"""

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


class build_MLP(object):
"""
build_MLP A factory class that ensemble different Multi Layer Perceptron models
"""
    def __init__(self, n_steps, n_features, n_labels, type="vanilla"):
	"""
	initialize the build_MLP class

	:param n_steps: Number of steps of input trajectory, shape the input layer
	:type n_steps: int
	:param n_features: Number of features of the input data, shape the input layer
	:type n_features: int
	:param n_labels: Number of labels of the input data, shape the output layer
	:type n_labels: int
	:param type: type of the models to be built, optional, default as 'vanilla'
	:type type: str
	"""
        self.type = type
        self.n_labels = n_labels
        self.n_features = n_features
        self.n_steps = n_steps

        if type == "vanilla":
            print("Building Simple MLP")
            clf = self.vanilla()
        elif type == "deep":
            print("Building Stacked MLP")
            clf = self.deep()

        self.model = clf
        return

    def vanilla(self, n_units=100, dropout=0.1):
	"""
	classic model, also called 'vanilla'

	:param n_units: Number of nerons of the hidden layer, optional, default as 100
	:type n_units: int
	:param dropout: Dropout rate of the dropout process, optional, default as 0.1
	:type dropout: float
	"""

        model = Sequential()

        # Add Dense layer with 100 units
        model.add(Dense(n_units, input_shape=(self.n_features,), activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(n_units, activation="relu"))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model

    def deep(self, n_units=100, dropout=0.1, n_layers=5):
	"""
	deep model with multiple hidden layers with the same numer of nerons at each layer

	:param n_units: Number of the nerons of each hidden layer, optional, default as 100
	:type n_units: int
	:param dropout: Dropout rate of the dropout process, optional, default as 0.1
	:type dropout: float
	:param n_layers: Number of the hidden layers emsembled by perceptrons, optional, default as 5
	:type n_layers: int
	"""
        model = Sequential()

        # Add Dense layer with 100 units
        model.add(Dense(n_units, input_shape=(self.n_features,), activation="relu"))
        model.add(Dropout(dropout))
        for n in range(n_layers):
            model.add(Dense(n_units, activation="relu"))
        model.add(Dropout(dropout))

        # Add output layer and
        # Compile with the loss function and optimizer
        model.add(Dense(self.n_labels, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


        print(model.summary())

        return model
