# This script contains classes that are used to store information about the models used for yield prediction such as
# the architecture of a neural network, its weights, the optimizer used for training, and the loss function.


class CNNObject:
    """Helper class used to store the main information of a CNN model."""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class AdaBoostObject:
    """Helper class used to store the main information of a AdaBoost model."""

    def __init__(self, nr_models, lr, hiddennodes, epochs, delta, boosting_type):
        self.nr_models = nr_models
        self.lr = lr
        self.hiddennodes = hiddennodes
        self.epochs = epochs
        self.delta = delta
        self.boosting_type = boosting_type
        self.ab = None


class SAEObject:
    """Helper class used to store the main information of a Stacked AutoEncoder model."""

    def __init__(self, model, final_layer, criterion, optimizer):
        self.network = model
        self.final_layer = final_layer
        self.criterion = criterion
        self.optimizer = optimizer
