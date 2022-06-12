import os
from ..Predictor import utils
import random
import pickle
from ..AdaBoost.FFNN import FFNN
from keras.models import load_model
from ..AdaBoost.AdaBoost import AdaBoost
from ..PredictorStrategy.networks import *
from ..PredictorStrategy.modelObject import *
from ..PredictorStrategy.PredictorInterface import PredictorInterface


class AdaBoostStrategy(PredictorInterface):

    def __init__(self):
        self.model = None
        self.method = None
        self.output_size = None
        self.device = None
        self.folder_models = None
        self.nr_models = 10

    def defineModel(self, device, nbands, windowSize, outputSize=1, method='AdaBoost'):
        """Override model declaration method"""
        self.method = method
        self.output_size = outputSize
        self.device = device

        # Training parameters
        lr = 0.01
        hiddennodes = 100
        epochs = 500
        delta = 1.5
        boosting_type = ""

        self.model = AdaBoostObject(self.nr_models, lr, hiddennodes, epochs, delta, boosting_type)

    def trainModel(self, trainx, train_y, batch_size, device, epochs, filepath, printProcess, beta_, yscale):
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        random.seed(7)

        # Shuffle
        indexes = np.arange(len(trainx))  # Prepare list of indexes for shuffling
        np.random.shuffle(indexes)
        trainx = trainx[indexes]
        train_y = train_y[indexes]

        # Vectorize data (4-D to 1-D)
        trainx = trainx.transpose((0, 3, 4, 1, 2))
        trainx = np.reshape(trainx, (trainx.shape[0] * trainx.shape[1] * trainx.shape[2] * trainx.shape[3],
                                     trainx.shape[4]))
        train_y = np.reshape(train_y, (train_y.shape[0] * train_y.shape[1] * train_y.shape[2]))
        # Remove repetitions
        trainx, kept_indices = np.unique(trainx, axis=0, return_index=True)
        train_y = train_y[kept_indices]

        # Set the folder name where the individual models will be saved
        self.folder_models = os.path.dirname(filepath) + "//models" + filepath[-1]
        if not os.path.exists(self.folder_models):
            os.mkdir(self.folder_models)

        # Start training
        final_predictions = []
        self.model.ab = AdaBoost(trainx, train_y, self.model.nr_models, delta_threshold=self.model.delta,
                                 learning_rate=self.model.lr, hidden_layer=self.model.hiddennodes,
                                 epochs=self.model.epochs)
        self.model.ab.set_weights(np.ones(trainx.shape[0]) / trainx.shape[0])
        self.model.ab.set_probabilities(np.divide(self.model.ab.get_weights(), np.sum(self.model.ab.get_weights())))
        final_predictions.append(self.model.ab.boost(type=self.model.boosting_type, models_path=self.folder_models))

        # Save model
        with open(filepath, 'wb') as fil:
            pickle.dump(self.model.ab.model_weights, fil)

    def predictSamples(self, datasample, means, stds, batch_size, device):
        """Predict yield values (in patches or single values) given a batch of samples."""
        valxn = utils.applyMinMaxScale(datasample, means, stds)[:, 0, :, 0, 0]

        return np.array([self.model.ab.predict(valxn)]).transpose((1, 0, 2))

    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, device, MC_samples):
        """Predict yield probability distributions given a batch of samples using MCDropout"""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)[:, 0, :, 0, 0]

        # Run model
        y_pred, std = self.model.ab.predictUncertainty(valxn)

        return y_pred[:, 0], std

    def loadModelStrategy(self, path):
        # Set AB object
        self.model.ab = AdaBoost(None, None, self.model.nr_models, delta_threshold=self.model.delta,
                                 learning_rate=self.model.lr, hidden_layer=self.model.hiddennodes,
                                 epochs=self.model.epochs)
        # Load weight models
        with open(path, 'rb') as f:
            w = pickle.load(f)
        self.model.ab.model_weights = w

        # Load the individual models of the ensemble
        self.folder_models = os.path.dirname(path) + "//weakModels"
        self.model.ab.weak_models = []  # Reset instance variable
        for m in range(self.nr_models):
            model = FFNN(None, learning_rate=None, batch_size=None)
            model.trainedModel = load_model(self.folder_models + "//model-" + str(m + 1) + ".h5")
            self.model.ab.weak_models.append(model)
