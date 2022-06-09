from src.Predictor import utils
import random
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.PredictorStrategy.PredictorInterface import PredictorInterface


class RFStrategy(PredictorInterface):
    def __init__(self):
        self.model = None
        self.method = None
        self.output_size = None
        self.device = None
        self.folder_models = None
        self.nr_models = 50

    def defineModel(self, device, nbands, windowSize, outputSize=1, method='RF'):
        """Override model declaration method"""
        self.method = method
        self.output_size = outputSize
        self.device = device

    def trainModel(self, trainx, train_y, batch_size, device, epochs, filepath, printProcess, beta_, yscale):
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        random.seed(7)

        # Separate 90% of the data for training
        trainx = trainx[0:int(len(trainx) * 90 / 100), :, :, :, :]
        train_y = train_y[0:int(len(train_y) * 90 / 100), :, :]

        # Vectorize data (4-D to 1-D)
        trainx = trainx.transpose((0, 3, 4, 1, 2))
        trainx = np.reshape(trainx, (trainx.shape[0] * trainx.shape[1] * trainx.shape[2] * trainx.shape[3],
                                     trainx.shape[4]))
        train_y = np.reshape(train_y, (train_y.shape[0] * train_y.shape[1] * train_y.shape[2]))
        # Remove repetitions
        trainx, kept_indices = np.unique(trainx, axis=0, return_index=True)
        train_y = train_y[kept_indices]

        # Start training
        self.model = RandomForestRegressor(n_estimators=1000, random_state=0)
        self.model.fit(trainx, train_y)

        # Save model
        with open(filepath, 'wb') as fil:
            pickle.dump(self.model, fil)

    def predictSamples(self, datasample, means, stds, batch_size, device):
        """Predict yield values (in patches or single values) given a batch of samples."""
        valxn = utils.applyMinMaxScale(datasample, means, stds)[:, 0, :, 0, 0]

        return self.model.predict(valxn)

    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, device, MC_samples):
        """Predict yield probability distributions given a batch of samples"""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)[:, 0, :, 0, 0]

        # TODO: How to calculate prediction intervals with RF?

        # Calculate confidence intervals
        confidence_intervals = self.model.confidence_intervals(valxn, width=0.95)
        # Calculate width of CIs
        uncertainties = confidence_intervals[:, 1] - confidence_intervals[:, 0]

        return self.model.predict(valxn), np.array(uncertainties)

    def loadModelStrategy(self, path):
        # Load weight models
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
