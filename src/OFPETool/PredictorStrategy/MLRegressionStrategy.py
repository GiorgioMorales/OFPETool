from ..Predictor import utils
import random
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from ..PredictorStrategy.PredictorInterface import PredictorInterface


class MLRegressionStrategy(PredictorInterface):
    def __init__(self):
        self.model = None
        self.method = None
        self.output_size = None
        self.device = None
        self.folder_models = None
        self.nr_models = 50

    def defineModel(self, device, nbands, windowSize, outputSize=1, method='MLRegression'):
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
        self.model = LinearRegression()
        self.model.fit(trainx, train_y)

        # Calculate training error
        y_pred = self.model.predict(trainx)

        # Calculate MSE of the estimator (n - p dof)
        MSE = np.sum((train_y - y_pred) ** 2) / (len(y_pred) - trainx.shape[1])

        # Add uncertainties of coefficients to model object
        self.model.MSE = MSE
        self.model.XTX = np.linalg.inv(np.matmul(trainx.transpose(), trainx))

        # Save model
        with open(filepath, 'wb') as fil:
            pickle.dump(self.model, fil)

    def predictSamples(self, datasample, means, stds, batch_size, device):
        """Predict yield values (in patches or single values) given a batch of samples."""
        valxn = utils.applyMinMaxScale(datasample, means, stds)[:, 0, :, 0, 0]

        return self.model.predict(valxn)

    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, device, MC_samples):
        """Predict yield probability distributions given a batch of samples using MCDropout"""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)[:, 0, :, 0, 0]

        # Calculate standard error of the prediction SE
        uncertainties = []
        # self.model.MSE = np.sum((out - np.reshape(target, (len(target),))) ** 2) / (len(target))
        for n in range(valxn.shape[0]):
            SE = np.sqrt(self.model.MSE +
                         np.abs(self.model.MSE * (np.matmul(valxn[n, :], np.matmul(self.model.XTX, valxn[n, :])))))
            uncertainties.append(SE)

        return self.model.predict(valxn), np.array(uncertainties)

    def loadModelStrategy(self, path):
        # Load weight models
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
