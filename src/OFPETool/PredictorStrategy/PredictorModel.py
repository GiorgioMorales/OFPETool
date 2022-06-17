import sys
from ..PredictorStrategy.RFStrategy import RFStrategy
from ..PredictorStrategy.GAMStrategy import GAMStrategy
from ..PredictorStrategy.SpatialCNNStrategy import SpatialCNNStrategy
from ..PredictorStrategy.MLRegressionStrategy import MLRegressionStrategy
from ..PredictorStrategy.BayesRegressionStrategy import BayesRegressionStrategy


class PredictorModel:

    def __init__(self, model, device, nbands, window_size, output_size):
        """Create a ML object used to predict yield maps.
        @param model: Type of regression model. Options: 'Hyper3DNet', 'Hyper3DNetSSIM',
                      'Hyper3DNetSSIM-MSE-EXP', 'AdaBoost', 'Russello', 'CNNLF', 'SAE', and 'MLRegression'.
        @param device: Type of device used for training (Used for the CNNs).
        @param nbands: Number of input channels.
        @param window_size: Window size.
        @param output_size: Size of the output patch.
        """
        # Set instance variables
        self.device = device
        self.nbands = nbands
        self.window_size = window_size
        self.output_size = output_size

        # Set the CrossValStrategy that the model will use
        if model in ['Hyper3DNet', 'Hyper3DNetQD', 'Russello', 'CNNLF']:
            self.strategy = SpatialCNNStrategy()
        elif model == 'RF':
            self.strategy = RFStrategy()
        elif model == 'GAM':
            self.strategy = GAMStrategy()
        elif model == 'MLRegression':
            self.strategy = MLRegressionStrategy()
        elif model == 'BayesianRegression':
            self.strategy = BayesRegressionStrategy()
        else:
            sys.exit('The only available models are: "Hyper3DNet", "Hyper3DNetQD", "BayesianRegression"'
                     '"Russello", "CNNLF", "AdaBoost", "SAE", "RF", "GAM", and "MLRegression".')

        # Define the model using the selected CrossValStrategy
        self.strategy.defineModel(self.device, self.nbands, self.window_size, self.output_size, method=model)

    def trainPrevious(self, trainx, train_y, batch_size, epochs, filepath, printProcess, beta_, yscale):
        """Train a model using information collected from previous years.
        @param trainx: Batches of samples that will be used as inputs for yield prediction.
        @param train_y: Labels corresponding to trainx.
        @param batch_size: Size of the mini-batch for training (Used for the CNN).
        @param epochs: Indicates how many epochs the training process is repeated.
        @param filepath: Path used to stored the trained model.
        @param printProcess: If True, print some details of the process.
        @param beta_: Parameter used for the Hyper3DNetQD loss function.
        @param yscale: Statistics that were used to normalize the output. Used to revert normalization.
        """
        return self.strategy.trainModel(trainx, train_y, batch_size, self.device, epochs, filepath, printProcess,
                                        beta_, yscale)

    def predictSamples(self, datasample, maxs, mins, batch_size):
        """Return the predicted vectors as numpy vectors.
        @param datasample: Batches of samples that will be used as inputs for yield prediction.
        @param maxs: Mean of each feature calculated in the training set.
        @param mins: Standard deviation of each feature calculated in the training set.
        @param batch_size: Size of the mini-batch (Used for the CNN).
        """
        return self.strategy.predictSamples(datasample, maxs, mins, batch_size, self.device)

    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, MC_samples):
        """Return the predicted vectors and their distributions as numpy vectors.
        @param datasample: Batches of samples that will be used as inputs for yield prediction.
        @param maxs: Max of each feature calculated in the training set.
        @param mins: Standard deviation of each feature calculated in the training set.
        @param batch_size: Size of the mini-batch (Used for the CNN).
        @param MC_samples: Number of random samples used for MCDropout.
        """
        return self.strategy.predictSamplesUncertainty(datasample, maxs, mins, batch_size, self.device,
                                                       MC_samples)

    def loadModel(self, path):
        """Load a saved model
        @param path: File path with the saved model.
        """
        self.strategy.loadModelStrategy(path)
