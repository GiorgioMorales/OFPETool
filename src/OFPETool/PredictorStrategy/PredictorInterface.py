import abc


class PredictorInterface(abc.ABC):
    """Interface containing the methods used to declare a model and predict samples"""
    @abc.abstractmethod
    def defineModel(self, device, nbands, windowSize, outputSize, method):
        pass

    @abc.abstractmethod
    def loadModelStrategy(self, path):
        pass

    @abc.abstractmethod
    def trainModel(self, trainx, train_y, batch_size, device, epochs, filepath, printProcess, beta_, yscale):
        pass

    @abc.abstractmethod
    def predictSamples(self, datasample, maxs, mins, batch_size, device):
        pass

    @abc.abstractmethod
    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, device, MC_samples):
        pass
