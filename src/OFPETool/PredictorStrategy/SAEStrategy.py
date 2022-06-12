from src.OFPETool.Predictor import utils
import torch
import pickle
import random
from src.OFPETool.PredictorStrategy.networks import *
from src.OFPETool.PredictorStrategy.modelObject import *
from src.OFPETool.PredictorStrategy.PredictorInterface import PredictorInterface

np.random.seed(seed=7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class SAEStrategy(PredictorInterface):

    def __init__(self):
        self.model = None
        self.method = None
        self.output_size = 1
        self.device = None

    def defineModel(self, device, nbands, windowSize, outputSize, method):
        """Override model declaration method"""
        self.method = method
        self.output_size = outputSize
        self.device = device

        network = StackedAutoEncoder(nbands=nbands)
        network.to(device)
        final_layer = nn.Linear(125, 1).cuda()
        final_layer.to(device)
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adadelta(final_layer.parameters(), lr=1.0)

        self.model = SAEObject(network, final_layer, criterion, optimizer)

    def trainModel(self, trainx, train_y, batch_size, device, epochs, filepath, printProcess, beta_, yscale):
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Separate 90% of the data for training
        valX = trainx[int(len(trainx) * 90 / 100):, :, :, :, :]
        valY = train_y[int(len(train_y) * 90 / 100):, :, :]
        trainx = trainx[0:int(len(trainx) * 90 / 100), :, :, :, :]
        train_y = train_y[0:int(len(train_y) * 90 / 100), :, :]

        # Vectorize data (4-D to 1-D)
        trainx = trainx.transpose((0, 3, 4, 1, 2))
        trainx = np.reshape(trainx, (trainx.shape[0] * trainx.shape[1] * trainx.shape[2] * trainx.shape[3],
                                     trainx.shape[4]))
        train_y = np.reshape(train_y, (train_y.shape[0] * train_y.shape[1] * train_y.shape[2]))
        valX = valX.transpose((0, 3, 4, 1, 2))
        valX = np.reshape(valX, (valX.shape[0] * valX.shape[1] * valX.shape[2] * valX.shape[3], valX.shape[4]))
        valY = np.reshape(valY, (valY.shape[0] * valY.shape[1] * valY.shape[2], 1))
        # Remove repetitions
        trainx, kept_indices = np.unique(trainx, axis=0, return_index=True)
        train_y = train_y[kept_indices]
        valX, kept_indices = np.unique(valX, axis=0, return_index=True)
        valY = valY[kept_indices]

        indexes = np.arange(len(trainx))  # Prepare list of indexes for shuffling
        T = np.ceil(1.0 * len(trainx) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch

        val_mse = np.infty
        for epoch in range(epochs):  # Epoch loop

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                trainxb = torch.from_numpy(trainx[inds]).float().to(device)
                trainyb = torch.from_numpy(np.reshape(train_y[inds], (len(inds), 1))).float().to(device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                features = self.model.network(trainxb).detach()
                outputs = self.model.final_layer(features)
                loss = self.model.criterion(outputs, trainyb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if printProcess and epoch % 10 == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, loss.item()))

            # Validation step
            with torch.no_grad():
                self.model.network.eval()
                ypred, _ = self.model.network(torch.from_numpy(valX).float().to(device))
                ypred = self.model.final_layer(ypred.detach()).cpu().numpy()

            mse = utils.mse(valY, ypred)

            # Save model if MSE decreases
            if mse < val_mse:
                val_mse = mse
                torch.save(self.model.network.state_dict(), filepath)
                torch.save(self.model.final_layer.state_dict(), filepath + 'final_layer')

            if printProcess and epoch % 10 == 0:
                print('VALIDATION: Best_MSE: %.5f' % val_mse)

        # Save model
        with open(filepath + '_validationMSE', 'wb') as fil:
            pickle.dump(val_mse, fil)

    def predictSamples(self, datasample, maxs, mins, batch_size, device):
        """Predict yield values (in patches or single values) given a batch of samples."""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)[:, 0, :, 0, 0]

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(datasample) / batch_size).astype(np.int32)
            indtest = np.arange(len(datasample))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch, _ = self.model.network(torch.from_numpy(valxn[inds]).float().to(device))
                ypred_batch = self.model.final_layer(ypred_batch.detach())
                ypred = ypred + (ypred_batch.cpu().numpy()).tolist()

        return ypred

    def predictSamplesUncertainty(self, datasample, means, stds, batch_size, device, MC_samples):
        """Predict yield probability distributions given a batch of samples using MCDropout"""
        valxn = utils.applyMinMaxScale(datasample, means, stds)[:, 0, :, 0, 0]

        with torch.no_grad():
            preds_MC = np.zeros((len(datasample), self.output_size, MC_samples))
            for it in range(0, MC_samples):  # Test the model 'MC_samples' times
                ypred = []
                self.model.network.eval()
                self.model.final_layer.eval()
                enable_dropout(self.model.network)  # Set Dropout layers to test mode
                Teva = np.ceil(1.0 * len(datasample) / batch_size).astype(np.int32)  # Number of batches
                indtest = np.arange(len(datasample))
                for b in range(Teva):
                    inds = indtest[b * batch_size:(b + 1) * batch_size]
                    ypred_batch, _ = self.model.network(torch.from_numpy(valxn[inds]).float().to(device))
                    ypred_batch = self.model.final_layer(ypred_batch.detach())
                    ypred = ypred + (ypred_batch.cpu().numpy()).tolist()

                preds_MC[:, :, it] = np.array(ypred)

        return preds_MC, None

    def loadModelStrategy(self, path):
        self.model.network.load_state_dict(torch.load(path))
        self.model.final_layer.load_state_dict(torch.load(path + 'final_layer'))
