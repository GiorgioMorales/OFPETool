import sys
import utils
import torch
import pickle
import random
from models.networks import *
from models.modelObject import *
from PredictorStrategy.PredictorInterface import PredictorInterface

np.random.seed(seed=7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


"""**********************************************************************
Static functions******************************************************"""


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def AQD_objective(y_pred, y_true, beta_):
    """AQD loss function,
    @param y_pred: NN output (y_u, y_l, y)
    @param y_true: Ground-truth.
    @param beta_: Specify the importance of the width factor."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0] * 15
    y_l = y_pred[:, 1] * 15
    y_o = y_pred[:, 2] * 15
    y_true = y_true * 15

    MSE = torch.mean((y_o - y_true) ** 2)  # Calculate MSE
    MPIW_p = torch.mean(torch.abs(y_u - y_o) + torch.abs(y_o - y_l))  # Calculate MPIW_penalty
    Constraints = (torch.exp(torch.mean(-y_u + y_o) + torch.max(torch.abs(y_o - y_true))) +
                   torch.exp(torch.mean(-y_o + y_l) + torch.max(torch.abs(y_o - y_true))) +
                   torch.exp(torch.mean(-y_u + y_l)))
    # Calculate loss
    Loss_S = (1 - beta_[0]) * MPIW_p + MSE * beta_[0] + Constraints * beta_[1]

    return Loss_S


"""**********************************************************************
Class definition******************************************************"""


class SpatialCNNStrategy(PredictorInterface):

    def __init__(self):
        self.model = None
        self.method = None
        self.output_size = None
        self.device = None

    def defineModel(self, device, nbands, windowSize, outputSize, method):
        """Override model declaration method"""
        self.method = method
        self.output_size = outputSize
        self.device = device

        criterion = nn.MSELoss()
        if self.method == 'Hyper3DNet':
            network = Hyper3DNetLiteReg(img_shape=(1, nbands, windowSize, windowSize), output_size=self.output_size)
        elif self.method == "Hyper3DNetQD":
            network = Hyper3DNetLiteReg(img_shape=(1, nbands, windowSize, windowSize), output_size=self.output_size,
                                        output_channels=3)
        elif self.method == 'Russello':
            network = Russello(img_shape=(1, nbands, windowSize, windowSize))
        elif self.method == 'CNNLF':
            network = CNNLF(img_shape=(1, nbands, windowSize, windowSize))
        else:
            sys.exit('The only available network architectures are: Hyper3DNet, Russello, and CNNLF (so far).')
        network.to(device)
        # Training parameters
        optimizer = optim.Adadelta(network.parameters(), lr=1.0)

        self.model = CNNObject(network, criterion, optimizer)

    def trainModel(self, trainx, train_y, batch_size, device, epochs, filepath, printProcess, beta_, yscale):
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # trainx, train_y = utils.add_rotation_flip(trainx, train_y)

        indexes = np.arange(len(trainx))  # Prepare list of indexes for shuffling
        np.random.shuffle(indexes)
        trainx = trainx[indexes]
        train_y = train_y[indexes]

        # Separate 90% of the data for training
        valX = trainx[int(len(trainx) * 90 / 100):, :, :, :, :]
        trainx = trainx[0:int(len(trainx) * 90 / 100), :, :, :, :]
        # If outputSize < windowSize, discard predictions from the outer regions
        cmin = int((train_y.shape[1] - 1) / 2 - (self.output_size - 1) / 2)
        cmax = int((train_y.shape[1] - 1) / 2 + (self.output_size - 1) / 2) + 1
        if self.output_size > 1:  # The output should be a 2-D patch
            valY = train_y[int(len(train_y) * 90 / 100):, cmin:cmax, cmin:cmax]
            train_y = train_y[0:int(len(train_y) * 90 / 100), cmin:cmax, cmin:cmax]
        else:  # Otherwise, the output is just the central value of the patch
            valY = train_y[int(len(train_y) * 90 / 100):, cmin, cmin]
            valY = valY.reshape((len(valY), 1))
            train_y = train_y[0:int(len(train_y) * 90 / 100), cmin, cmin]
            train_y = train_y.reshape((len(train_y), 1))

        indexes = np.arange(len(trainx))  # Prepare list of indexes for shuffling
        np.random.shuffle(indexes)
        T = np.ceil(1.0 * len(trainx) / batch_size).astype(np.int32)  # Compute the number of steps in an epoch

        val_mse = np.infty
        val_picp = 0
        val_mpiw = np.infty
        MPIWtr = []
        PICPtr = []
        MSEtr = []
        MPIW = []
        PICP = []
        MSE = []
        widths = [0]
        picp = 0
        first95 = True  # This is a flag used to check if PICP has already reached 95% PICP during the training
        for epoch in range(epochs):  # Epoch loop
            if epoch > 0 and (self.method in ['Hyper3DNetQD']):
                indexes = np.argsort(widths)  # Batch-sorting

            self.model.network.train()  # Sets training mode
            running_loss = 0.0
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                inds = indexes[step * batch_size:(step + 1) * batch_size]

                # Get actual batches
                trainxb = torch.from_numpy(trainx[inds]).float().to(device)
                trainyb = torch.from_numpy(train_y[inds]).float().to(device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(trainxb, self.device)
                if self.method == 'Hyper3DNetQD':
                    loss = AQD_objective(outputs, trainyb, beta_)
                else:
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
                # if self.method in ['Hyper3DNetQD']:  # These methods use mult forward passes w active dropout
                #     enable_dropout(self.model.network)
                #     ypredtr, ypred = 0, 0
                #     for r in range(5):
                #         ypredtr += self.model.network(torch.from_numpy(trainx).float().to(self.device),
                #                                       self.device).cpu().numpy()
                #         ypred += self.model.network(torch.from_numpy(valX).float().to(self.device),
                #                                     self.device).cpu().numpy()
                #     ypredtr /= 5
                #     ypred /= 5
                # else:
                ypredtr = self.model.network(torch.from_numpy(trainx).float().to(self.device),
                                             self.device).cpu().numpy()
                ypred = self.model.network(torch.from_numpy(valX).float().to(self.device),
                                           self.device).cpu().numpy()
                # Revert nornmalization
                Ytrain_original = utils.reverseMinMaxScale(train_y, yscale[0], yscale[1])
                Yval_original = utils.reverseMinMaxScale(valY, yscale[0], yscale[1])
                ypredtr = utils.reverseMinMaxScale(ypredtr, yscale[0], yscale[1])
                ypred = utils.reverseMinMaxScale(ypred, yscale[0], yscale[1])

                if self.method == 'Hyper3DNetQD':  # Average upper and lower limit to obtain expected output
                    msetr = utils.mse(Ytrain_original, ypredtr[:, 2, :, :])
                    mse = utils.mse(Yval_original, ypred[:, 2, :, :])
                else:
                    msetr = utils.mse(Ytrain_original, ypredtr)
                    mse = utils.mse(Yval_original, ypred)
                MSEtr.append(msetr)
                MSE.append(mse)
                if self.method == 'Hyper3DNetQD':
                    # Calculate MPIW and PICP
                    y_true = torch.from_numpy(Ytrain_original).float().to(self.device)
                    y_utr = torch.from_numpy(ypredtr[:, 0]).float().to(self.device)
                    y_ltr = torch.from_numpy(ypredtr[:, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_utr - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_ltr))
                    Ktr = torch.mul(K_U, K_L)
                    y_true = torch.from_numpy(Yval_original).float().to(self.device)
                    y_u = torch.from_numpy(ypred[:, 0]).float().to(self.device)
                    y_l = torch.from_numpy(ypred[:, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_u - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_l))
                    K = torch.mul(K_U, K_L)
                    # Update curves
                    MPIWtr.append((torch.sum(torch.mul((y_utr - y_ltr), Ktr)) / (torch.sum(Ktr) + 0.0001)).item())
                    PICPtr.append(torch.mean(Ktr).item())
                    width = (torch.sum(torch.mul((y_u - y_l), K)) / (torch.sum(K) + 0.0001)).item()
                    picp = torch.mean(K).item()
                    MPIW.append(width)
                    PICP.append(torch.mean(K).item())
                    # Get a vector of all the PI widths in the training set
                    widths = np.mean((y_utr - y_ltr).cpu().numpy(), axis=(1, 2))

            # Save model if PICP increases
            if self.method == 'Hyper3DNetQD':
                # Criteria 1: If <95, choose max picp, if picp>95, choose any picp if width<minimum width
                if mse < val_mse:
                    val_mse = mse
                    val_picp = picp
                    val_mpiw = width
                    if filepath is not None:
                        torch.save(self.model.network.state_dict(), filepath)
            else:  # Save model if MSE decreases
                if mse < val_mse:
                    val_mse = mse
                    torch.save(self.model.network.state_dict(), filepath)

            # Print every 10 epochs
            if printProcess and epoch % 10 == 0:
                if self.method != 'Hyper3DNetQD':
                    print('VALIDATION: Training_RMSE: %.5f. Best_RMSEval: %.5f' % (msetr ** .5, val_mse ** .5))
                else:
                    print('VALIDATION: Training_RMSE: %.5f. Best_RMSEval: %.5f. RMSE val: %.5f. PICP val: %.5f. '
                          'MPIW val: %.5f' % (msetr ** .5, val_mse ** .5, mse ** .5, picp, width))
                    print(val_picp)
                    print(val_mpiw)

        # Save model
        with open(filepath + '_validationMSE', 'wb') as fil:
            pickle.dump(val_mse, fil)
        # Save history
        np.save(filepath + '_historyMSEtr', MSEtr)
        np.save(filepath + '_historyMSE', MSE)
        if self.method == 'Hyper3DNetQD' or \
                self.method == 'Hyper3DNetQDPearce':  # Average upper and lower limit to obtain expected output
            np.save(filepath + '_historyMPIWtr', MPIWtr)
            np.save(filepath + '_historyPICPtr', PICPtr)
            np.save(filepath + '_historyMPIW', MPIW)
            np.save(filepath + '_historyPICP', PICP)

        return MPIW, PICP, MSE

    def predictSamples(self, datasample, maxs, mins, batch_size, device):
        """Predict yield values (in patches or single values) given a batch of samples."""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)

        with torch.no_grad():
            self.model.network.eval()
            enable_dropout(self.model.network)
            ypred = []
            Teva = np.ceil(1.0 * len(datasample) / batch_size).astype(np.int32)
            indtest = np.arange(len(datasample))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(device), device)
                if self.method == 'Hyper3DNetQD':
                    ypred_batch = ypred_batch[:, 2, :, :]
                ypred = ypred + (ypred_batch.cpu().numpy()).tolist()

        return ypred

    def predictSamplesUncertainty(self, datasample, maxs, mins, batch_size, device, MC_samples):
        """Predict yield probability distributions given a batch of samples using MCDropout"""
        valxn = utils.applyMinMaxScale(datasample, maxs, mins)

        with torch.no_grad():
            if self.output_size == 1:
                preds_MC = np.zeros((len(datasample), self.output_size, MC_samples))
            else:
                preds_MC = np.zeros((len(datasample), self.output_size, self.output_size, MC_samples))
                if self.method == "Hyper3DNetQD":
                    preds_MC = np.zeros((len(datasample), 3, self.output_size, self.output_size, MC_samples))
            for it in range(0, MC_samples):  # Test the model 'MC_samples' times
                ypred = []
                self.model.network.eval()
                enable_dropout(self.model.network)  # Set Dropout layers to test mode
                Teva = np.ceil(1.0 * len(datasample) / batch_size).astype(np.int32)  # Number of batches
                indtest = np.arange(len(datasample))
                for b in range(Teva):
                    inds = indtest[b * batch_size:(b + 1) * batch_size]
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(device), device)
                    ypred = ypred + ypred_batch.tolist()

                if self.output_size == 1:
                    preds_MC[:, :, it] = np.array(ypred)
                else:
                    if self.method == "Hyper3DNetQD":
                        preds_MC[:, :, :, :, it] = np.array(ypred)
                    else:
                        preds_MC[:, :, :, it] = np.array(ypred)

        # Return the useful predictions
        return preds_MC, None

    def loadModelStrategy(self, path):
        self.model.network.load_state_dict(torch.load(path))
