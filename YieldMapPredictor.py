import os
import sys
import torch
import utils
import pickle
import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import loadData
from PredictorStrategy.PredictorModel import PredictorModel

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Static Functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def clear_border(map_in, r_value):
    """Set the value of the pixels on the borders"""
    map_in[0:3, :] = r_value
    map_in[-3:, :] = r_value
    map_in[:, 0:3] = r_value
    map_in[:, -3:] = r_value

    return map_in


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Main Class Definition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class YieldMapPredictor:
    """Class used to predict an entire yield map using on a previously trained model."""

    def __init__(self, filename: str, field: str, training_years: list, pred_year: int, data_mode="AggRADARPrec",
                 cov=None):
        """
        @param filename: Path containing the data (CSV)
        @param field: Name of the specific field that will be analyzed.
        @param training_years: Declare the years of data that will be used for training.
        @param pred_year: The year that will be used for prediction.
        @param data_mode: Determine the features that will be used. Options: 'All', 'AggRADAR', 'AllPrec'.
        @param cov:  Optional. Explicit list of covariates that will be used for training. Ex: ['N', 'NDVI', 'elev'].
        """
        # Set parameters
        self.filename = filename
        self.field = field
        self.training_years = training_years
        self.pred_year = pred_year
        self.data_mode = data_mode

        # Define variables
        self.windowSize = None
        self.outputSize = None
        self.modelType = None
        self.device = None
        self.model = None
        self.cellids = None
        self.coords = None
        self.data = None
        self.nbands = None
        self.mask_field = None
        self.prev_n = None
        self.path_model = None
        self.maxs = None
        self.mins = None
        self.maxY = None
        self.minY = None

        # Initialize data loader object
        self.dataLoader = DataLoader.DataLoader(filename, field=field, training_years=training_years,
                                                pred_year=pred_year, mode=data_mode, cov=cov)

    def init_model(self, modelType: str):
        """Initialize  ML model."""
        self.modelType = modelType
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set input window size. If the model is a CNN, the input is 2D
        if modelType in ['Hyper3DNet', 'Hyper3DNetQD', 'Russello', 'CNNLF']:
            self.windowSize = 5
        else:
            self.windowSize = 1
            self.outputSize = 1
        # Set output window size. If the model is Hyper3DNet, the output is 2D
        if 'Hyper3DNet' in modelType:
            self.outputSize = 5
        else:
            self.outputSize = 1

        return PredictorModel(self.modelType, self.device, self.nbands, self.windowSize, self.outputSize)

    def load_pred_data(self, objective):
        """Load and prepare the data that will be used for prediction"""
        # Load the entire yield map and the features of the selected year
        self.data, self.cellids, self.coords = self.dataLoader.load_raster(objective=objective)
        self.nbands = self.data.shape[2]  # Stores the number of features in the dataset.
        self.prev_n = self.data[:, :, 0].copy()  # Save for plotting later in case new prescription maps are used.

        # Obtain a binary mask of the field (mask==1: field, mask==0: out of the field)
        self.mask_field = (self.data[:, :, 2] * 0).astype(np.uint8)  # The elevation raster is used as a reference.
        self.mask_field[np.where(self.data[:, :, 2] != 0)] = 1  # There is field where the value is different than 0
        self.mask_field[np.where(self.data[:, :, 0] == -1)] = 0  # Remove from analysis points with missing N values
        self.mask_field = clear_border(self.mask_field, 0)

    def extract2DPatches(self):
        """Extract patches that will be analyzed by the CNN and their corresponding center positions"""

        # Initialize vectors to store patches and their center positions
        patches = []
        centers = []

        # Slide the window through the entire field
        for x in range(int((self.windowSize - 1) / 2), self.data.shape[0] - int((self.windowSize - 1) / 2), 1):
            for y in range(int((self.windowSize - 1) / 2), self.data.shape[1] - int((self.windowSize - 1) / 2),
                           1):
                # Define min and max coordinates of the patch around point (x, y)
                xmin = x - int((self.windowSize - 1) / 2)
                xmax = x + int((self.windowSize - 1) / 2) + 1
                ymin = y - int((self.windowSize - 1) / 2)
                ymax = y + int((self.windowSize - 1) / 2) + 1

                # Extract patch
                patch = self.data[xmin:xmax, ymin:ymax, :]

                # Avoid wasting time including patches that are full of zeros
                if np.sum(self.mask_field[xmin:xmax, ymin:ymax]) == 0:
                    continue

                # Add patch to batch and save position
                # print("Saving block at position X: " + str(x) + ", Y: " + str(y))
                patches.append(patch)
                centers.append((x, y))

                right = False
                bottom = False
                # Add rightmost patch
                if (ymax < self.data.shape[1]) and ((ymax + 1) > self.data.shape[1]):
                    patch = self.data[xmin:xmax, self.data.shape[1] - self.windowSize:, :]
                    print("Saving block at position X: " + str(x) + ", Y: " +
                          str(int(self.data.shape[1] - 0.5 - self.windowSize / 2)))
                    patches.append(patch)
                    centers.append((x, int(self.data.shape[1] - 0.5 - self.windowSize / 2)))
                    right = True

                # Add bottom patch
                if (xmax < self.data.shape[0]) and ((xmax + 1) > self.data.shape[0]):
                    patch = self.data[self.data.shape[0] - self.windowSize:, ymin:ymax, :]
                    print("Saving block at position X: " + str(int(self.data.shape[0] - 0.5 - self.windowSize / 2)) +
                          ", Y: " + str(y))
                    patches.append(patch)
                    centers.append((int(self.data.shape[0] - 0.5 - self.windowSize / 2), y))
                    bottom = True

                # Add rightmost bottom patch
                if right and bottom:
                    patch = self.data[self.data.shape[0] - self.windowSize:, self.data.shape[1] - self.windowSize:, :]
                    print("Saving block at position X: " + str(int(self.data.shape[0] - 0.5 - self.windowSize / 2)) +
                          ", Y: " + str(int(self.data.shape[1] - 0.5 - self.windowSize / 2)))
                    patches.append(patch)
                    centers.append((int(self.data.shape[0] - 0.5 - self.windowSize / 2),
                                    int(self.data.shape[1] - 0.5 - self.windowSize / 2)))

        patches = np.array(patches)
        # Reshape data as 4-D TENSOR
        patches = np.reshape(patches, (patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], 1))
        # Transpose dimensions to fit Pytorch order
        patches = patches.transpose((0, 4, 3, 1, 2))

        return patches, centers

    def modifyPrescription(self, presc_path: str, method='uniform', n_rate=None):
        """Used a modified prescription map before predicting yield.
        @param presc_path: Path of the prescription map.
        @param method: Options: 'FEAMOO', 'CCEAMOO', 'NSGA2', 'uniform'.
        @param n_rate: If method='uniform', specify the uniform nitrogen rate that will be applied to the entire field.
        """
        if method == 'uniform':
            # Set a prescription map with uniform values of nitrogen
            n_map = self.data[:, :, 0] * 0
            n_map[self.mask_field == 1] = n_rate
        else:
            # Load data
            n_map, _, _ = DataLoader.loadData(path=presc_path, cov=['N'], field=self.field, read_column=True)
            n_map = n_map[:, :, 0]

            # Points outside the boundary should be assigned the base-rate nitrogen
            n_map[(n_map == 0) & (self.data[:, :, 2] != 0)] = 100

        # Replace nitrogen map
        self.data[:, :, 0] = n_map

    def trainPreviousYears(self, batch_size=96, epochs=500, modelType='Hyper3DNet', print_process=True, objective='yld',
                           beta_=None):
        """Train using all the data from the selected years.
        @param batch_size: Size of the mini-batches used for training (used for the CNNs).
        @param epochs: Number pf epochs used for training (used for the CNNs).
        @param modelType: Name of the model that will be used. Options: 'Hyper3DNet', 'Hyper3DNetQD', 'AdaBoost',
                          'Russello', 'CNNLF', 'SAE', 'RF', 'GAM', and 'MLRegression'.
        @param print_process: If True, shows the evolution of the performance while training.
        @param objective: Name of the target. Default: 'yld' (Yield).
        @param beta_: Hyperparameters used for generation of the prediction intervals."""

        # Load training data of the previous years
        trainx, train_y = self.dataLoader.create_training_set(objective=objective)
        self.nbands = trainx.shape[2]

        print("\n******************************")
        print("Start Training: ")
        print("******************************")
        # Normalize features using the training set
        trainx, self.maxs, self.mins = utils.normalize(trainx)
        # Normalize outputs using the training set
        train_y, self.maxY, self.minY = utils.minMaxScale(train_y)
        # Save statistics
        np.save('models/temp/field_statistics.npy', [self.maxs, self.mins, self.maxY, self.minY])

        self.model = self.init_model(modelType=modelType)  # Initialize ML model
        self.path_model = 'models/temp/' + self.modelType + "-" + self.field + "--Objective-" + objective

        # Train the model using the current training-validation split
        return self.model.trainPrevious(trainx, train_y, batch_size, epochs, self.path_model, print_process,
                                        beta_=beta_, yscale=[self.maxY, self.minY])

    def predictYield(self, uncertainty=False, stats_path=None, model_path=None, modelType='Hyper3DNet',
                     objective='yld'):
        """Predict the yield map using a sliding window.
        @param uncertainty: If True, calculate and return prediction intervals.
        @param stats_path: Path that contains the statistics of the training set.
        @param model_path: Path that contains the trained model (Optional. Otherwise, the model needs to be trained).
        @param modelType: Name of the model that will be used. Options: 'Hyper3DNet', 'Hyper3DNetQD', 'AdaBoost',
                          'Russello', 'CNNLF', 'SAE', 'RF', 'GAM', and 'MLRegression'.
        @param objective: Name of the target. Default: 'yld' (Yield)."""

        self.load_pred_data(objective=objective)  # Load prediction data

        # Load model if there is one
        self.model = self.init_model(modelType=modelType)  # Initialize ML model
        print("Loading model...")
        if model_path is None:
            # If the model and the statistics are not provided, check if they are in the temp file
            self.path_model = 'models/temp/' + modelType + '-' + self.field + "--Objective-" + objective
            if os.path.exists(self.path_model):
                self.model.loadModel(path=self.path_model)
            else:
                sys.exit("There is no trained model saved. You need to execute the trainPreviousYear method first.")
        else:
            self.model.loadModel(path=model_path)

        # In case the statistics are not provided, read the training set to calculate the statistics
        if stats_path is None:
            # Try to check if there's a file in the temp folder with the statistics of the field
            stats_path = 'models\\temp\\' + self.field + '_statistics.npy'
            if os.path.exists(stats_path):
                [self.maxs, self.mins, self.maxY, self.minY] = np.load(stats_path, allow_pickle=True)
            else:  # Or calculate them
                # Load training data of the previous years
                trainx, train_y = self.dataLoader.create_training_set()
                # Calculate statistics using the training set
                _, self.maxs, self.mins = utils.normalize(trainx)
                _, self.maxY, self.minY = utils.minMaxScale(train_y)
                del trainx
                del train_y  # Remove from memory
        else:
            if os.path.exists(stats_path):
                # Read statistics from file
                [self.maxs, self.mins] = np.load(stats_path)
            else:
                sys.exit("There provided path does not exist.")

        # Remove areas affected by fire in sec35middle
        if self.field == 'sec35middle':
            self.mask_field[83:106, 46:] = 0
            self.mask_field[102:116, 43:60] = 0
        self.coords[self.mask_field == 0] = None

        # Extract patches and their centers
        patches, centers = self.extract2DPatches()

        # Predict yield values for each patch in the batch
        uncPatches = None
        if uncertainty:
            yieldPatches, uncPatches = np.array(
                self.model.predictSamplesUncertainty(datasample=patches, maxs=self.maxs,
                                                     mins=self.mins, batch_size=256,
                                                     MC_samples=50))
        else:
            yieldPatches = np.array(self.model.predictSamples(datasample=patches, maxs=self.maxs,
                                                              mins=self.mins, batch_size=256))

        # ####################################################################
        # Reconstruct map one block at a time. Check if there is overlapping.
        # ####################################################################
        yield_map = np.zeros((self.data.shape[0], self.data.shape[1]))  # Initialize empty yield map
        PI_map = np.zeros((self.data.shape[0], self.data.shape[1]))  # Initialize empty confidence map
        temp_yield_map = np.frompyfunc(list, 0, 1)(np.empty((self.data.shape[0], self.data.shape[1]), dtype=object))
        # Initialize variables for when the upper and lower bounds are given instead of the predicted yield
        y_u, y_l, temp_y_u, temp_y_l, temp_y_p = None, None, None, None, None
        if self.modelType == 'Hyper3DNetQD':
            y_u = np.zeros((self.data.shape[0], self.data.shape[1]))  # Store upper bounds
            y_l = np.zeros((self.data.shape[0], self.data.shape[1]))  # Store lower bounds
            temp_y_u = np.frompyfunc(list, 0, 1)(np.empty((self.data.shape[0], self.data.shape[1]), dtype=object))
            temp_y_l = np.frompyfunc(list, 0, 1)(np.empty((self.data.shape[0], self.data.shape[1]), dtype=object))
            temp_y_p = np.frompyfunc(list, 0, 1)(np.empty((self.data.shape[0], self.data.shape[1]), dtype=object))

        for i, ypatch in enumerate(yieldPatches):
            # Get original coordinates of the center of ypatch
            (x, y) = centers[i]
            # Define min and max coordinates of the patch around point (x, y)
            xmin = x - int((self.outputSize - 1) / 2)
            ymin = y - int((self.outputSize - 1) / 2)
            # Put ypatch in temp_map considering its original position
            if self.outputSize == 1:
                if self.modelType in ['MLRegression', 'AdaBoost', 'SAE', 'RF', 'GAM']:
                    temp_yield_map[xmin, ymin].append(ypatch)
                    if uncertainty:
                        PI_map[xmin, ymin] = uncPatches[i]
                else:
                    if self.modelType == 'Hyper3DNetQD':
                        temp_y_u[xmin, ymin] += list(ypatch[0, :, :])
                        temp_y_l[xmin, ymin] += list(ypatch[1, :, :])
                        temp_y_p[xmin, ymin] += list(ypatch[2, :, :])
                    else:
                        if str(type(ypatch[0])) == '<class \'numpy.float64\'>':
                            temp_yield_map[xmin, ymin].append(ypatch)
                        else:
                            temp_yield_map[xmin, ymin] += list(ypatch[0, :])
            else:
                for patch_x in range(ypatch.shape[0]):
                    for patch_y in range(ypatch.shape[1]):
                        if self.modelType == 'Hyper3DNetQD':
                            if str(type(ypatch[patch_x, patch_y])) == '<class \'numpy.float64\'>':
                                temp_y_u[xmin + patch_x, ymin + patch_y].append(ypatch[0])
                                temp_y_l[xmin + patch_x, ymin + patch_y].append(ypatch[1])
                                temp_y_p[xmin + patch_x, ymin + patch_y].append(ypatch[2])
                            else:
                                temp_y_u[xmin + patch_x, ymin + patch_y] += list(ypatch[0, patch_x, patch_y])
                                temp_y_l[xmin + patch_x, ymin + patch_y] += list(ypatch[1, patch_x, patch_y])
                                temp_y_p[xmin + patch_x, ymin + patch_y] += list(ypatch[2, patch_x, patch_y])
                        else:
                            if str(type(ypatch[patch_x, patch_y])) == '<class \'numpy.float64\'>':
                                temp_yield_map[xmin + patch_x, ymin + patch_y].append(ypatch[patch_x, patch_y])
                            else:
                                temp_yield_map[xmin + patch_x, ymin + patch_y] += list(ypatch[patch_x, patch_y])

        # Average overlapping regions
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.modelType == 'Hyper3DNetQD':
                    if not temp_y_u[i, j]:
                        continue
                    # Calculate mean upper and lower bounds
                    up = np.mean(temp_y_u[i, j])
                    lo = np.mean(temp_y_l[i, j])
                    # Calculate mean response
                    yield_map[i, j] = np.mean(temp_y_p[i, j])
                    # Calculate uncertainty width
                    PI_map[i, j] = up - lo
                    y_u[i, j] = up
                    y_l[i, j] = lo
                else:
                    # Discard zeroes
                    temp_vec = [v for v in temp_yield_map[i, j] if v != 0]
                    if len(temp_vec) == 0:
                        continue
                    # Calculate mean
                    yield_map[i, j] = np.mean(temp_vec)
                    # Obtain the standard deviation of each pixel for CNNs methods using MCDropout
                    if uncertainty and (self.modelType in ['Hyper3DNet', 'Russello', 'CNNLF']):
                        PI_map[i, j] = np.std(temp_vec)  # This is the model uncertainty (variance)

        # Discard results that are outside the field
        yield_map = np.multiply(yield_map, self.mask_field)
        yield_map = utils.reverseMinMaxScale(yield_map, maxs=self.maxY, mins=self.minY)
        yield_map[np.where(yield_map < 0)] = 0
        PI_map = np.multiply(PI_map, self.mask_field)

        # If using MCDropout, add the data noise variance
        if uncertainty and self.modelType not in ['MLRegression', 'Hyper3DNetQD']:
            # Load validation MSE
            with open(self.path_model + '_validationMSE', 'rb') as f:
                val_MSE = pickle.load(f)
            PI_map = np.sqrt(PI_map ** 2 + val_MSE)

        # Save maps as shapefiles
        coords = np.reshape(self.coords, (self.coords.shape[0] * self.coords.shape[1]))  # Vectorize coordinates map
        yield_vector = np.reshape(yield_map, (yield_map.shape[0] * yield_map.shape[1]))  # Vectorize yield map
        yield_vector = [y for y, cr in zip(yield_vector, coords) if cr is not None]
        if self.modelType == "Hyper3DNetQD":
            yu_vector = np.reshape(y_u, (y_u.shape[0] * y_u.shape[1]))  # Vectorize upper bound map
            yu_vector = [y for y, cr in zip(yu_vector, coords) if cr is not None]
            yl_vector = np.reshape(y_l, (y_l.shape[0] * y_l.shape[1]))  # Vectorize lower bound map
            yl_vector = [y for y, cr in zip(yl_vector, coords) if cr is not None]
            results = [yield_vector, yu_vector, yl_vector]
        else:
            results = [yield_vector]
        coords = [cr for cr in coords if cr is not None]
        utils.createShapefile(coords=coords, columns=results, filepath=self.path_model + '_Shapefile', obj=objective)

        # Return yield map
        if not uncertainty:
            return yield_map
        # Or return prediction and upper and lower bounds
        else:
            if self.modelType == "Hyper3DNetQD":
                y_u = utils.reverseMinMaxScale(y_u, maxs=self.maxY, mins=self.minY)
                y_l = utils.reverseMinMaxScale(y_l, maxs=self.maxY, mins=self.minY)
                PI_map = utils.reverseMinMaxScale(PI_map, maxs=self.maxY, mins=self.minY)
                return yield_map, y_u * self.mask_field, y_l * self.mask_field, PI_map
            else:
                # Include the z-multiplier
                PI_map *= 1.96
                return yield_map, yield_map + PI_map, yield_map - PI_map, PI_map * 2


if __name__ == '__main__':

    #####################################################
    # Corn Field Simulation
    #####################################################
    filepath = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\sim_data.csv'
    fieldname = ''
    cvars = ['N', 'par1', 'par2', 'par3', 'par4', 'par5', 'par6', 'par7', 'par8', 'par9', 'par10', 'par11', 'par12',
             'par13', 'par14']
    modelname = "Hyper3DNet"
    goal = 'yld'
    # method = "GAM"
    # 10-fold cross validation
    RMSE = []
    prediction, target = None, None
    for nt in range(10):
        print("************************************************************************************************")
        print("Fold " + str(nt + 1) + " / 10")
        print("************************************************************************************************")
        tyears = list(np.arange(1, 11))
        tyears.remove(10 - nt)
        pyear = 10 - nt
        predictor = YieldMapPredictor(filename=filepath, field=fieldname, training_years=tyears, pred_year=pyear,
                                      cov=cvars)
        # Train and validate
        # predictor.trainPreviousYears(epochs=500, batch_size=64, modelType=method, objective=goal)
        prediction = np.clip(predictor.predictYield(modelType=modelname, objective=goal), a_min=0, a_max=2E4)
        # Compare to the ground-truth and calculate the RMSE
        target, _, _, _ = loadData(path=filepath, field=fieldname, year=10 - nt, cov=cvars, inpaint=True,
                                   inpaint_features=False, base_N=120, test=False, obj=goal)
        RMSE.append(utils.mse(prediction, target) ** .5)
        print("Validation RMSE = " + str(RMSE[nt]))
    # Plot lat results for reference
    plt.figure()
    plt.imshow(prediction)
    plt.title("Predicted yield map")
    plt.axis("off")
    plt.figure()
    plt.imshow(target)
    plt.title("Ground-truth yield map")
    plt.axis("off")
