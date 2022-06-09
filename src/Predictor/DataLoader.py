import cv2
import sys
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from itertools import cycle, islice


#######################################################################################################################
# Static functions
#######################################################################################################################
def autopatches(data, window=5, overlap=.75, plot=False):
    """Function used to automatically extract patches from the data
    @param data: Numpy array with containing the data that will be cropped. The first channel is the target variable.
    @param window: Size of the square patches.
    @param overlap: Percentage of overlap allowed between patches.
    @param plot: If True, it plots the target raster and the patches selected."""

    ###################################################################
    # Step 1: Detect the number of connected components within the data
    ###################################################################
    targ = data[0, :, :].copy()
    sats = data[8, :, :].copy()
    # Create binary mask
    bmask = np.ones((targ.shape[0], targ.shape[1]), dtype=np.uint8)
    bmask[np.where(targ <= 0)] = 0  # If there is useful information=1, otherwise=0
    # Label image
    num_labels, labels = cv2.connectedComponents(bmask)
    ax, fig = None, None
    if plot:
        # Plot target
        fig, ax = plt.subplots(1)
        ax.imshow(targ)

    ###################################################################
    # Step 2: Extract patches for each component
    ###################################################################
    visited = np.zeros((targ.shape[0], targ.shape[1]), dtype=np.uint8)  # track the points that were visited
    centers = np.zeros((targ.shape[0], targ.shape[1]), dtype=np.uint8)  # track the selected centers
    list_patches = []   # Array that will contain all selected patches
    for nc in range(1, num_labels):
        # Step 2.1: Evaluate if this component is big enough to be considered
        cords = np.where(labels == nc)
        xcords = cords[0]
        ycords = cords[1]
        width = np.max(xcords) - np.min(xcords)
        height = np.max(ycords) - np.min(ycords)
        if width < window or height < window:
            continue

        # For each point in the component, create a patch of (window x window) pixels around it
        # and verify if it is inside the contour of the component and if it does not exceed the desired overlap.
        for point in zip(cords[0], cords[1]):
            # Define min and max coordinates of the patch around "point"
            xmin = point[0] - int((window - 1) / 2)
            xmax = point[0] + int((window - 1) / 2) + 1
            ymin = point[1] - int((window - 1) / 2)
            ymax = point[1] + int((window - 1) / 2) + 1
            # Step 2.2: Check if the patch is within the bounding box
            if xmin < np.min(xcords) or ymin < np.min(ycords) or xmax > np.max(xcords) or ymax > np.max(ycords):
                continue
            # Step 2.3: Check if the patch contains points without information (i.e. if contains background pixels)
            if len(np.where(targ[xmin:xmax, ymin:ymax] == -1)[0]) > 0 or \
                    len(np.where(targ[xmin:xmax, ymin:ymax] <= 5)[0]) > 10 or \
                        len(np.where(sats[xmin:xmax, ymin:ymax] == -1)[0]) > 0 \
                             or len(np.where(data[1, :, :][xmin:xmax, ymin:ymax] == -1)[0]) > 0:
                continue
            # Step 2.4: Check if the patch exceeds the maximum overlap allowed
            if np.sum(visited[xmin:xmax, ymin:ymax]) / (window * window) > overlap:
                continue
            # Step 2.5: Check if there is a distance of at least (window - 1) / 2 w.r.t to the closest center
            neighbors = np.where(centers[xmin:xmax, ymin:ymax] == 1)
            mindist = np.infty
            for center in zip(neighbors[0], neighbors[1]):
                newdist = np.sqrt(center[0] ** 2 + center[1] ** 2)
                if newdist < mindist:
                    mindist = newdist
            if mindist < (window - 1) / 2:
                continue

            # Updated the matrix of visited points and centers
            visited[xmin:xmax, ymin:ymax] = np.ones((window, window))
            centers[point[0], point[1]] = 1

            # Step 2.5: Add patch to list
            list_patches.append(data[:, xmin:xmax, ymin:ymax])

            # Show selected patches
            if plot:
                # Paint the current patch
                rect = patches.Rectangle((ymin - 0.5, xmin - 0.5), window, window, linewidth=1,
                                         edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
                fig.canvas.draw()

    return list_patches


def loadData(path=None, obj='yld', year=2018, field='', cov=None, mode='AggRADARPrec', inpaint=True,
             inpaint_features=False, base_N=120, read_column=False, test=False):
    """Load aggregated data in .CSV format and Sentinel-1 images
    @param path: Specify the absolute path of the CSV file that will be read. It can be also a FileStorage object.
    @param obj: Specify the type of target that will be evaluated. Options: 'yld', 'pro'.
    @param year: Year that will be evaluated. Options: '2016', '2017', '2018', '2019', or '2020'.
    @param field: Field that will be evaluated.
    @param cov: List of the names of the desired covariates. E.g. ['aa_n', 'slope', 'elev', 'aspect_rad', 'prec_cy_g'].
    @param mode: Define the features that will be read. Options: 'All', 'AggRADAR', 'AllPrec'.
    @param inpaint: If True, use in-painting to handle missing yield information.
    @param inpaint_features: If True, use in-painting to handle missing information of the features.
    @param base_N: Farmer selected Nitrogen rate.
    @param read_column: If True, read all the data from the specified column only (no target or year is considered).
    @param test: If True, ignore the target column and retrieve only the date.
    return 'target', a np array containing a raster with the values of interest, and 'data', a multidimensional array
        that contains the corresponding covariates for each point.
    """

    # Set default covariates that will be read depending on the selected mode
    if mode == 'All':
        covariates = ['aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'prec_cy_g', 'ndvi_py_f', 'ndwi_py_f',
                      'vv_cy_f', 'vh_cy_f']
    elif mode == 'AggRADAR':
        covariates = ['aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'vv_cy_f', 'vh_cy_f']
    elif mode == 'AggRADARPrec':
        covariates = ['aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f', 'vh_cy_f']
    else:
        sys.exit("The possible combinations are 'All', 'AggRADAR', and 'AggRADARPrec'.")

    if cov is None or cov == ['N']:  # Set default features in case it is not specified.
        covB = covariates
    else:
        covB = cov

    # Read the data file to know the original dimensions
    if isinstance(path, str):
        data = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data)
    else:
        df = path
    # In addition to the selected features, we also read the coordinates, the year, and the desired target
    if not test:
        df = df[covB + ['x', 'y', 'year', 'field', 'cell_id', obj]]
    else:
        df = df[covB + ['x', 'y', 'year', 'field', 'cell_id']]

    if field != '':
        df = df[df['field'] == field]
    else:
        field = 'nan'
    # Sort unique UTM XY coordinates
    rows = np.round(df['y'].unique(), 6)
    rows.sort()
    cols = np.round(df['x'].unique(), 6)
    cols.sort()

    # Transform XY coordinates to indexes
    data = np.ones((rows.shape[0], cols.shape[0], len(covB))) * -2
    target = np.ones((rows.shape[0], cols.shape[0],)) * -2
    cellids = np.empty((rows.shape[0], cols.shape[0]), dtype=object)
    XYcoords = np.empty((rows.shape[0], cols.shape[0]), dtype=object)
    for d in df.iterrows():
        if not read_column:
            if d[1]['year'] == year and str(d[1]['field']) == field:
                xcord = np.where(rows == np.round(d[1]['y'], 6))[0][0]
                ycord = np.where(cols == np.round(d[1]['x'], 6))[0][0]
                # If there is missing values, fill with -1
                if not test:
                    if np.isnan(d[1][obj]):  # Check the target
                        d[1][obj] = -1
                for f in covB:  # Check each one of the features
                    if np.isnan(d[1][f]):
                        d[1][f] = -1
                if not test:
                    target[xcord, ycord, ] = d[1][obj]
                cellids[xcord, ycord] = d[1]['cell_id']
                XYcoords[xcord, ycord] = [d[1]['x'], d[1]['y']]
                if not test:
                    d2 = d[1].drop(['x', 'y', 'year', 'field', 'cell_id', obj])
                else:
                    d2 = d[1].drop(['x', 'y', 'year', 'field', 'cell_id'])
                data[xcord, ycord, :] = d2.iloc[:]
        elif d[1]['field'] == field:
            xcord = np.where(rows == np.round(d[1]['y'], 6))[0][0]
            ycord = np.where(cols == np.round(d[1]['x'], 6))[0][0]
            # If there is missing values, fill with -1
            for f in covB:  # Check each one of the features
                if np.isnan(d[1][f]):
                    d[1][f] = -1
            cellids[xcord, ycord] = d[1]['cell_id']
            XYcoords[xcord, ycord] = [d[1]['x'], d[1]['y']]
            d2 = d[1].drop(['x', 'y', 'field', 'cell_id'])
            data[xcord, ycord, :] = d2.iloc[:]

    # Flip the rasters horizontally
    target = np.flip(target, 0)
    data = np.flip(data, 0)
    cellids = np.flip(cellids, 0)
    XYcoords = np.flip(XYcoords, 0)

    # Apply in-painting to handle missing information of the target
    if not test:
        temp, maskfield = inpainting(target)
        if inpaint:
            target = temp
    if inpaint_features:
        for d in range(len(covB)):  # Always use in-painting for the features
            if covB[d] != 'ndwi_py_f' and covB[d] != 'vh_cy_f' and covB[d] != 'vv_cy_f':
                data[:, :, d], _ = inpainting(data[:, :, d])

    # Set background to 0
    target[target == -2] = 0
    data[data == -2] = 0

    # Missing nitrogen data at the boundaries is be replaced by the base nitrogen rate
    N = data[:, :, 0]
    N[N == -1] = base_N
    data[:, :, 0] = N

    # Apply median filter to radar images
    if 'RADAR' in mode:
        data[:, :, -1] = ndimage.median_filter(data[:, :, -1], size=3)
        data[:, :, -2] = ndimage.median_filter(data[:, :, -2], size=3)

    if not read_column:
        if not test:
            return target.astype('float32'), data, cellids, XYcoords
        else:
            return data, cellids, XYcoords
    else:
        return data, cellids, XYcoords


def inpainting(M):
    """Use in-painting to handle missing information in raster M"""
    # Create binary mask to detect field objects
    maskfields = (M * 0).astype(np.uint8)
    maskfields[np.where(M >= 0)] = 1
    # Close object border
    maskfields = cv2.morphologyEx(maskfields, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    maskfields[0:maskfields.shape[0] - 1, 0:maskfields.shape[1] - 1] = \
        maskfields[1:maskfields.shape[0], 1:maskfields.shape[1]]  # Compensate the shift produced after closing
    # Set the first and last rows and columns to 0 so that the background can be represented as one connected component
    maskfields[0, :] = 0
    maskfields[-1, :] = 0
    maskfields[:, 0] = 0
    maskfields[:, -1] = 0
    # Detect background
    backglabels = cv2.connectedComponents(1 - maskfields)
    biggestObjectCount = 0
    backgroundLabel = None
    for b in range(1, backglabels[0]):
        count = len(np.where(backglabels[1] == b))
        if count > biggestObjectCount:
            backgroundLabel = b
            biggestObjectCount = count
    for b in range(1, backglabels[0]):
        if b != backgroundLabel:
            xm, ym = np.where(backglabels[1] == b)
            maskfields[xm, ym] = 1
    # Create binary mask for missing objects only if they're inside a field object
    mask = (M * 0).astype(np.uint8)
    mask[np.where(M < 0)] = 1
    mask = mask * maskfields
    # In-paint
    M = cv2.inpaint(M.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)
    return M, maskfields


#######################################################################################################################
# Class definition
#######################################################################################################################
class DataLoader:

    def __init__(self, field: str, training_years: list, pred_year: int, mode="AggRADARPrec", cov=None, filename=None):
        """Initialize data loader class
        @param filename: Path containing the data (CSV)
        @param field: Name of the specific field that will be analyzed.
        @param training_years: Declare the years of data that will be used for training.
        @param pred_year: The year that will be used for prediction.
        @param mode: Declare the type of variable combination that will be used.
        @param cov: Custom list of the names of the desired covariates.
        """
        self.filename = filename
        self.field = field
        self.training_years = training_years
        self.pred_year = pred_year
        self.mode = mode
        self.cov = cov

    def create_training_set(self, objective='yld'):
        """Create a training set from the training years using 5x5 patches"""
        # Extract training patches year by year
        X = []
        for year in self.training_years:
            # Read the CSV file
            Yd, Xd, _, _ = loadData(path=self.filename, field=self.field, year=int(year), cov=self.cov, inpaint=True,
                                          inpaint_features=False, base_N=120, mode=self.mode, obj=objective)
            # Combine target and data into a data cube
            rastercollection = [Yd] + list(Xd.transpose((2, 0, 1)))
            # Format list as multi-dimensional array
            rastercollection = np.array(rastercollection)
            # Extract patches
            pa = autopatches(rastercollection, window=5, plot=False)
            X = X + pa

        # Separate target and covariates
        X = np.array(X)
        Y = X[:, 0, :, :]
        X = X[:, 1:, :, :]

        # Reshape data as 4-D TENSOR
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
        # Transpose dimensions to fit Pytorch order
        X = X.transpose((0, 4, 1, 2, 3))

        # # Count the number of samples for each nitrogen rate level
        # range0_20 = []
        # range20_40 = []
        # range40_60 = []
        # range60_80 = []
        # range80_100 = []
        # range100_120 = []
        # range120_150 = []
        # for ip, s in enumerate(X):
        #     average_nitrogen = np.average(s[0, 0, 1:4, 1:4])
        #     if len(np.where(s[0, 0, 1:4, 1:4:] < 20)[0]) > 3:
        #         range0_20.append(ip)
        #     elif len(np.where(120 <= s[0, 0, 1:4, 1:4])[0]) > 3:
        #         range120_150.append(ip)
        #     else:
        #         if 20 <= average_nitrogen < 40:
        #             range20_40.append(ip)
        #         elif 40 <= average_nitrogen < 60:
        #             range40_60.append(ip)
        #         elif 60 <= average_nitrogen < 80:
        #             range60_80.append(ip)
        #         elif 80 <= average_nitrogen < 100:
        #             range80_100.append(ip)
        #         elif 100 <= average_nitrogen < 120:
        #             range100_120.append(ip)
        #
        # # Repeat under-represented samples to obtain a uniform distribution
        # max_n = int(np.max([len(range0_20), len(range20_40), len(range40_60), len(range60_80), len(range80_100),
        #                     len(range100_120), len(range120_150)]))
        # elements = list(islice(cycle(range0_20), max_n))
        # elements = list(islice(cycle(range40_60), max_n)) + elements
        # elements = list(islice(cycle(range60_80), max_n)) + elements
        # elements = list(islice(cycle(range80_100), max_n)) + elements
        # elements = list(islice(cycle(range100_120), max_n)) + elements
        # elements = list(islice(cycle(range120_150), max_n)) + elements
        # temp = np.zeros((len(elements), X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        # temp2 = np.zeros((len(elements), Y.shape[1], Y.shape[2]))
        # for n in range(len(elements)):
        #     temp[n, :, :, :, :] = X[elements[n], :, :, :, :]
        #     temp2[n, :, :] = Y[elements[n], :, :]
        # X, Y = temp, temp2

        # Shuffle dataset
        np.random.seed(seed=7)  # Initialize seed to get reproducible results
        ind = [x for x in range(X.shape[0])]
        np.random.shuffle(ind)
        X = X[ind]
        Y = Y[ind]
        return X, Y

    def load_raster(self, objective='yld'):
        return loadData(path=self.filename, field=self.field, year=int(self.pred_year), cov=self.cov, inpaint=True,
                                          inpaint_features=False, base_N=120, mode=self.mode, test=True, obj=objective)


if __name__ == '__main__':
    # Test loading real data
    ex_path = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\farmers\\' \
              'broyles_10m_yldDat_with_sentinel.csv'
    loader = DataLoader(ex_path, field='sec35middle', training_years=[2016, 2018], pred_year=2020)
    dataset = loader.create_training_set()
