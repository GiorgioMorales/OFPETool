import matplotlib.pyplot as plt
from YieldMapPredictor import YieldMapPredictor


if __name__ == '__main__':
    #####################################################
    # INPUT
    #####################################################
    """
    Input parameters:***************************************************************************************************
    @param filepath: Path containing the data (CSV). The CSV files I used include information from the Sentinel-1 
                     satellite. Download the from here: https://montanaedu-my.sharepoint.com/:f:/g/personal/w63x712_msu_montana_edu/EqtdDCcAjYhKqxe3e0v9YvsBH7BQoYFDu0MiXLZC_gv_5g?e=CEOPQJ
    @param fieldName: Name of the specific field that will be analyzed. Ex: 'sec35middle', 'henrys'
    @param trainingYears: Declare the years of data that will be used for training (List). Ex: [2016, 2018]
    @param objective: Defines what is the target variable. Default: 'yld'. Other options: 'pro', 'EONR' 
    @param predYear: The year that will be used for prediction (Int). Ex: 2020
    @param dataMode: Specified the predetermined  combination of covariates that will be used. Options: 'All', 
                      'AggRADAR', 'AggRADARPrec' (Default). Check DataLoader.py:126 for more details
    @param cov: This argument is used in case 'data_mode' is not used. It's an explicit list of covariate names that 
                will be used for the analysis. Ex: ['N', 'NDVI', 'elev']
    @param modelName: Options: 'Hyper3DNet', 'Hyper3DNetQD', 'AdaBoost', 'Russello', 'CNNLF', 'SAE', 'RF', 'GAM', and 
                      'MLRegression'. Note that 'Hyper3DNet' uses a single CNN for prediction. On the other hand, 
                      'Hyper3DNetQD' uses two CNNs to generate point estimates as well as the upper and lower bounds
                      
    Training parameters:***********************************************************************************************
    @param batchSize: Recommended for sec35middle=128. Recommended for henrys=32.
    @param epochs: Usually, no more than 500
    @param beta: Only used when generating PIs with 'Hyper3DNetQD'. Hyperparameter that balance the PI width and 
                  probability coverage. Recommended for sec35middle=6.75. Recommended for henrys=5
    @param printProcess: If True, print the results on the training and validation sets when training neural nets
                      
    Testing parameters:************************************************************************************************
    @param uncertainty: If False, "predict" only returns the prediction map. If True, "predict" returns the prediction 
                        map, the upper bounds, the lower bounds, and the uncertainty map.
    """
    filepath = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\farmers\\' \
               'wood_10m_yldDat_with_sentinel.csv'
    fieldName = 'henrys'
    trainingYears = [2016, 2018]
    predYear = 2020
    modelName = 'Hyper3DNetQD'
    # Note that 'dataMode' and 'obj' were not specified, so dataMode='AggRADARPred' and obj='yld' by default.
    batchSize = 32  # For henrys: batchSize = 32; for sec35middle: batchSize = 128
    epochs = 300
    beta = 5.5  # For henrys: beta = 5.5; for sec35middle: beta = 7
    printProcess = True
    uncertainty = True

    #####################################################
    # TRAIN / PREDICTION
    #####################################################
    """A prediction object is created using the input arguments declared above. The pubic methods of this class are:
       * trainPreviousYears (YieldMapPredictor.py:197)
       * predict (YieldMapPredictor.py:229)
       * modifyPrescription (YieldMapPredictor.py:176) 
    """
    # DEFINE
    predictor = YieldMapPredictor(filename=filepath, field=fieldName, training_years=trainingYears, pred_year=predYear)
    # TRAIN
    predictor.trainPreviousYears(modelType=modelName, batch_size=batchSize, epochs=epochs, beta_=beta,
                                 print_process=printProcess)
    # PREDICT
    results = predictor.predict(modelType=modelName, uncertainty=uncertainty)
    # PLOT
    if uncertainty:
        y_map_QD, u_map_QD, l_map_QD, PI_map_QD = results
        plt.figure()
        plt.imshow(y_map_QD, vmin=0, vmax=150)
        plt.colorbar()
        plt.title("Prediction map")
        plt.axis("off")
        plt.figure()
        plt.imshow(PI_map_QD, vmin=0, vmax=80)
        plt.colorbar()
        plt.title("Uncertainty map (PI width)")
        plt.axis("off")
    else:
        plt.figure()
        plt.imshow(results, vmin=0, vmax=150)
        plt.colorbar()
        plt.title("Prediction map")
        plt.axis("off")
