from YieldMapPredictor import YieldMapPredictor

if __name__ == '__main__':
    #####################################################
    # Winter What Experiments
    #####################################################
    filepath = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\farmers\\' \
               'broyles_10m_yldDat_with_sentinel.csv'
    fieldname = 'sec35middle'
    modelname = "Hyper3DNet"

    predictor = YieldMapPredictor(filename=filepath, field=fieldname, training_years=[2016, 2018], pred_year=2020,
                                  data_mode="AggRADARPrec")
    predictor.trainPreviousYears(modelType=modelname)
    result = predictor.predictYield(modelType=modelname)

    #####################################################
    # Corn Field Simulation
    #####################################################
    # filepath = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\sim_data.csv'
    # fieldname = ''
    # cvars = ['N', 'par1', 'par2', 'par3', 'par4', 'par5', 'par6', 'par7', 'par8', 'par9', 'par10', 'par11', 'par12',
    #          'par13', 'par14']
    # modelname = "Hyper3DNet"
    # goal = 'yld'
    # # method = "GAM"
    # # 10-fold cross validation
    # RMSE = []
    # prediction, target = None, None
    # for nt in range(10):
    #     print("************************************************************************************************")
    #     print("Fold " + str(nt + 1) + " / 10")
    #     print("************************************************************************************************")
    #     tyears = list(np.arange(1, 11))
    #     tyears.remove(10 - nt)
    #     pyear = 10 - nt
    #     predictor = YieldMapPredictor(filename=filepath, field=fieldname, training_years=tyears, pred_year=pyear,
    #                                   cov=cvars)
    #     # Train and validate
    #     predictor.trainPreviousYears(epochs=500, batch_size=64, modelType=method, objective=goal)
    #     prediction = np.clip(predictor.predictYield(modelType=modelname, objective=goal), a_min=0, a_max=2E4)
    #     # Compare to the ground-truth and calculate the RMSE
    #     target, _, _ = loadData(path=filepath, field=fieldname, year=10 - nt, cov=cvars, inpaint=True,
    #                             inpaint_features=False, base_N=120, test=False, obj=goal)
    #     RMSE.append(utils.mse(prediction, target) ** .5)
    #     print("Validation RMSE = " + str(RMSE[nt]))
    # # Plot lat results for reference
    # plt.figure()
    # plt.imshow(prediction)
    # plt.title("Predicted yield map")
    # plt.axis("off")
    # plt.figure()
    # plt.imshow(target)
    # plt.title("Ground-truth yield map")
    # plt.axis("off")
