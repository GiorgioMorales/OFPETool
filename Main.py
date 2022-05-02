import utils
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import loadData
from YieldMapPredictor import YieldMapPredictor


if __name__ == '__main__':
    #####################################################
    # Winter What Experiments
    #####################################################
    filepath = 'C:\\Users\\w63x712\\Documents\\Machine_Learning\\OFPE\\Data\\CSV_Files\\farmers\\' \
               'wood_10m_yldDat_with_sentinel.csv'
    fieldname = 'henrys'
    modelname = "Hyper3DNetQD"

    for b1 in np.arange(5, 10, 0.5):
        print("*************************************")
        print("Tuning with beta = " + str([b1]))
        print("*************************************")
        predictor = YieldMapPredictor(filename=filepath, field=fieldname, training_years=[2016, 2018], pred_year=2020,
                                      data_mode="AggRADARPrec")
        predictor.trainPreviousYears(modelType=modelname, batch_size=16, epochs=150, beta_=b1, print_process=True)
        # y_map_QD = predictor.predictYield(modelType=modelname)
        y_map_QD, u_map_QD, l_map_QD, PI_map_QD = predictor.predictYield(modelType=modelname, uncertainty=True)  #
        target, _, _, _ = loadData(path=filepath, field=fieldname, year=2020, inpaint=True,
                                   inpaint_features=False, base_N=120, test=False)
        target = target * predictor.mask_field
        y_map_QD[target == -1] = -1
        u_map_QD[target == -1] = -1
        l_map_QD[target == -1] = -1
        PI_map_QD[target == -1] = -1
        RMSE = utils.mse(y_map_QD, target, removeZeros=True) ** .5
        Ymap_vec = np.reshape(target, (y_map_QD.shape[0] * y_map_QD.shape[1], 1))
        yield_map_vec = np.reshape(y_map_QD, (y_map_QD.shape[0] * y_map_QD.shape[1], 1))
        y_ur = np.reshape(u_map_QD, (y_map_QD.shape[0] * y_map_QD.shape[1], 1))
        y_ur = np.array([i for m, i in zip(yield_map_vec, y_ur) if m > 0])
        y_lr = np.reshape(l_map_QD, (y_map_QD.shape[0] * y_map_QD.shape[1], 1))
        y_lr = np.array([i for m, i in zip(yield_map_vec, y_lr) if m > 0])
        # Vectorize maps and remove points outside the field
        Ymap_vec2 = np.array([i for m, i in zip(yield_map_vec, Ymap_vec) if m > 0])
        yield_map_vec2 = np.array([i for i in yield_map_vec if i > 0])
        PI_map_vec = np.reshape(PI_map_QD, (PI_map_QD.shape[0] * PI_map_QD.shape[1], 1))
        PI_map_vec = np.array([i for m, i in zip(yield_map_vec, PI_map_vec) if m > 0])
        MPIW0, MPIW, PICP, K, N = utils.MPIW_PICP(y_true=Ymap_vec2, y_u=y_ur, y_l=y_lr, unc=PI_map_vec)
        print("RMSE: " + str(round(RMSE, 2)))
        print("MPIW: " + str(round(MPIW0, 3)))
        print("# capured: " + str(K) + " / " + str(N))
        print("MPIWcapt: " + str(round(MPIW, 3)))
        print("PICP: " + str(round(PICP, 3)))
        plt.figure()
        plt.imshow(y_map_QD, vmin=0, vmax=150)
        plt.title("PI map")
        plt.axis("off")

        plt.figure()
        plt.imshow(PI_map_QD, vmin=0, vmax=80)
        plt.title("PI map")
        plt.axis("off")
