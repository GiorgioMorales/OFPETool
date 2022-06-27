import os
import glob
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.OFPETool.Predictor import YieldMapPredictor, utils
from src.OFPETool.Predictor.DataLoader import loadData
from flask import Flask, flash, request, redirect, url_for, render_template, send_file


app = Flask(__name__)


######################################################################################
#   ADDITIONAL METHODS
######################################################################################

def colorize(filename):
    """Takes a 2-D image and save it using a colormap"""
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)


######################################################################################
#   DEFINE VARIABLES
######################################################################################

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv'}

global predictor  # This will be an object of the 'YieldPredictor' class


######################################################################################
#   UPLOAD THE FILE
######################################################################################
def allowed_file(filename):
    """Verify that the file is one of the allowed types"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/<filename>')
def display_image(filename):
    """Take a temporal file and display it"""
    return redirect(url_for('static', filename=filename), code=301)


@app.route('/', methods=['POST'])
def buttons():
    ###################################################################################################################
    # START PREDICTION
    ###################################################################################################################
    loaded_model = False
    if request.form.get('predict') == "Start Prediction":
        returnMessages = False  # Will be true if any input argument is not correct
        modelNames = []
        filename = None

        # Check filepath
        if 'file' not in request.files:
            flash('No valid file selected')
            returnMessages = True
        filePath = request.files['file']
        if filePath.filename == '':
            flash('No valid file selected')
            return redirect(request.url)
        else:
            data = pd.read_csv(filePath, low_memory=False)
            df = pd.DataFrame(data)

            # Check field
            fieldName = request.form['fieldName']
            df2 = df[df['field'] == fieldName]
            if len(df2) == 0:
                flash('The field name "' + fieldName + '" does not exist in the CSV file', 'error')
                returnMessages = True

            # Check loaded model
            modelFile = request.files['modelFile']
            model_path, stats_path = None, None
            if modelFile.filename != '':  # If filename not empty it means the user has uploaded a model
                loaded_model = True
                filename = modelFile.filename
                foldername = filename.replace('.zip', '')
                # Unzip in the upload folder
                modelFile.save(os.path.join(UPLOAD_FOLDER, filename))
                zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
                zip_ref.extractall('output/')
                zip_ref.close()
                modelNames = [filename.split('-')[1]]  # Store the model name
                # Set model path and stats path
                model_path, stats_path = 'output/' + foldername + '/' + foldername.replace('Model-', ''), '_statistics.npy'
                modelfiles = glob.glob('output/' + foldername + '/*')
                for f in modelfiles:
                    if 'statistics' in f:
                        stats_path = f

            # Check training years
            trainingYears = request.form['trainingYears']
            trainingYears = trainingYears.replace(" ", "")
            trainingYears = trainingYears.split(',')
            if any([not y.isnumeric() for y in trainingYears]):
                flash('"Training Years" must be a list of integers', 'error')
                returnMessages = True
            else:
                if any(['.' in y for y in trainingYears]):
                    flash('"Training Years" must be a list of integers', 'error')
                    returnMessages = True
                else:
                    trainingYears = [int(y) for y in trainingYears]

            # Check pred year
            predYear = request.form['predYear']
            if not predYear.isnumeric():
                flash('"Prediction Year" must be an integer number', 'error')
                returnMessages = True
            elif '.' in predYear:
                flash('"Prediction Year" must be an integer number', 'error')
                returnMessages = True
            else:
                predYear = int(predYear)
            # Check if there's data about the prediction year
            df2 = df[df['year'] == predYear]
            show_performance = True
            if len(df2) == 0:
                show_performance = False  # If there's no data, hide the performance button

            # Check prediction objective
            objective = request.form['objective']
            if objective not in df:
                flash('The objective name "' + objective + '"  is not a column of the CSV file', 'error')
                returnMessages = True

            # Check covariates
            cov = request.form['cov']
            cov = cov.replace(" ", "")
            cov = cov.split(',')
            for c in cov:
                if c not in df:
                    flash('The covariate name "' + objective + '"  is not a column of the CSV file', 'error')
                    returnMessages = True

            # Check epochs
            epochs = request.form['epochs']
            if not epochs.isnumeric():
                flash('"Epochs" must be an integer number', 'error')
                returnMessages = True
            elif '.' in epochs:
                flash('"Epochs" must be an integer number', 'error')
                returnMessages = True
            else:
                epochs = int(epochs)

            # Check model type
            if not loaded_model:
                # modelName = request.form['modelName']
                modelNames = []
                if request.form.get('Hyper3DNet') is not None:
                    modelNames.append('Hyper3DNet')
                if request.form.get('Hyper3DNetQD') is not None:
                    modelNames.append('Hyper3DNetQD')
                if request.form.get('3D-CNN') is not None:
                    modelNames.append('3D-CNN')
                if request.form.get('CNN-LF') is not None:
                    modelNames.append('CNN-LF')
                if request.form.get('RF') is not None:
                    modelNames.append('RF')
                if request.form.get('GAM') is not None:
                    modelNames.append('GAM')
                if request.form.get('RF') is not None:
                    modelNames.append('RF')
                if request.form.get('BayesianRegression') is not None:
                    modelNames.append('BayesianRegression')
                if request.form.get('MLRegression') is not None:
                    modelNames.append('MLRegression')

                if len(modelNames) == 0:
                    flash('No model type was selected', 'error')
                    returnMessages = True

            # Check batch size
            batchSize = request.form['batchSize']
            batchSize = int(batchSize)

            # Check beta
            beta = request.form['beta']
            if not beta.isnumeric():
                flash('"Epochs" must be an integer number', 'error')
                returnMessages = True
            else:
                beta = float(beta)

            if returnMessages:
                return redirect(request.url)

            # Check uncertainty
            uncertainty = True
            if request.form.get('uncertainty') is None:
                uncertainty = False

            #######################################################################
            # If the filename is valid, create a YieldMapPredictor an read the file
            #######################################################################
            global predictor
            predictor = []
            for n, modelName in enumerate(modelNames):
                # DEFINE
                predictor.append(YieldMapPredictor.YieldMapPredictor(filename=df, field=fieldName,
                                                                     training_years=trainingYears,
                                                                     pred_year=predYear, cov=cov))
                # TRAIN if no trained model was provided
                if not loaded_model:
                    predictor[n].trainPreviousYears(modelType=modelName, batch_size=batchSize, epochs=epochs,
                                                    beta_=beta, print_process=False)
                # PREDICT
                if not loaded_model:
                    results = predictor[n].predict(modelType=modelName, uncertainty=uncertainty)
                else:
                    results = predictor[n].predict(modelType=modelName, uncertainty=uncertainty, model_path=model_path,
                                                   stats_path=stats_path)

                # PLOT
                filename = "static/YMAP.png"
                if not uncertainty:
                    plt.figure()
                    ax = plt.imshow(results, vmin=0, vmax=150)
                    plt.colorbar(ax)
                    colorize(filename)
                else:
                    y_map_QD, u_map_QD, l_map_QD, PI_map_QD = results
                    figure, axis = plt.subplots(1, 3)
                    y_map_QD[y_map_QD == 0] = np.nan
                    _ = axis.flat[0].imshow(y_map_QD, vmax=150, vmin=0)
                    axis.flat[0].set_title("Estimated Yield Map", fontsize=10)
                    axis.flat[0].set_yticklabels([])
                    axis.flat[0].set_xticklabels([])
                    axis.flat[0].axes.xaxis.set_visible(False)
                    axis.flat[0].axes.yaxis.set_visible(False)
                    l_map_QD[y_map_QD == np.nan] = np.nan
                    _ = axis.flat[1].imshow(l_map_QD, vmax=150, vmin=0)
                    axis.flat[1].set_title("Lower Bound", fontsize=10)
                    axis.flat[1].set_yticklabels([])
                    axis.flat[1].set_xticklabels([])
                    axis.flat[1].axes.xaxis.set_visible(False)
                    axis.flat[1].axes.yaxis.set_visible(False)
                    u_map_QD[y_map_QD == np.nan] = np.nan
                    pl = axis.flat[2].imshow(u_map_QD, vmax=150, vmin=0)
                    axis.flat[2].set_title("Upper Bound", fontsize=10)
                    axis.flat[2].set_yticklabels([])
                    axis.flat[2].set_xticklabels([])
                    axis.flat[2].axes.xaxis.set_visible(False)
                    axis.flat[2].axes.yaxis.set_visible(False)

                    cbar = figure.colorbar(pl, ax=axis.ravel().tolist(), aspect=30, shrink=0.65)
                    cbar.ax.tick_params(labelsize=9)
                    colorize(filename)

                # PERFORMANCE
                if show_performance:
                    performancePath = (os.path.dirname(predictor[n].path_model) + '_Performance.txt').replace('Model-', '')
                    performancePath = performancePath.replace(modelName + '-', '')
                    if uncertainty:
                        y_map_QD, u_map_QD, l_map_QD, PI_map_QD = results
                        target, _, _, _ = loadData(path=df, field=fieldName, year=predYear, inpaint=True,
                                                   inpaint_features=False, test=False)
                        target = target * predictor[n].mask_field
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
                        PI_map_vec = np.reshape(PI_map_QD, (PI_map_QD.shape[0] * PI_map_QD.shape[1], 1))
                        PI_map_vec = np.array([i for m, i in zip(yield_map_vec, PI_map_vec) if m > 0])
                        MPIW0, MPIW, PICP, K, N = utils.MPIW_PICP(y_true=Ymap_vec2, y_u=y_ur, y_l=y_lr, unc=PI_map_vec)
                        with open(performancePath, 'a') as x_file:
                            x_file.write('Model type: ' + predictor[n].modelType + '\n')
                            x_file.write("RMSE = %.2f" % (float(RMSE)))
                            x_file.write('\n')
                            x_file.write("Mean PI Width (MPIW) = %.3f" % (float(MPIW)))
                            x_file.write('\n')
                            x_file.write("PI Coverage Probability (PICP) = %.3f" % (float(PICP)))
                            x_file.write('\n')
                    else:
                        y_map = results
                        target, _, _, _ = loadData(path=df, field=fieldName, year=predYear, inpaint=True,
                                                   inpaint_features=False, test=False)
                        target = target * predictor[n].mask_field
                        y_map[target == -1] = -1
                        RMSE = utils.mse(y_map, target, removeZeros=True) ** .5
                        with open(performancePath, 'a') as x_file:
                            x_file.write('Model type: ' + predictor[n].modelType + '\n')
                            x_file.write("RMSE = %.2f" % (float(RMSE)))
                            x_file.write('\n')

            return render_template('index.html', filename=filename, performance=show_performance)


###################################################################################################################
# DOWNLOAD BUTTONS
###################################################################################################################


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    if len(predictor) == 1:
        shapePath = (os.path.dirname(predictor[0].path_model) + '_Shapefile').replace('Model-', '')
        shutil.make_archive(shapePath, 'zip', shapePath)
        shutil.make_archive(os.path.dirname(predictor[0].path_model), 'zip', os.path.dirname(predictor[0].path_model))
        if filename == 'shapefile':
            return send_file((os.path.dirname(predictor[0].path_model) + '_Shapefile').replace('Model-', '') + '.zip', as_attachment=True)
        elif filename == 'model':
            return send_file(os.path.dirname(predictor[0].path_model) + '.zip', as_attachment=True)
        else:
            perfPath = (os.path.dirname(predictor[0].path_model) + '_Performance.txt').replace('Model-', '')
            return send_file(perfPath, as_attachment=True)
    else:
        shapePaths, modelPaths = [], []
        for n in range(len(predictor)):
            shapePath = (os.path.dirname(predictor[n].path_model) + '_Shapefile').replace('Model-', '')
            shutil.make_archive(shapePath, 'zip', shapePath)
            shutil.make_archive(os.path.dirname(predictor[n].path_model), 'zip',
                                os.path.dirname(predictor[n].path_model))
            shapePaths.append(shapePath)
            modelPaths.append(os.path.dirname(predictor[n].path_model))

        if filename == 'shapefile':
            myzipfile = zipfile.ZipFile("output/shapefiles.zip", mode='a')
            for shapePath in shapePaths:
                myzipfile.write(shapePath + '.zip')
            myzipfile.close()
            return send_file('output/shapefiles.zip', as_attachment=True)
        elif filename == 'model':
            myzipfile = zipfile.ZipFile("output/models.zip", mode='a')
            for modelPath in modelPaths:
                myzipfile.write(modelPath + '.zip')
            myzipfile.close()
            return send_file('output/models.zip', as_attachment=True)
        else:
            perfPath = (os.path.dirname(predictor[0].path_model) + '_Performance.txt').replace('Model-', '')
            perfPath = perfPath.replace(predictor[0].modelType + '-', '')
            return send_file(perfPath, as_attachment=True)


######################################################################################
#   RENDER
######################################################################################
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()


# Run app in terminal
# set FLASK_APP=app.py
# set FLASK_ENV=development
# flask run
