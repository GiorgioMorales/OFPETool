import os
import glob
import shutil
import zipfile
import numpy as np
import pandas as pd
from src.OFPETool.Predictor import YieldMapPredictor
import matplotlib.pyplot as plt
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
        modelName = None

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
                zip_ref.extractall(UPLOAD_FOLDER)
                zip_ref.close()
                modelName = filename.split('-')[1]  # Store the model name
                # Set model path and stats path
                model_path, stats_path = UPLOAD_FOLDER + '/' + foldername + '/' + foldername.replace('Model-', ''), '_statistics.npy'
                modelfiles = glob.glob(UPLOAD_FOLDER + '/' + foldername + '/*')
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
                modelName = request.form['modelName']

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
            # DEFINE
            predictor = YieldMapPredictor.YieldMapPredictor(filename=df, field=fieldName,
                                                            training_years=trainingYears, pred_year=predYear, cov=cov)
            # TRAIN if no trained model was provided
            if not loaded_model:
                predictor.trainPreviousYears(modelType=modelName, batch_size=batchSize, epochs=epochs, beta_=beta,
                                             print_process=False)
            # PREDICT
            if not loaded_model:
                results = predictor.predict(modelType=modelName, uncertainty=uncertainty)
            else:
                results = predictor.predict(modelType=modelName, uncertainty=uncertainty, model_path=model_path,
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

            return render_template('index.html', filename=filename)


###################################################################################################################
# DOWNLOAD BUTTONS
###################################################################################################################


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    print(filename)
    shapePath = (os.path.dirname(predictor.path_model) + '_Shapefile').replace('Model-', '')
    shutil.make_archive(shapePath, 'zip', shapePath)
    shutil.make_archive(os.path.dirname(predictor.path_model), 'zip', os.path.dirname(predictor.path_model))
    if filename == 'shapefile':
        return send_file((os.path.dirname(predictor.path_model) + '_Shapefile').replace('Model-', '') + '.zip', as_attachment=True)
    else:
        return send_file(os.path.dirname(predictor.path_model) + '.zip', as_attachment=True)


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
