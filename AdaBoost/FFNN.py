import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import History
from sklearn.preprocessing import MinMaxScaler
from keras.layers import *
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import tensorflow as tf
import os

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

class FFNN:
    def __init__(self, data, labels=[], dataweights=[], learning_rate=1.0, batch_size=96, hidden_layer=512, epochs=200):
        self.input = data  # data[:,1:]
        self.regressionLabels = labels  # data[:,0]
        if dataweights == []:
            self.dataweights = [1 for l in labels]
        else:
            self.dataweights = dataweights
        self.predictions = []
        self.trainedModel = Sequential()
        self.history = History()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_nodes = hidden_layer
        self.epochs = epochs

    def model(self):
        output_size = 1
        input_size = self.input[0].shape
        activation_function = "relu"
        model = Sequential()
        model.add(Dense(self.hidden_nodes, input_shape=input_size, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
        model.add(Dense(32, activation=activation_function))
        model.add(Dense(output_size, activation=activation_function))
        model.compile(loss="mean_squared_error",
                      optimizer=RMSprop(self.learning_rate), metrics=["mae"])
        return model

    def getDataWeights(self):
        return self.weights

    def setDataWeights(self, weights):
        self.dataweights = weights

    # self.initializer = keras.initializers.Constant(value=weights)

    def simpleTrain(self, models_path=None, iteration=None):
        # scaler = MinMaxScaler()
        # regressionScaler = MinMaxScaler()
        features = self.input
        labels = self.regressionLabels.reshape(-1, 1)
        model = self.model()
        if isinstance(self.input, keras.utils.Sequence):
            self.history = model.fit_generator(features,
                                               epochs=self.epochs,
                                               verbose=0)
        elif labels.any():
            print("fitting")
            checkpoint_filepath = models_path + "//model-" + str(iteration) + ".h5"
            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor='val_mean_absolute_error',
                mode='min',
                save_best_only=True)
            self.history = model.fit(features, labels, sample_weight=self.dataweights,
                                     validation_split=0.1,
                                     batch_size=96,
                                     epochs=self.epochs,
                                     verbose=1,
                                     callbacks=[model_checkpoint_callback]
                                     )

        self.predictions = model.predict(features)
        self.trainedModel = model

        # If a path is given, save the model
        if models_path is not None:
            model.save(models_path + "//model-" + str(iteration) + ".h5")

        return self.predictions

    def train(self):
        scaler = MinMaxScaler()
        regressionScaler = MinMaxScaler()
        features = scaler.fit_transform(self.input)
        labels = self.regressionLabels.reshape(-1, 1)
        # labels = regressionScaler.fit_transform(labels)

        seed = 5
        np.random.seed(seed)
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        totalPredictions = []
        fold = 1
        for train, test in kfold.split(features, labels):
            model = self.model()
            # stop_criteria = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
            history = model.fit(features[train], labels[train],
                                validation_split=0.1,
                                batch_size=32,
                                epochs=200,
                                verbose=2
                                )
            """
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error')
            plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
            plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label = 'Val loss')
            plt.legend()
            plt.ylim(plt.ylim())
            plt.xlim(plt.xlim())
            plt.show()
            plt.savefig("fold_" + str(fold) + '_train_val_score.png')
            plt.clf()
            """

            [loss, mae] = model.evaluate(features[test], labels[test], verbose=2)
            print("Testing set Mean Abs Error: {:.4f}".format(mae))

            predictions = model.predict(features[test])
            totalPredictions.append(predictions)
            """
            plt.scatter(labels[test], predictions, edgecolors=(0, 0, 0))
            plt.xlabel("Actual values")
            plt.ylabel("Predictions")
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], lw=2)
            plt.show()
            plt.savefig("fold_" + str(fold) + '_predictions_vs_actual.png')
            plt.close()
            """
            fold = fold + 1
        return totalPredictions

    # result = np.asarray(scores)
    # averages = [np.average(result, axis=0)]
    # print("10-fold average:", averages)


class SAE:
    def __init__(self, data):
        self.input = data[:, 1:]
        self.regressionLabels = data[:, 0]

    def model(self):
        input_size = self.input[0].shape
        output_size = 1

        model = Sequential()
