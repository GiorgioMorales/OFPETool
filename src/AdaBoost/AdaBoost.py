from src.AdaBoost.FFNN import *
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from keras.utils import Sequence
from keras.wrappers.scikit_learn import KerasRegressor
from statsmodels.stats.weightstats import DescrStatsW
import csv


class SamplingSequence(Sequence):

    def __init__(self, features, labels, batch_size, probabilities):
        self.features = features
        self.labels = labels
        self.x, self.y = features, labels
        self.epoch = 0
        self.batch_size = batch_size
        self.N = 2
        self.probabilities = probabilities

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.epoch % self.N == 0:
            pass
            sub_data = np.random.choice(np.concatenate(self.features, self.labels, axis=1),
                                        size=len(self.probabilities) - (len(self.probabilities) * 0.1),
                                        p=self.probabilities)
            self.y = sub_data[-1]
            self.x = np.delete(sub_data, -1, 1)
        self.epoch += 1


class AdaBoost:
    def __init__(self, X, y, nr_of_models, delta_threshold=0, learning_rate=0.01, hidden_layer=100, epochs=250):
        self.features = X  # data[:,1:]
        self.labels = y
        self.amount = nr_of_models
        self.weights = []
        self.probabilities = []
        self.error_rates = []
        self.beta_vals = []
        self.alpha_vals = []
        self.model_weights = []
        self.weak_models = []
        self.learning_rate = learning_rate
        self.delta = delta_threshold
        self.batch_size = 32
        self.epochs = epochs
        self.hidden_layer = hidden_layer

    #################################################
    # DIFFERENT ERROR CALCULATIONS					#
    # Original AdaBoost classification error:		#
    # Sum of weights where prediction != label,		#
    # divided by sum of all weights.					#
    #################################################

    """
    Bertoni et al. (2014)
    Error calculation based on R delta.
    """

    def calculate_rdelta_error(self, predictions, actual, final=False):
        faults = []
        if not final:
            for i, pred in enumerate(predictions):
                single_error = abs(pred - actual[i]) - self.delta
                if single_error >= 0:
                    faults.append(1)
                else:
                    faults.append(0)
        else:
            for i, pred in enumerate(predictions):
                if (self.delta - abs(pred - actual[i])) >= 0:
                    faults.append(1)
                else:
                    faults.append(0)
        error = np.sum(np.multiply(self.probabilities, faults))
        return error, faults

    """
    Liu et al. (2015)
    Generalized error calculation for regression.
    Average of errors for each point.
    Where single point error is calculated by taking the difference of 
    the predicted and actual value and dividing by original value.
    """

    def calculate_average_error(self, predictions, actual):
        faults = [abs(actual[i] - pred) / actual[i] for i, pred in enumerate(predictions)]
        error = np.sum(faults) / len(actual)
        return error, faults

    """
    Error calculation with tolerance/approximation
    """

    def calculate_approximation_error(self, predictions, actual):
        faults = [self.approximate(pred, actual[i], tolerance=self.delta) for i, pred in enumerate(predictions)]
        scaler = preprocessing.MinMaxScaler()
        faults = scaler.fit_transform(np.reshape(faults, (-1, 1)))
        error = np.sum(faults) / len(actual)
        print("ERROR: ", error)
        return error, faults

    def approximate(self, x, y, tolerance=1.5):
        if abs(x - y) <= tolerance:
            return 0
        else:
            return np.asscalar(abs(x - y))

    #################################################
    # MODEL WEIGHT CALCULATIONS                      #
    #################################################

    def calculate_model_weight(self, error):
        model_weight = 0.5 * np.log((1 - error) / error)
        self.model_weights.append(model_weight)
        print("model weight: ", model_weight)

    def calculate_m_weight_rdelta(self, beta):
        alpha = np.log(1 / beta)
        self.model_weights.append(alpha)

    def calculate_beta(self, error):
        # log(  (1-error)/error)
        # beta = 0.5 * np.log((1-error)/error)
        beta = error / (1 - error)
        self.beta_vals.append(beta)
        return beta

    #################################################
    # DATA WEIGHT UPDATE CALCULATIONS			#
    #################################################
    """	
    Bengio & Schwenk (02)
    But training time was lowest with the E method (with stochastic gradient descent),
    which samples each new training pattern from the original data with the AdaBoost weights.
    Although our experiments are insu!cient to conclude, it is possible that the weighted
    training" method (W) with conjugate gradients might be faster than the others for small
    training sets
    """

    """
    Bertoni et al. (2015) Delta based weight updates
    """

    def update_delta_weights(self, faults, beta):
        updated_weights = []
        for i, f in enumerate(faults):
            val = np.power(beta, (1 - f))
            updated_weights.append(val)
        return updated_weights

    """
    Liu et al. (2015)
    """

    def update_simple_weights(self, faults, beta):
        updated_weights = []
        for i, f in enumerate(faults):
            updated_weights.append(np.power(beta, (-f)))
        return updated_weights

    """
    Approximation weight update
    """

    def update_approximate_weights(self, faults, beta):
        updated_weights = []
        for i, f in enumerate(faults):
            if f != 0:
                updated_weights.append(np.power(beta, -f))
            else:
                updated_weights.append(np.power(1, 10000000000))
        return updated_weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    #################################################
    # ACTUAL BOOSTING ALGORITHM						#
    # Trains all weak models and saves information 	#
    # on each of these models.						#
    #################################################
    def boost(self, type="", weight_sampling=False, models_path=None):
        iter = 1
        while iter <= self.amount:
            print(str(iter) + " / " + str(self.amount))
            model = None
            if weight_sampling:
                seq = SamplingSequence(self.features, self.labels,
                                       self.batch_size, np.reshape(self.probabilities, (self.probabilities.shape[0],)))
                model = FFNN(seq, learning_rate=self.learning_rate, batch_size=self.batch_size)
            else:
                # print(self.probabilities)
                model = FFNN(self.features, labels=self.labels, dataweights=np.reshape(self.probabilities,
                                                                                       (self.probabilities.shape[0],)),
                             learning_rate=self.learning_rate, batch_size=self.batch_size,
                             hidden_layer=self.hidden_layer, epochs=self.epochs)

            predictions = model.simpleTrain(models_path=models_path, iteration=iter)

            if type == "rdelta":
                self.rdelta_boost(predictions)
            elif type == "simple":
                self.simple_boost(predictions)
            else:
                self.custom_boost(predictions)

                # error = model.history.history['loss'][-1]

                # Convergence only happens when error is lower than 50%
                # if (error > 1):
                #   continue
                """
                self.weights = -self.weights
                model.setDataWeights(self.weights)
                predictions = model.simpleTrain()
                error = model.history.history['loss'][-1]
                """

            self.weak_models.append(model)
            iter += 1
        # final_predictionsv = []
        if type == "rdelta":
            final_predictionsv = self.combine_models_delta()
        else:
            final_predictionsv = self.combine_models_weighted_average()
        return final_predictionsv

    def rdelta_boost(self, predictions):
        error, faults = self.calculate_rdelta_error(predictions, self.labels)
        print("ERROR: ", error)
        self.error_rates.append(error)
        beta = self.calculate_beta(error)
        self.calculate_m_weight_rdelta(beta)
        updated_weights = self.update_delta_weights(faults, beta)
        self.weights = [w * updated_weights[i] for i, w in
                        enumerate(self.weights)]  # UPDATE WEIGHTS FOR MISSCLASSIFIED DATA POINTS
        self.probabilities = np.divide(self.weights, np.sum(self.weights))

    def simple_boost(self, predictions):
        error, faults = self.calculate_average_error(predictions, self.labels)
        print("ERROR: ", error)
        self.error_rates.append(error)
        beta = self.calculate_beta(error)
        self.calculate_model_weight(error)
        updated_weights = self.update_simple_weights(faults, beta)
        temp_weights = [np.asscalar(w * updated_weights[i]) for i, w in
                        enumerate(self.weights)]  # UPDATE WEIGHTS FOR MISSCLASSIFIED DATA POINTS
        self.weights = np.divide(temp_weights, np.sum(temp_weights))
        self.probabilities = self.weights

    def custom_boost(self, predictions):
        error, faults = self.calculate_approximation_error(predictions, self.labels)
        self.error_rates.append(error)
        beta = self.calculate_beta(error)
        self.calculate_model_weight(error)
        updated_weights = self.update_approximate_weights(faults, beta)
        temp_weights = [w * updated_weights[i] for i, w in
                        enumerate(self.weights)]  # UPDATE WEIGHTS FOR MISSCLASSIFIED DATA POINTS
        self.weights = np.divide(temp_weights, np.sum(temp_weights))
        self.probabilities = self.weights

    #################################################
    # DIFFERENT MODEL COMBINATIONS					#
    #################################################

    """
    Bertoni et al. (2014): R Delta 
    For test set model: use resulting weights to average 
    over values for each model to get final result.
    """

    def combine_models_delta(self):
        final_models = []
        # [np.sum([self.alpha_vals[i]* self.calculate_rdelta_error(model.predictions,labels[train], True)])
        # for i, model in enumerate(self.weak_models)]
        for i, model in enumerate(self.weak_models):
            error, faults = self.calculate_rdelta_error(model.predictions, self.labels, final=True)
            final_models.append(self.model_weights[i] * np.array(faults))
        model_indeces = np.argmax(np.asarray(final_models), axis=0)
        # print([np.asscalar(np.asarray(test_pred)[model,i]) for i, model in enumerate(model_indeces)])
        return [(self.weak_models[model].predictions[i]).item() for i, model in enumerate(model_indeces)]

    """
    Liu et al. (2014): Simple	
    
    def combine_models_simple(self):
        total_predictions = []
        norm_weights = np.divide(self.model_weights, np.sum(self.model_weights))
        for i,model in enumerate(self.weak_models):
            total_predictions.append(norm_weights[i]*model.predictions)
        final_predictions = np.sum(total_predictions,axis=0)
        print(final_predictions)
        return final_predictions
    """

    """
    Zhou et al. & Liu et al. (2014)
    Weighted average of models
    """

    def combine_models_weighted_average(self):
        total_predictions = []
        for i, model in enumerate(self.weak_models):
            total_predictions.append(model.predictions)
        return np.average(total_predictions, axis=0, weights=self.model_weights)

    def predict(self, X_test):
        test_pred = []
        for i, model in enumerate(self.weak_models):
            test_pred.append(model.trainedModel.predict(X_test))
        final_test = np.average(test_pred, axis=0, weights=self.model_weights)
        return final_test

    def predictUncertainty(self, X_test):
        test_pred = []
        for i, model in enumerate(self.weak_models):
            test_pred.append(model.trainedModel.predict(X_test))

        test_pred = np.array(test_pred)
        stds = []
        for v in range(test_pred.shape[1]):
            weighted_stats = DescrStatsW(test_pred[:, v, 0], weights=self.model_weights, ddof=0)
            stds.append(weighted_stats.std)
        return np.average(test_pred, axis=0, weights=self.model_weights), np.array(stds)

    """
    Scikit learn adaboost implementation
    """

    def sk_boost(self):
        features = self.features
        labels = self.labels
        ffnn = FFNN(features, labels)
        model = ffnn.model
        skmodel = KerasRegressor(build_fn=model, nb_epoch=200, batch_size=64, verbose=2)
        boost = AdaBoostRegressor(base_estimator=skmodel, n_estimators=1000, learning_rate=.1, loss="linear")
        seed = 5
        np.random.seed(seed)
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        totalTestPredictions = []

        fold = 1
        for train, test in kfold.split(features, labels):
            boost.fit(features[train], labels[train])
            totalTestPredictions.append(boost.predict(features[test]))


def run_k_fold(X, y, path):
    seed = 5
    np.random.seed(seed)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    final_test_predictions = []
    testErrorOverTime = []
    trainErrorOverTime = []
    final_predictions = []
    original_test = []

    param_nr_models = 50
    param_lr = 0.01
    param_hiddennodes = 100
    param_epochs = 500
    param_delta = 1.5
    boosting_type = ""

    with open(path + "/parameters", "w") as file:
        wr = csv.writer(file)
        wr.writerows([["number of models", str(param_nr_models)], ["learning rate", str(param_lr)],
                      ["hidden nodes", str(param_hiddennodes)], ["epochs", str(param_epochs)],
                      ["boosting method", str(boosting_type)], ["delta bound", str(param_delta)]])

    fold = 0
    for train, test in kfold.split(X, y):
        print("Starting fold " + str(fold))
        ab = AdaBoost(X[train], y[train], param_nr_models, delta_threshold=param_delta, learning_rate=param_lr,
                      hidden_layer=param_hiddennodes, epochs=param_epochs)
        ab.set_weights(np.ones(X[train].shape[0]) / X[train].shape[0])
        ab.set_probabilities(np.divide(ab.get_weights(), np.sum(ab.get_weights())))
        final_predictions.append(ab.boost(
            type=boosting_type))  # type sets which adaboost you wish to perform: rdelta, simple or approximation
        # for loop in range(self.amount):
        #   sumTrain = np.zeros(features[train].shape[0])
        #   sumTest = np.zeros(features[test].shape[0])
        final_test_predictions.append(ab.predict(X[test]))
        original_test.append(y[test])

        # trainErrorOverTime.append(np.sum(np.sign(sumTrain) != labels[train])/labels[train].shape[0])
        # testErrorOverTime.append(np.sum(np.sign(sumTest) != labels[test])/labels[test].shape[0])

        # plt.scatter(y[test], final_test_predictions[fold], edgecolors=(0, 0, 0))
        # plt.xlabel("Actual values")
        # plt.ylabel("Predictions")
        # plt.xlim(plt.xlim())
        # plt.ylim(plt.ylim())
        # plt.plot([y.min(), y.max()], [y.min(), y.max()], lw=2)
        # plt.savefig(path + "/adaboost_" + boosting_type + "_fold_" + str(fold) + '_predictions_vs_actual.png')
        # plt.close()

        # plt.figure()
        # plt.plot(range(1,self.amount+1), trainErrorOverTime, label="Training error")
        # plt.plot(range(1,self.amount+1), testErrorOverTime, label="Testing error")
        # plt.show()
        # plt.clf()

        fold = fold + 1
    return np.asarray(final_test_predictions), np.asarray(original_test)

