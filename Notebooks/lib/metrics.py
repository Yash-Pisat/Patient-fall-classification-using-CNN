import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from numpy import argmax
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from pathlib import Path
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling2D ,Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score,ConfusionMatrixDisplay, f1_score,roc_curve, auc,precision_recall_curve, average_precision_score
from sklearn.utils import shuffle
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import confusion_matrix
import seaborn as sns

import warnings
from IPython.display import clear_output
warnings.filterwarnings('ignore')


def train_skfold_model(X, y, num_folds=5, no_epochs=75, optimizer='adam', loss_function='binary_crossentropy', verbosity=1):
    """
    This above code defines a function train_skfold_model that trains a binary classification model using the Stratified K-Fold cross-             validation method. The inputs to the function are:

    X: the input feature data.
    y: the target labels.
    num_folds: the number of folds for the cross-validation (default value is 5).
    no_epochs: the number of epochs to train the model (default value is 75).
    optimizer: the optimizer to use for training the model (default value is 'adam').
    loss_function: the loss function to use for training the model (default value is 'binary_crossentropy').
    verbosity: the verbosity level for training the model (default value is 1).

    The function splits the data into num_folds parts using StratifiedKFold and trains a model on each fold. The training data is passed to     
    the fit method of the model, which trains the model using the specified loss function and optimizer. The model performance is evaluated on  
    the validation data using the evaluate method, which returns the loss and accuracy of the model.

    The predictions on the validation data are stored in y_pred and a binary version of the predictions is created using the np.where     
    function. The ROC curve, precision-recall curve, and AUC score are calculated using the roc_curve, precision_recall_curve, and    
    roc_auc_score functions, respectively. The F1 score is calculated using the f1_score function.The confusion matrix is calculated using the
    confusion_matrix function.

    Finally, the function returns the trained model, the accuracy and loss values per fold, the confusion matrices, the ROC curve values, the
    precision-recall curve values, the AUC scores, and the F1 scores.
    
    """
    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []
    roc_values = []
    precision_recall_values = []
    confusion_matrices = []
    auc = []
    F1_score = []
    
    fold_no = 1
    for train, test in skfold.split(X, y):
        mdl = Sequential()
        mdl.add(Conv2D(8, (63, 5), activation="relu", input_shape=(63, 251, 1)))
        mdl.add(MaxPooling2D((1, 4)))
        mdl.add(Flatten())
        mdl.add(Dense(1, activation="sigmoid"))
        mdl.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        
        history = mdl.fit(X[train], y[train], epochs=no_epochs, verbose=verbosity)
        scores = mdl.evaluate(X[test], y[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        y_pred = mdl.predict(X[test,])
        y_pred_binary = np.where(y_pred > 0.10, 1, 0)
        
        fpr, tpr, thresholds = roc_curve(y[test], y_pred)
        roc_values.append((fpr, tpr, thresholds))
        
        precision, recall, thresholds_1 = precision_recall_curve(y[test], y_pred)
        precision_recall_values.append((precision, recall, thresholds_1)) 
        
        auc_score = roc_auc_score(y[test], y_pred)
        auc.append(auc_score)
    
        F1 = f1_score(y[test], y_pred_binary, average='binary')
        F1_score.append(F1)
        
        confusion_matrices.append(confusion_matrix(y[test], y_pred_binary))
        
        fold_no = fold_no + 1
    
    return mdl, acc_per_fold, loss_per_fold,confusion_matrices,roc_values,precision_recall_values,auc,F1_score


def retrain_skfold_model(mdl, X, y, num_folds, no_epochs, optimizer, loss_function, verbosity):
    """
    This above code defines a function retrain_skfold_model that retrains a binary classification model using the Stratified K-Fold cross-         validation method. The inputs to the function are:

    mdl:" Trained CNN model
    X: the input feature data.
    y: the target labels.
    num_folds: the number of folds for the cross-validation.
    no_epochs: the number of epochs to train the model.
    optimizer: the optimizer to use for training the model.
    loss_function: the loss function to use for training the model.
    verbosity: the verbosity level for training the model.

    The function splits the data into num_folds parts using StratifiedKFold and retrains the trained model on each fold. The  training data is
    passed to the fit method of the model, which trains the model using the specified loss function and optimizer. The model performance is
    evaluated on the validation data using the evaluate method, which returns the loss and accuracy of the model.

    The predictions on the validation data are stored in y_pred and a binary version of the predictions is created using the np.where function. 
    The ROC curve, precision-recall curve, and AUC score are calculated using the roc_curve, precision_recall_curve, and roc_auc_score functions,
    respectively. The F1 score is calculated using the f1_score function.The confusion matrix is calculated using the confusion_matrix function.

    Finally, the function returns the retrained model, the accuracy and loss values per fold, the confusion matrices, the ROC curve values, the
    precision-recall curve values, the AUC scores, and the F1 scores.
    """
    
    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []
    roc_values = []
    precision_recall_values = []
    confusion_matrices = []
    F1_score = []
    auc = []
    
    fold_no = 1
    for train, test in skfold.split(X, y):
        # Compile the model
        mdl.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        
        history = mdl.fit(X[train], y[train], epochs=no_epochs, verbose=verbosity)
        scores = mdl.evaluate(X[test], y[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        y_pred = mdl.predict(X[test,])
        y_pred_binary = np.where(y_pred > 0.10, 1, 0)
        
        fpr, tpr, thresholds = roc_curve(y[test], y_pred)
        roc_values.append((fpr, tpr, thresholds))
        
        precision, recall, thresholds_1 = precision_recall_curve(y[test], y_pred)
        precision_recall_values.append((precision, recall, thresholds_1)) 
        
        auc_score = roc_auc_score(y[test], y_pred)
        auc.append(auc_score)
    
        F1 = f1_score(y[test], y_pred_binary, average='binary')
        F1_score.append(F1)
        
        confusion_matrices.append(confusion_matrix(y[test], y_pred_binary))
        
        fold_no = fold_no + 1
    
    return mdl, acc_per_fold, loss_per_fold,confusion_matrices,roc_values,precision_recall_values,auc,F1_score

def store_cv_results(Config, acc_per_fold, loss_per_fold, confusion_matrices, roc_values, precision_recall_values, auc, F1_score):
    """ 
    The function store_cv_results takes several outputs of a cross-validation process as input and stores them in two pandas 
    dataframes df and df2.

    The first dataframe df stores the following information for each fold of the cross-validation:

    Configuration: the input configuration
    fold: the fold number
    accuracy: the accuracy for the fold
    loss: the loss for the fold
    auc: the area under the receiver operating characteristic curve for the fold
    F1_score: the F1 score for the fold
    confusion_matrix: the confusion matrix for the fold

    The second dataframe df2 stores additional information for each fold:

    Configuration: the input configuration
    fold: the fold number
    roc_threshold: the thresholds for the receiver operating characteristic curve
    precision: the precision values
    recall: the recall values
    precision_recall_threshold: the thresholds for precision and recall
    fpr: false positive rate values
    tpr: true positive rate values
    The function returns both dataframes.
    """
    
    results = []
    result_other = []
    
    for fold in range(len(acc_per_fold)):
        result_part1 = {
            'Configuration': Config,
            'fold': fold + 1,
            'accuracy': acc_per_fold[fold],
            'loss': loss_per_fold[fold],
            'auc': auc[fold],
            'F1_score': F1_score[fold],
            'confusion_matrix': confusion_matrices[fold]
        }
        results.append(result_part1)
        
        result_part2 = {
            'Configuration': Config,
            'fold': fold + 1,
            'roc_threshold': roc_values[fold][2],
            'precision': precision_recall_values[fold][0],
            'recall': precision_recall_values[fold][1],
            'precision_recall_threshold': precision_recall_values[fold][2],
            'fpr': roc_values[fold][0],
            'tpr': roc_values[fold][1]
        }
        result_other.append(result_part2)
        
    df = pd.DataFrame(results)
    df2 = pd.DataFrame(result_other)
    return df,df2


def plot_confusion_matrix(cm):
    """
    Plot a confusion matrix using seaborn heatmap.
    
    Parameters:
    cm (np.ndarray): Confusion matrix to be plotted.
    
    Returns:
    None: Plot the confusion matrix using seaborn heatmap.
    """
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.show()


def predict_class(models, X_tst, y_tst, test_df, threshold):
    """
    Given a list of trained models, test data and a threshold, the function predicts binary classes (0/1) for the test data.
    The predicted binary classes are calculated by applying the threshold to the predicted class probabilities.
    The function returns several results: predicted class probabilities for the test data, predicted binary classes for the
    test data, confusion
    matrices for the test data, and a contingency table of the predicted binary classes and actual classes.

    Parameters:
    models (list): A list of trained Keras models.
    X_tst (ndarray): An array of test data features.
    y_tst (ndarray): An array of actual binary class labels for the test data.
    test_df (DataFrame): A Pandas DataFrame containing the test data.
    threshold (float or list): Threshold for converting predicted class probabilities to binary classes. Can either be 
    single value or a list of values.

    Returns:
    y_pred_tst_list (list): A list of predicted class probabilities for the test data.
    y_pred_binary_tst_list (list): A list of predicted binary classes for the test data.
    cm_test_cv1_list (list): A list of confusion matrices for the test data.
    ct_list (list): A list of contingency tables of predicted binary classes and actual classes for the test data.

    """
    y_pred_tst_list = []
    y_pred_binary_tst_list = []
    cm_test_cv1_list = []
    ct_list = []

    if isinstance(threshold, list):
        i = 0
        for mdl in models:
            y_pred_tst = mdl.predict(X_tst).ravel()
            y_pred_binary_tst = np.where(y_pred_tst > threshold[i], 1, 0)
            cm_test_cv1 = confusion_matrix(y_tst, y_pred_binary_tst)
            ct = pd.crosstab(test_df['event'], y_pred_binary_tst)

            y_pred_tst_list.append(y_pred_tst)
            y_pred_binary_tst_list.append(y_pred_binary_tst)
            cm_test_cv1_list.append(cm_test_cv1)
            ct_list.append(ct)
            i = i +1
    else:
        
        for mdl in models:
            y_pred_tst = mdl.predict(X_tst).ravel()
            y_pred_binary_tst = np.where(y_pred_tst > threshold, 1, 0)
            cm_test_cv1 = confusion_matrix(y_tst, y_pred_binary_tst)
            ct = pd.crosstab(test_df['event'], y_pred_binary_tst)

            y_pred_tst_list.append(y_pred_tst)
            y_pred_binary_tst_list.append(y_pred_binary_tst)
            cm_test_cv1_list.append(cm_test_cv1)
            ct_list.append(ct)

    
    return y_pred_tst_list, y_pred_binary_tst_list, cm_test_cv1_list, ct_list


def evaluate_test(y_pred, y_test):
    """
    The evaluate_test function takes two arguments, y_pred and y_test, which are the predicted values and actual values of the       target variable respectively.

    It evaluates the model's performance using various metrics and returns the results as a tuple. The results include:

    Area Under the ROC Curve (auc_kpi)
    Average Precision Score (apc_prc)
    Threshold (threshold)
    Confusion Matrix (conf_mat_model2)
    Accuracy (accuracy)
    Precision (precision)
    Recall (recall)
    Specificity (specificity)
    F1 Score (f1)
    ROC Curve values (roc_list)
    Precision-Recall Curve values (prc_list)
    These metrics are calculated by evaluating the model on the y_test data and comparing it with y_pred. The evaluation is based on various
    techniques such as ROC Curve, Precision-Recall Curve, Confusion Matrix, and F1 Score.
    """
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    precision_prc, recall_prc, thresholds_prc = precision_recall_curve(y_test, y_pred, pos_label=1)
    auc_kpi = np.round(auc(fpr, tpr), 2)
    apc_prc = np.round(average_precision_score(y_test, y_pred), 2)
    
    i = 0
    for index, th in enumerate(tpr):
        if th >= 1:
            i = index
            break

    threshold = thresholds[i]
    pred_threshold = y_pred.copy()
    pred_threshold[pred_threshold >= threshold] = 1
    pred_threshold[pred_threshold < threshold] = 0

    conf_mat_model2 = confusion_matrix(y_test, pred_threshold) 

    accuracy = (conf_mat_model2[1,1]+conf_mat_model2[0,0]) / sum(sum(conf_mat_model2[:,:]))
    precision = conf_mat_model2[1,1] / sum(conf_mat_model2[:,1])
    recall = conf_mat_model2[1,1] / sum(conf_mat_model2[1,:])
    specificity = conf_mat_model2[0,0] / sum(conf_mat_model2[0,:])
    f1 = f1_score(y_test, pred_threshold, average='binary')
    roc_list = [fpr, tpr, thresholds]
    prc_list = [precision_prc, recall_prc, thresholds_prc]
    
    return auc_kpi, apc_prc, threshold, conf_mat_model2, accuracy, precision, recall, specificity, f1, roc_list, prc_list


def get_score(model, x_train, x_test, y_train, y_test):
    """
    Fit a model on training data and return the fit model and its score on test data.
    
    Parameters
    ----------
    model: object
        A model object with a fit and score method.
    x_train: array-like
        Training data for the independent variables.
    x_test: array-like
        Test data for the independent variables.
    y_train: array-like
        Training data for the dependent variables.
    y_test: array-like
        Test data for the dependent variables.
    
    Returns
    -------
    mdl_fit: object
        The fit model.
    score: float
        The score of the fit model on the test data.
        
    """
    
    mdl_fit = model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    return mdl_fit, score


def test_metrics(config,threshold_tst, conf_mat_model2_tst, Accuracy_tst, Precision_tst, Recall_tst, Specificity_tst, F1_tst):
    """
    This function computes and returns the test metrics of a machine learning model as a dataframe.

    Parameters:
    config (str): A string representing the configuration used for the model.
    threshold_tst (float): Threshold value used for binary classification.
    conf_mat_model2_tst (numpy.ndarray): The confusion matrix for the model.
    Accuracy_tst (float): The accuracy score for the model.
    Precision_tst (float): The precision score for the model.
    Recall_tst (float): The recall score for the model.
    Specificity_tst (float): The specificity score for the model.
    F1_tst (float): The F1 score for the model.

    Returns:
    pandas.DataFrame: A dataframe containing the test metrics of the model.
    
    """
    
    data = {'Configuration': config,
            'Accuracy': [Accuracy_tst], 
            'Precision': [Precision_tst], 
            'Recall': [Recall_tst], 
            'Specificity': [Specificity_tst], 
            'F1 Score': [F1_tst],
            'Threshold': [threshold_tst], 
            'Confusion Matrix': [conf_mat_model2_tst]}
    
    df = pd.DataFrame(data)
    
    return df


class PlotLearning(keras.callbacks.Callback):
    """
    Class that inherits from keras.callbacks.Callback and implements the method on_train_begin and on_epoch_end to create a learning curve of   
    the metrics during the training of a Keras model. The learning curve is plotted using matplotlib. The on_train_begin method initializes an
    empty dictionary metrics to store the training metrics during the training process. The on_epoch_end method appends the current values of   
    the metrics for each epoch to the metrics dictionary, and plots the learning curve for each metric in the same figure using the
    matplotlib.pyplot library. The learning curve is updated at the end of each epoch.
    
    """

    def on_train_begin(self, logs={}):   
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].set_xlabel('epoch')
            axs[i].grid()

        plt.tight_layout()
        plt.show()
        

def plot_roc_plot(roc_values_list, num_folds,auc):
     
    """
    Plots ROC (Receiver Operating Characteristic) curve and average G-Mean for a given set of ROC values and number of folds.
    
    Parameters:
    roc_values (list): List of tuples, where each tuple contains false positive rate (fpr), true positive rate (tpr), and thresholds values.
    num_folds (int): Total number of folds for which ROC values are calculated.
    
    Returns:
    None
    
    Note:
    The function also outputs the optimal threshold and average G-Mean for each fold and overall.
    """
    fig, axis = plt.subplots(1, len(roc_values_list), figsize=(20, 5))
    print(axis)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    j =0
    m=0
    
    for index, roc_values_new in enumerate(roc_values_list):
        tprs = []
        for inner_index,(fpr, tpr, thresholds) in enumerate(roc_values_new):
            ix = np.argmax(np.sqrt(tpr * (1-fpr)))
            print('For fold ' + str(inner_index+1) + ', optimal Threshold=%f, G-Mean=%.3f' % (thresholds[ix], np.sqrt(tpr[ix] * (1-fpr[ix]))))

            #axis[index].plot(fpr, tpr, 'b', alpha=0.15, marker='.', label='ROC for Fold {}'.format(inner_index+1))
            axis[index].plot(fpr, tpr, 'b', alpha=0.15, marker='.', label='Fold {} (AUC = {:.3f})'.format(inner_index+1, auc[m]))
            axis[index].scatter(fpr[ix], tpr[ix], s=20, alpha=0.5, marker='o', color='black')
            tpr_new = np.interp(base_fpr, fpr, tpr)
            tpr_new[0] = 0.0
            tprs.append(tpr_new)

            if inner_index == num_folds-1:
                axis[index].scatter(fpr[ix], tpr[ix], s=20, alpha=0.5, marker='o', color='black', label='G-mean')
                axis[index].plot([0,1], [0,1], linestyle='--', label='No Skill')
            m = m+1
        
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std

        axis[index].plot(base_fpr, mean_tprs, 'g', linewidth=2.0, label='Mean ROC')
        axis[index].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
                        
        axis[index].plot([0, 1], [0, 1],'r--')
        axis[index].axis(xmin = -0.01, xmax = 1.01)
        axis[index].axis(ymin = -0.01, ymax =1.01)        
        if  index == 0:
            axis[index].set_ylabel('True Positive Rate')
        
        axis[index].set_xlabel('False Positive Rate')
        axis[index].legend(loc = 'lower right')
        j = j + 1
    plt.show()
    
def plot_precision_recall(precision_recall_values_list, num_folds, figsize=(20, 5)):
    """
    The function plot_precision_recall_curve plots the precision-recall curve. It takes one parameter precision_recall_values, which is a list   
    of tuples of precision, recall, and thresholds values for each fold of a cross-validation. In the function,
    first a figure is created with a specified figure size. The for loop iterates through each fold of the precision_recall_values list, where
    precision, recall, and thresholds_1 are extracted from the tuple. For each fold, the F-score is calculated as the harmonic mean of
    precision and recall, and the index of the best threshold (i.e. the threshold       that maximizes the F-score) is obtained using np.argmax.
    The precision-recall curve is then plotted using the plt.plot function.

    At the end of each fold, the optimal F-score is plotted using plt.scatter, and the x-axis is labeled 'Recall' and y-axis is     labeled
    'Precision'. The legend is also added and the plot is displayed using the plt.show function.
    """
    fig, axes = plt.subplots(1, len(precision_recall_values_list), figsize=figsize)
    tprs = []
    j = 0
    
    for index, precision_recall_values in enumerate(precision_recall_values_list):
        precisions = []
        recalls = []
        for inner_index, (precision, recall, thresholds) in enumerate(precision_recall_values):
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)
            auc_prc = auc(recall, precision)
            print(f'For Config{index+1} fold {inner_index+1} Best Threshold={thresholds[ix]:.3f}, F-Score={fscore[ix]:.3f}')
            
            axes[index].plot(recall, precision, marker=",", label=f' Fold {inner_index+1} AUC={auc_prc:.3f}')
            axes[index].scatter(recall[ix], precision[ix], s=20, alpha=0.5, marker='o', color='black')
            precisions.append(precision)
            recalls.append(recall)
        
            if inner_index == num_folds-1:
                axes[index].scatter(recall[ix], precision[ix], s=20, alpha=0.5, marker='o', color='black', label='Max F-Score')
        
        if  index == 0:
            axes[index].set_ylabel('Precision')
        
        axes[index].set_xlabel('Recall')
        axes[index].legend()
        j = j + 1
    plt.show()