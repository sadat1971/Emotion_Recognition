#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:14:58 2019

@author: sadat
"""
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.utils import np_utils
import keras
from sklearn.model_selection import LeaveOneGroupOut
from keras.callbacks import EarlyStopping
from sklearn.naive_bayes import GaussianNB



class ML_SVM:
    
    """
       This class will create model for SVM with RBF kernel
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
        
    def SVM_RBF_gridsearch(self, nfolds):
        # This function will search the best combination of given C and gamma 
        # values. 'nfolds' is number of folds to be used for CV search of
        # parameters
        
        C = [0.001, 0.01, 0.1, 1, 10]
        gamma = [0.001, 0.01, 0.1, 1]
        Param_tunable = {'C': C, 'gamma': gamma}
        Optimized_model = GridSearchCV(svm.SVC(kernel='rbf'), 
                                       Param_tunable, cv=nfolds, verbose = 1, 
                                       n_jobs = -1)
        Optimized_model.fit(self.X, self.Y)
        return Optimized_model
        
        
    def SVM_performance(self, model, X_test, Y_test):
        prediction = model.predict(X_test)
        return confusion_matrix(Y_test, prediction), recall_score(Y_test, 
                               prediction, average='macro')
    
    
    
class ML_RF:
    
    """
       This class will create model for RF
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
        
    def RF_gridsearch(self, nfolds):
        # This function will search the best number of estimators.
        #. 'nfolds' is number of folds to be used for CV search of
        # parameters
        
        
        Estimators = [80, 100, 120]
        Param_tunable = {'n_estimators': Estimators}
        Optimized_model = GridSearchCV(RandomForestClassifier(), Param_tunable, 
                                       cv=nfolds, verbose = 1, n_jobs = -1)
        Optimized_model.fit(self.X, self.Y)
        return Optimized_model
        
        
    def RF_performance(self, model, X_test, Y_test):
        prediction = model.predict(X_test)
        return confusion_matrix(Y_test, prediction), recall_score(Y_test, 
                               prediction, average='macro')
                                
       
class ML_FC_ANN:
    
    """ 
       This class will create a fully connected deep neural network model
    """
    def __init__(self, X, Y, sp):
        self.X = X
        self.Y = Y
        self.sp = sp
    
    def FC_model(self, layer_size):
        
#        This model is created after some peliminary analysis 
#        referring some prior work in this field.
        
        FCmodel = Sequential()
        FCmodel.add(Dense(layer_size, activation='relu'))
        FCmodel.add(Dropout(0.5))
        FCmodel.add(Dense(layer_size, activation='relu'))
        FCmodel.add(Dropout(0.5))
        FCmodel.add(Dense(layer_size, activation='relu'))
        FCmodel.add(Dropout(0.5))
        FCmodel.add(Dense(4, activation='softmax'))
        return FCmodel
                

    def hyperparameter_tuning(self):

#       This function will take in our defined FC_model and output the 
#       cross validated layer size

        logo_FC = LeaveOneGroupOut()
        X_tr = self.X
        Y_int_tr = self.Y
        grp_tr = self.sp
        # UWR_cv stores unweighted recall for our cross-validation purpose
        UWR_cv = np.ones((9, 3)) 
        Layer_list = [128, 256, 512] #List of layers to be cross validated
        row = 0
        for tr, val in logo_FC.split(X_tr, Y_int_tr, grp_tr):
            col = 0
            for layers in Layer_list:
                callbacks = [EarlyStopping(monitor='val_acc', patience=5)]
                # create model
                model = self.FC_model(layers)
                # Compile model
                ADAM = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, 
                                             beta_2=0.999, epsilon=None, 
                                             decay=0.0, amsgrad=False)
                model.compile(loss='categorical_crossentropy', 
                              optimizer = ADAM, metrics=['accuracy'])
                Y = np_utils.to_categorical(Y_int_tr[tr])
                model.fit(X_tr[tr], Y, epochs=50, batch_size=256, 
                                    validation_split=0.2, 
                                    callbacks=callbacks, verbose=0)
                X_pred = model.predict(X_tr[val, :])
                Y_pred = np.argmax(X_pred, axis=1) 
                UWR_cv[row, col] = recall_score(Y_int_tr[val], Y_pred, 
                                           average='macro')
                col = col + 1                  
            row = row + 1    
        return Layer_list[np.argmax(np.sum(UWR_cv, axis=0))]
    
    
    def model_train(self, layer_size):
        # After the hyperparameter tuning, we want our model to be trained. The
        # function takes in the tuned parameter "layer size" and use it to
        # train the FC model
                
        callbacks = [EarlyStopping(monitor='val_acc', patience=5)] #early stop
        model = self.FC_model(layer_size)
        ADAM = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
                                     epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss = 'categorical_crossentropy', optimizer = ADAM, 
                      metrics = ['accuracy'])
        Y = np_utils.to_categorical(self.Y)
        history = model.fit(self.X, Y, epochs=50, batch_size=256, 
                            validation_split=0.2, callbacks=callbacks, 
                            verbose=0)
        #actual training
        model_tr = self.FC_model(layer_size)
        # Compile model
        ADAM = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
                                     epsilon=None, decay=0.0, amsgrad=False)
        model_tr.compile(loss = 'categorical_crossentropy', optimizer = ADAM, 
                         metrics = ['accuracy'])
        model_tr.fit(self.X, Y, epochs = len(history.history['acc']), 
                     batch_size=256, verbose=0)
        return model_tr

    
    def NN_performance(self, model, X_test, Y_test):
        prediction = model.predict(X_test)
        Y_pred  = np.argmax(prediction, axis=1)
        return confusion_matrix(Y_test, Y_pred), recall_score(Y_test, 
                               Y_pred, average='macro') 
        
        
        
class baseline_Gaussian_NB:
    
    """ This class performs the task of applying gaussian naive byes for emotion
    recogition """
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def baseline_train(self):
        gnb_clf = GaussianNB()
        gnb_clf.fit(self.X, self.Y)
        return gnb_clf
    
    def gnb_baseline_performance(self, model, X_test, Y_test):
        prediction = model.predict(X_test)
        return confusion_matrix(Y_test, prediction), recall_score(Y_test, 
                               prediction, average='macro')
        
        
        
    

        
        
        
        
        
        
            