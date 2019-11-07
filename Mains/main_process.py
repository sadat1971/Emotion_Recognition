#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:04:39 2019

@author: sadat
"""

from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from Codes.loading_data import load_data
from Codes.Machine_Learning_Algo import ML_SVM, ML_RF, ML_FC_ANN


def main():
    
    # Conf_Mat and UWR will keep saving the confusion matrix and the unweighted 
    # recall across the speakers
    Conf_Mat = pd.DataFrame(columns = ["speakers", "SVM", "RF", "ANN"]) 
    UWR = pd.DataFrame(columns = ["speakers", "SVM", "RF", "ANN"]) 
    
    speakers = [i+1 for i in range(10)] #speakers list 1 to 10
    Conf_Mat["speakers"] = speakers
    UWR["speakers"] = speakers
    
    
    Data = load_data("files/AV_features.csv", "files/Emotion_labels.csv",
                     "files/Speakers.csv")
    features, class_labels, speakers = Data.importdata()
    logo = LeaveOneGroupOut() #creates the leave one speaker out CV

    
    sp_id = 1
    

    for train, test in logo.split(features, class_labels, speakers):
        
        # block for SVM
        data_SVM = ML_SVM(features[train,:], class_labels[train])
        model_trained = data_SVM.SVM_RBF_gridsearch(3)
        Conf_Mat["SVM"][sp_id-1], UWR["SVM"][sp_id-1] = data_SVM.SVM_performance(model_trained,
           features[test,:], class_labels[test])
        
        #block for Random Forest
        data_RF = ML_RF(features[train,:], class_labels[train])
        model_trained = data_RF.RF_gridsearch(3)
        Conf_Mat["RF"][sp_id-1], UWR["RF"][sp_id-1] = data_RF.RF_performance(model_trained, 
           features[test,:], class_labels[test])        
        
        # block for Fully Connected Neural Network 
        data_ANN = ML_FC_ANN(features[train,:], class_labels[train], speakers[train])
        opt_layer_size = data_ANN.hyperparameter_tuning()
        trained_model = data_ANN.model_train(opt_layer_size)
        Conf_Mat["ANN"][sp_id-1], 
        UWR["ANN"][sp_id-1] = data_ANN.NN_performance(trained_model, 
           features[test,:], class_labels[test])

        sp_id+=1

        
    

if __name__ == main():
    main()
    

