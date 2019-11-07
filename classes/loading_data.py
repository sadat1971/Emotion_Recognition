#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:52:58 2019

@author: sadat
"""

import numpy as np

class load_data:
    
    """
       This class simply takes takes in the data path which is in CSV format and produces as 
       features and labels.
    """
    
    def __init__(self, feature_path, label_path, group_path):
        
        self.feature_path = feature_path
        self.label_path = label_path
        self.group_path = group_path
        
        
    def importdata(self):
        # this function creates, loads and produces the features, labels, and
        # speaker list
        features = np.loadtxt(self.feature_path, delimiter = ",")
        labels = np.loadtxt(self.label_path, delimiter = ",")
        speakers = np.loadtxt(self.group_path, delimiter = ",")
        
        return features, labels, speakers