# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:23:10 2024

@author: Alfred
"""

from scipy import stats
import numpy as np
import scikit_posthocs as sp

#error baseline 

score_2 = model_vermont.evaluate(hawaii_data1, hawaii_labels1)

score_3 = model_vermont.evaluate(nevada_data1, nevada_labels1)


score_02 = model_hawaii.evaluate(vermont_data1, vermont_labels1)

score_03 = model_hawaii.evaluate(nevada_data1, nevada_labels1)



score_12 = model_nevada.evaluate(vermont_data1, vermont_labels1)

score_13 = model_nevada.evaluate(hawaii_data1, hawaii_labels1)

#Kruskal-Wallis H-test
stats.kruskal(error, error1, error2)

stats.f_oneway(error, error1, error2[0:74])
