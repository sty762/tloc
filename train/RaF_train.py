from numpy import *
import numpy as np
import pandas as pd
import math as Math
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from util_tloc import ext_feat_and_label

def non_transfer_train(tr_data, raf):
    tr_f, tr_label = ext_feat_and_label(tr_data)

    raf.fit(tr_f, tr_label)
    
    return raf