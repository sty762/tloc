import numpy as np
import math

from util_tloc import distance,compute_relative_feature

def raf_test(te_data,raf):
    serving_bs = te_data[['Lon','Lat']].iloc[0,:]
    te_label = te_data[['re_lon', 're_lat']].values
    te_f = compute_relative_feature(te_data).values

    pred = raf.predict(te_f)
    pred[:, 0] += serving_bs[0]
    pred[:, 1] += serving_bs[1]
    
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pred, te_data[['Longitude','Latitude']].values)]
    error = sorted(error)
    print("RaF Test Loss: {}".format(np.median(error)))
    
    return np.median(error), error