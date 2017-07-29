import numpy as np
import pandas as pd
from numpy import loadtxt
import cluster_prediction as cp

def weather_feture_mf(data): # get membership value for weather feature
    hd = cp.predict_humidity(data[0])
    ph = cp.predict_peekhour(data[1])
    rf = cp.predict_rainfall(data[2])
    tp = cp.predict_temp(data[3])
    wd = cp.predict_wind(data[4])
    all_feature = np.vstack((hd, ph, rf, tp, wd))
    
    return all_feature


def get_road_data(data): # collecte road data info
    return data



def get_newdata(): # combine all data
    wd_mf = weather_feture_mf([64, 79, 0, 23, 9])
    wd_mf = wd_mf.T
    wd_mf = wd_mf[0]
    rd_mf = get_road_data([0, 0, 0, 90]) # rs, rc, ra, vhl
    newdata = np.hstack((wd_mf, rd_mf))
    return newdata

#print (get_newdata())