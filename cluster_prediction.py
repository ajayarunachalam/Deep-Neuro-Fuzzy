import numpy as np
import pandas as pd
from numpy import loadtxt
#%matplotlib inline
import matplotlib.pyplot as plt
import skfuzzy as fuzz


# store cluster centers for features
hcntr = np.array(loadtxt("humidity_cntr.txt", comments="#", delimiter=",", unpack=False))
pcntr = np.array(loadtxt("peekhour_cntr.txt", comments="#", delimiter=",", unpack=False))
rcntr = np.array(loadtxt("rainfall_cntr.txt", comments="#", delimiter=",", unpack=False))
tcntr = np.array(loadtxt("temp_cntr.txt", comments="#", delimiter=",", unpack=False))
wcntr = np.array(loadtxt("wind_cntr.txt", comments="#", delimiter=",", unpack=False))


def predict_humidity(values):
    # value -numpy array
    xpts = np.zeros(len(values))
    data = np.vstack((values, xpts))
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data, hcntr, 1.5, error=0.0005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    
    return cluster_membership



def predict_peekhour(values):
    xpts = np.zeros(len(values))
    data = np.vstack((values, xpts))
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data, pcntr, 1.5, error=0.0005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    
    return cluster_membership




def predict_rainfall(values):
    xpts = np.zeros(len(values))
    data = np.vstack((values, xpts))
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data, rcntr, 1.5, error=0.0005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    
    return cluster_membership



def predict_temp(values):
    xpts = np.zeros(len(values))
    data = np.vstack((values, xpts))
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data, tcntr, 1.5, error=0.0005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    
    return cluster_membership



def predict_wind(values):
    xpts = np.zeros(len(values))
    data = np.vstack((values, xpts))
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data, wcntr, 1.5, error=0.0005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)
    
    return cluster_membership


