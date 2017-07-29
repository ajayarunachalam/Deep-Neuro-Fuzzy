import numpy as np
import pandas as pd
from numpy import loadtxt
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import skfuzzy as fuzz
import mahalanobisDist as md
global str
import csv as csv

hd = np.array(loadtxt("humidity-l.txt", comments="#", delimiter=",", unpack=False))
ph = np.array(loadtxt("peek_hour-l.txt", comments="#", delimiter=",", unpack=False))
rf = np.array(loadtxt("rainfall-l.txt", comments="#", delimiter=",", unpack=False))
tp = np.array(loadtxt("temp-l.txt", comments="#", delimiter=",", unpack=False))
wd = np.array(loadtxt("wind-l.txt", comments="#", delimiter=",", unpack=False))
alldata = np.vstack((hd, ph, rf, tp, wd))


def get_cleandata(data):
    clean_data, outliers, cl_idx, ol_idx = md.removeOutliers(alldata)
    days_id = np.array([])
    _ = 'Day#'
    for i in range (len(cl_idx)):
        days_id = np.append(days_id, _ + str(int(cl_idx[i])))

    return clean_data, days_id

#clean_data , days_id = get_cleandata(alldata)

def create_dataframe(data):
    clean_data, days_id = get_cleandata(data)
    cleandata_df = pd.DataFrame(clean_data, index=days_id)
    #cleandata_df.columns = ['daysID','humidity', 'peek-hour', 'rainfall', 'temp', 'wind']
    cleandata_df = cleandata_df.rename(columns = {0:'humidity'})
    cleandata_df = cleandata_df.rename(columns = {1:'peek_hour'})
    cleandata_df = cleandata_df.rename(columns = {2:'rainfall'})
    cleandata_df = cleandata_df.rename(columns = {3:'temp'})
    cleandata_df = cleandata_df.rename(columns = {4:'wind'})
    cleandata_df.index.names = ['Day']
    cleandata_df.columns.names = ['(weather feature)']
    #print (cleandata_df.head())

    return cleandata_df


def pca_analysis(data):
    cleandata_df = create_dataframe(data)
    pca = PCA(copy=True, n_components=2, whiten=False)
    pca.fit(cleandata_df)

    cleandata_2d = pca.transform(cleandata_df)

    cleandata_df_2d = pd.DataFrame(cleandata_2d)
    cleandata_df_2d.index = cleandata_df.index
    cleandata_df_2d.columns = ['PCA1','PCA2']

    #print ("\ncleandata_df_2d.head() :\n", cleandata_df_2d.head())
    #print('\npca.explained_variance_ratio_ :\n') 
    #print(pca.explained_variance_ratio_) 

    return cleandata_df_2d, cleandata_df


def visualize_pca_dimentionalReduction(data):
    cleandata_df_2d, cleandata_df = pca_analysis(data)

    cleandata_df_2d['Day_mean'] = pd.Series(cleandata_df.mean(axis=1), index=cleandata_df_2d.index)
    Day_mean_max = cleandata_df_2d['Day_mean'].max()
    Day_mean_min = cleandata_df_2d['Day_mean'].min()
    Day_mean_scaled = (cleandata_df_2d.Day_mean-Day_mean_min) / Day_mean_max
    cleandata_df_2d['Day_mean_scaled'] = pd.Series(
                                       Day_mean_scaled, 
                                       index=cleandata_df_2d.index) 
    print (cleandata_df_2d.head())

    cleandata_df_2d.plot(
        kind='scatter', 
        title = 'Principal Component Analysis\n Dimentionality Reduction\n (Day-Mean-Scaled))',
        x='PCA2', 
        y='PCA1', 
        s=cleandata_df_2d['Day_mean_scaled']*100, 
        figsize=(16,8))


#visualize_pca_dimentionalReduction(alldata)

def c_mean_weather_clustering(data):
    cleandata_df_2d, cleandata_df = pca_analysis(data)

    cleandata_df.astype(np.float32)
    
    ncntr = 7
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        cleandata_df.T, ncntr, 1.1, error=0.0005, maxiter=1500, init=None)

    weather_membership = np.argmax(u, axis=0)

    #print ('cluster membership->shape : ', cluster_membership.shape)
    #print (cluster_membership.min())
    #print (cluster_membership.max())

    return weather_membership, ncntr, cntr

#c_mean_clustering(alldata

cleandata_df = create_dataframe(alldata) # get clean dataframe


def c_mean_humidity_clustering():
    hd =cleandata_df['humidity'].astype(np.float32)
    hdcntr = 5
    hd = hd.values
    xpts = np.zeros(hd.shape)
    data = np.vstack((hd, xpts))
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, hdcntr, 1.5, error=0.0005, maxiter=1000, init=None)

    humidity_membership = np.argmax(u, axis=0)

    return humidity_membership, hdcntr, cntr


def c_mean_peekhour_clustering():
    ph =cleandata_df['peek_hour'].astype(np.float32)
    ph = ph.values
    xpts = np.zeros(ph.shape)
    data = np.vstack((ph, xpts))
    phcntr = 3
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, phcntr, 1.5, error=0.0005, maxiter=1000, init=None)

    peekhour_membership = np.argmax(u, axis=0)

    return peekhour_membership, phcntr, cntr


def c_mean_rainfall_clustering():
    rf =cleandata_df['rainfall'].astype(np.float32)
    rf = rf.values
    xpts = np.zeros(rf.shape)
    data = np.vstack((rf, xpts))
    rfcntr = 3
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, rfcntr, 1.5, error=0.0005, maxiter=1000, init=None)

    rainfall_membership = np.argmax(u, axis=0)

    return rainfall_membership, rfcntr, cntr



def c_mean_temp_clustering():
    tp =cleandata_df['temp'].astype(np.float32)
    tp = tp.values
    xpts = np.zeros(tp.shape)
    data = np.vstack((tp, xpts))
    tpcntr = 3
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, tpcntr, 1.5, error=0.0005, maxiter=1000, init=None)

    temp_membership = np.argmax(u, axis=0)

    return temp_membership, tpcntr, cntr



def c_mean_wind_clustering():
    wd =cleandata_df['wind'].astype(np.float32)
    wd = wd.values
    xpts = np.zeros(wd.shape)
    data = np.vstack((wd, xpts))
    wdcntr = 4
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, wdcntr, 1.5, error=0.0005, maxiter=1000, init=None)

    wind_membership = np.argmax(u, axis=0)

    return wind_membership, wdcntr, cntr


def visualize_cluster(data):
    cluster_membership, ncntr = c_mean_clustering(data)
    cleandata_df_2d, cleandata_df = pca_analysis(data)
    
    cleandata_df_2d['cluster'] = pd.Series(cluster_membership, 
                                              index=cleandata_df_2d.index)
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']


    cleandata_df_2d.plot(
        kind='scatter',
        title = 'Fuzzy C-means Clustering (5)',
        x='PCA2',y='PCA1',
        c = cleandata_df_2d.cluster.astype(np.float),
        figsize=(16,8)
        )
    #print ('visualize cluster')

"""
hd_mem, hc = c_mean_humidity_clustering()
ph_mem, pc = c_mean_peekhour_clustering()
rf_mem, rc = c_mean_rainfall_clustering()
tm_mem, tc = c_mean_temp_clustering()
wd_mem, wc = c_mean_wind_clustering()

print ('humidity membership->shape : ', hd_mem.shape)
print ('humidity membership : ')
print (hd_mem)
print ('peek hour membership-.shape : ', ph_mem.shape)
print ('peek hour mem : ')
print (ph_mem)
print ('rf meme->shape : ', rf_mem.shape)
print ('rf mem : ')
print (rf_mem)
print ('tm mem->shape : ', tm_mem.shape)
print ("tm mem :")
print (tm_mem)
print ('wd mem->shape : ', wd_mem.shape)
print ('wd mem : ')
print (wd_mem)
print ('centers : ' )
print (hc, pc, rc, tc, wc)
"""