import numpy as np
from scipy.stats import mode
from numpy import loadtxt


"""
hd = np.array(loadtxt("humidity-l.txt", comments="#", delimiter=",", unpack=False))
ph = np.array(loadtxt("peek_hour-l.txt", comments="#", delimiter=",", unpack=False))
rf = np.array(loadtxt("rainfall-l.txt", comments="#", delimiter=",", unpack=False))
tp = np.array(loadtxt("temp-l.txt", comments="#", delimiter=",", unpack=False))
wd = np.array(loadtxt("wind-l.txt", comments="#", delimiter=",", unpack=False))
alldata = np.vstack((hd, ph, rf, tp, wd))
"""

def mahalanobis_dist(data):
    x = np.copy(data)
    cov_x = np.cov(x)
    invcov_x = np.linalg.inv(cov_x)
    x_mean = np.array([])
    
    for i in range(len(x)):
        x_mean = np.append(x_mean, np.mean(x[i]))
    
    x_diff = np.array([])
    diff = np.array([v - x_mean[0] for v in x[0]])
    x_diff = np.hstack((x_diff, diff))
    
    for i in range (1, len(x)):
        diff = np.array([v - x_mean[i] for v in x[i]])
        x_diff = np.vstack((x_diff, diff))
    
    diff_x = np.transpose(x_diff)
    md = []
    for i in range(len(diff_x)):
            md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_x[i]),invcov_x),diff_x[i])))

    return md



def removeOutliers(data):
    md = mahalanobis_dist(data)
    threshold = np.mean(md) * 1.5# adjust 1.5 accordingly 
    save_x, outliers = np.array([]), np.array([])
    save_item, outlier_item = np.array([]), np.array([])
    for i in range(len(md)):
        if md[i] <= threshold:
            if len(save_x) == 0:
                save_x = np.hstack((save_x, data[:, i]))
            else:
                save_x = np.vstack((save_x, data[:, i]))
            save_item = np.append(save_item, i+1)
        else:
            if len(outliers) == 0:
                outliers = np.hstack((outliers, data[:, i]))
            else:
                outliers = np.vstack((outliers, data[:, i]))
            outlier_item = np.append(outlier_item, i+1)
             
    return save_x, outliers, save_item, outlier_item




#md = mahalanobis_dist(alldata)
#save_x, outliers, s, o= removeOutliers(alldata)

"""
print ('save_x : ')
print (save_x)
print ('save_x->shape : ', save_x.shape)
print ('outliers : ')
print (outliers)
print ('outliers->shape : ', outliers.shape)

print ('s :')
print (s)
print ('o :')
print (o)
"""
"""
print ('humidity')
print ('min : ', hd.min(), 'max : ', hd.max())
print ('peek hour')
print ('min : ', ph.min(), 'max : ', ph.max())
print ('rain fall')
print ('min : ', rf.min(), 'max : ', rf.max())
print ('temp')
print ('min : ', tp.min(), 'max : ', tp.max())
print ('wind')
print ('min : ', wd.min(), 'max : ', wd.max())

print ('save_x min & max')
for i in range (5):
    print ('min : ', save_x[:, i].min(), 'max : ', save_x[:, i].max())
    
print ('\n outliers min & max')
for i in range (5):
    print ('min : ', outliers[:, i].min(), 'max : ', outliers[:, i].max())
"""
