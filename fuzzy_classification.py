import numpy as np
import skfuzzy as fuzz
#%matplotlib inline  
import matplotlib.pyplot as plt
from numpy import loadtxt
import cluster_prediction as cp
import pandas as pd

# load weather data from text file
hdv = np.array(loadtxt("humidity-l.txt", comments="#", delimiter=",", unpack=False))
phv = np.array(loadtxt("peek_hour-l.txt", comments="#", delimiter=",", unpack=False))
rfv = np.array(loadtxt("rainfall-l.txt", comments="#", delimiter=",", unpack=False))
tpv = np.array(loadtxt("temp-l.txt", comments="#", delimiter=",", unpack=False))
wdv = np.array(loadtxt("wind-l.txt", comments="#", delimiter=",", unpack=False))
#alldata = np.vstack((hd, ph, rf, tp, wd)

# get membership value of weather data.
hd_mem = cp.predict_humidity(hdv)
ph_mem = cp.predict_peekhour(phv)
rf_mem = cp.predict_rainfall(rfv)
tp_mem = cp.predict_temp(tpv)
wd_mem = cp.predict_wind(wdv)
## contract training data
# road stutas, road construction, road accedent.
road_st = np.zeros(600)
#print (cluster_membership.shape)
ones = np.ones(82)
twos = np.array([2] * 50)
road_st = np.append(road_st, ones)
np.random.shuffle(road_st)
road_ct = np.zeros(550)
road_ct = np.append(road_ct, ones)
road_ct = np.append(road_ct, twos)
np.random.shuffle(road_ct)
road_ac = np.zeros(600)
road_ac = np.append(road_ac, ones)
np.random.shuffle(road_ac)
vehicle_v = np.random.uniform(0, 200, size = 682) # vehicle speed.


# Generate universe variables
# humidity cluster range (0, 4) (5 cluster)
# peekhour cluster range (0, 2) (3 cluster)
# rainfall cluster range (0, 2) (3 cluster)
# temperature cluster range(0, 2) (3 cluster)
# wind cluster range(0, 3) (4 cluster)
# road status range (0, 1) (0 or 1)
# road construction range (0, 2) (0 or 1 or 2)
# road accedent range (0, 1) (0 or 1)
hd = np.arange(0, 5, 1) # humidity
ph = np.arange(0, 3, 1) # peekhour
rf = np.arange(0, 3, 1) # rainfall
tp = np.arange(0, 3, 1) # temp
wd = np.arange(0, 4, 1) # wind
rs = np.arange(0, 2, 1) # road status
rc = np.arange(0, 3, 1) # road construction
ra = np.arange(0, 2, 1) # road accident
rw = np.arange(0, 11, 1) # road weaight
vhl = np.arange(0, 201, 1) # vehicle speed

# Generate fuzzy membership functions
# humidity Membership Function
hd_0 = fuzz.trimf(hd, [0, 0, 0])
hd_1 = fuzz.trimf(hd, [0, 1, 1])
hd_2 = fuzz.trimf(hd, [1, 2, 2])
hd_3 = fuzz.trimf(hd, [2, 3, 3])
hd_4 = fuzz.trimf(hd, [3, 4, 4])

# peek hour MF
ph_0 = fuzz.trimf(ph, [0, 0, 0])
ph_1 = fuzz.trimf(ph, [0, 1, 1])
ph_2 = fuzz.trimf(ph, [1, 2, 2])

# rainfall MF
rf_0 = fuzz.trimf(rf, [0, 0, 0])
rf_1 = fuzz.trimf(rf, [0, 1, 1])
rf_2 = fuzz.trimf(rf, [1, 2, 2])

# temperature MF
tp_0 = fuzz.trimf(tp, [0, 0, 0])
tp_1 = fuzz.trimf(tp, [0, 1, 1])
tp_2 = fuzz.trimf(tp, [1, 2, 2])

# wind membership function
wd_0 = fuzz.trimf(wd, [0, 0, 0])
wd_1 = fuzz.trimf(wd, [0, 1, 1])
wd_2 = fuzz.trimf(wd, [1, 2, 2])
wd_3 = fuzz.trimf(wd, [2, 3, 3])

# road status mf
rs_0 = fuzz.trimf(rs, [0, 0, 0])
rs_1 = fuzz.trimf(rs, [0, 1, 1])

# road construction MF
rc_0 = fuzz.trimf(rc, [0, 0, 0])
rc_1 = fuzz.trimf(rc, [0, 1, 1])
rc_2 = fuzz.trimf(rc, [1, 2, 2])

# road accident MF
ra_0 = fuzz.trimf(ra, [0, 0, 0])
ra_1 = fuzz.trimf(ra, [0, 1, 1])

# road weight MF
#rweight = fuzz.trimf(rw, [0, 3, 6])
rweight_0 = fuzz.trimf(rw, [0, 0, 0])
rweight_1 = fuzz.trimf(rw, [0, 1, 1])
rweight_2 = fuzz.trimf(rw, [1, 2, 2])
rweight_3 = fuzz.trimf(rw, [2, 3, 3])
rweight_4 = fuzz.trimf(rw, [3, 4, 4])
rweight_5 = fuzz.trimf(rw, [4, 5, 5])
rweight_6 = fuzz.trimf(rw, [5, 6, 6])
rweight_7 = fuzz.trimf(rw, [6, 7, 7])
rweight_8 = fuzz.trimf(rw, [7, 8, 8])
rweight_9 = fuzz.trimf(rw, [8, 9, 9])
rweight_10 = fuzz.trimf(rw, [9, 10, 10])
# vehicle MF
vhl_0 = fuzz.trimf(vhl, [0, 0, 10])
vhl_1 = fuzz.trimf(vhl, [10, 10, 30])
vhl_2 = fuzz.trimf(vhl, [30, 30, 200])


# fuzz.inter_membership
# activation of membership function
road_weight = np.array([])

traindata = (np.array([['hdl0', 'hdl1', 'hdl2', 'hdl3', 'hdl4','phl0', 'phl1', 'phl2',
                       'rfl0', 'rfl1', 'rfl2','tpl0', 'tpl1', 'tpl2',
                       'wdl0', 'wdl1', 'wdl2', 'wdl3', 'rsl0', 'rsl1',
                       'rcl0', 'rcl1', 'rcl2','ral0', 'ral1', 'vhll0','vhll1', 'vhll2']])) ## training data heads

def get_data():
    global traindata
    global road_weight
    for i in range (0, 682):
        interp = np.array([])
        # for humidity interp mf
        hd_level_0 = fuzz.interp_membership(hd, hd_0, hd_mem[i])
        hd_level_1 = fuzz.interp_membership(hd, hd_1, hd_mem[i])
        hd_level_2 = fuzz.interp_membership(hd, hd_2, hd_mem[i])
        hd_level_3 = fuzz.interp_membership(hd, hd_3, hd_mem[i])
        hd_level_4 = fuzz.interp_membership(hd, hd_4, hd_mem[i])
        
        interp = (np.append(interp, [hd_level_0, hd_level_1, hd_level_2, 
                                          hd_level_3, hd_level_4]))
        # for peek hour interp mf
        ph_level_0 = fuzz.interp_membership(ph, ph_0, ph_mem[i])
        ph_level_1 = fuzz.interp_membership(ph, ph_1, ph_mem[i])
        ph_level_2 = fuzz.interp_membership(ph, ph_2, ph_mem[i])

        interp = np.append(interp, [ph_level_0, ph_level_1, ph_level_2])
        # rainfall interp mf
        rf_level_0 = fuzz.interp_membership(rf, rf_0, rf_mem[i])
        rf_level_1 = fuzz.interp_membership(rf, rf_1, rf_mem[i])
        rf_level_2 = fuzz.interp_membership(rf, rf_2, rf_mem[i])

        interp = np.append(interp, [rf_level_0, rf_level_1, rf_level_2])
        # temperature interp mf
        tp_level_0 = fuzz.interp_membership(tp, tp_0, tp_mem[i])
        tp_level_1 = fuzz.interp_membership(tp, tp_1, tp_mem[i])
        tp_level_2 = fuzz.interp_membership(tp, tp_2, tp_mem[i])
        
        interp = np.append(interp, [tp_level_0, tp_level_1, tp_level_2])
        # wind interp mf
        wd_level_0 = fuzz.interp_membership(wd, wd_0, wd_mem[i])
        wd_level_1 = fuzz.interp_membership(wd, wd_1, wd_mem[i])
        wd_level_2 = fuzz.interp_membership(wd, wd_2, wd_mem[i])
        wd_level_3 = fuzz.interp_membership(wd, wd_3, wd_mem[i])

        interp = np.append(interp, [wd_level_0, wd_level_1, wd_level_2, wd_level_3])
        # for road status
        rs_level_0 = fuzz.interp_membership(rs, rs_0, road_st[i])
        rs_level_1 = fuzz.interp_membership(rs, rs_1, road_st[i])
        
        interp = np.append(interp, [rs_level_0, rs_level_1])

        # for road construction
        rc_level_0 = fuzz.interp_membership(rc, rc_0, road_ct[i])
        rc_level_1 = fuzz.interp_membership(rc, rc_1, road_ct[i])
        rc_level_2 = fuzz.interp_membership(rc, rc_2, road_ct[i])

        interp = np.append(interp, [rc_level_0, rc_level_1, rc_level_2])
        # for road accedent
        ra_level_0 = fuzz.interp_membership(ra, ra_0, road_ac[i])
        ra_level_1 = fuzz.interp_membership(ra, ra_1, road_ac[i])

        interp = np.append(interp, [ra_level_0, ra_level_1])
        
        # for vechile speed
        vhl_level_0 = fuzz.interp_membership(vhl, vhl_0, vehicle_v[i])
        vhl_level_1 = fuzz.interp_membership(vhl, vhl_1, vehicle_v[i])
        vhl_level_2 = fuzz.interp_membership(vhl, vhl_2, vehicle_v[i])

        interp = np.append(interp, [vhl_level_0, vhl_level_1, vhl_level_2])

        traindata = np.vstack((traindata, interp))
        # activation rule
        weather_active_rule0 = np.fmax(np.fmax(np.fmax(hd_level_2, wd_level_2),
                                   np.fmax(tp_level_2, rf_level_0)),
                                  np.fmax(vhl_level_2, ph_level_1))
        #print (weather_active_rule0)
        weather_active_rule1 = np.fmax(np.fmax(hd_level_4, wd_level_1), 
                                        np.fmax(tp_level_0, rf_level_1))
        #print (weather_active_rule1)

        weather_active_rule2 = np.fmax(np.fmax(hd_level_0, wd_level_3),
                                       np.fmax(tp_level_0, rf_level_0))

        weather_active_rule3 = np.fmax(np.fmax(hd_level_3, hd_level_1),
                                        tp_level_1)
        #print (weather_active_rule3)
        rstatus_active_rule4 = np.fmax(np.fmax(ph_level_0, vhl_level_1), rf_level_1)

        rstatus_active_rule5 = np.fmax( np.fmax(ph_level_2, vhl_level_1), rf_level_2)
            
        rstatus_active_rule6 = np.fmax(ph_level_0, rf_level_2)

        rstatus_active_rule7 = np.fmax(vhl_level_1, ph_level_2)

        rstatus_active_rule8 = np.fmax(vhl_level_0, ph_level_0)

        rstatus_active_rule9 = np.fmax( np.fmax(rs_level_1, ph_level_0 ), rc_level_1)

        rstatus_active_rule10 = np.fmax( np.fmax(rs_level_1, ph_level_0), 
                                        np.fmax(rc_level_2, ra_level_1))


        weight_active_level0 = np.fmax(weather_active_rule0, rweight_0)
        weight_active_level1 = np.fmax(weather_active_rule1, rweight_1)
        weight_active_level2 = np.fmax(weather_active_rule1, rweight_2)
        weight_active_level3 = np.fmax(weather_active_rule3, rweight_3)
        weight_active_level4 = np.fmax(rstatus_active_rule4, rweight_4)
        weight_active_level5 = np.fmax(rstatus_active_rule5, rweight_5)
        weight_active_level6 = np.fmax(rstatus_active_rule6, rweight_6)
        weight_active_level7 = np.fmax(rstatus_active_rule7, rweight_7)
        weight_active_level8 = np.fmax(rstatus_active_rule8, rweight_8)
        weight_active_level9 = np.fmax(rstatus_active_rule9, rweight_9)
        weight_active_level10 = np.fmax(rstatus_active_rule10, rweight_10)

        # Aggregate all  output membership functions together

        aggregated = np.zeros(11)
        sums = np.zeros(11)
        sums[0] = np.sum(weight_active_level0)
        sums[1] = np.sum(weight_active_level1)
        sums[2] = np.sum(weight_active_level2)
        sums[3] = np.sum(weight_active_level3)
        sums[4] = np.sum(weight_active_level4)
        sums[5] = np.sum(weight_active_level5)
        sums[6] = np.sum(weight_active_level6)
        sums[7] = np.sum(weight_active_level7)
        sums[8] = np.sum(weight_active_level8)
        sums[9] = np.sum(weight_active_level9)
        sums[10] = np.sum(weight_active_level10)

        count = 0
        index = 0
        for i in range(0, len(sums)):
            if count <= sums[i]:
                count = sums[i]
                index = i
                
        aggregated[index] = 1
        #print ('aggregated : ')
        #print (aggregated)

        # Calculate defuzzified result
        est_road_weight = fuzz.defuzz(rw, aggregated, 'mom')
        road_weight = np.append(road_weight, est_road_weight)
        #print ('est_Road Weight :')
        #print (est_road_weight)
        road_weight_activation = fuzz.interp_membership(rw, aggregated, est_road_weight)  # for plot
        #print ('road weight activation :')
        #print (road_weight_activation)
        # Visualize this
        # cleandata_df = pd.DataFrame(clean_data, index=days_id)
    index = range(0, 682)
    head = traindata[0]
    #print ('head :')
    #print (head)
    traindata = np.delete(traindata, (0), axis=0)
    #print (traindata.shape)
    traindata_df = pd.DataFrame(data=traindata, index=index, columns=head)
    #print ('traindata_df :')
    #print (traindata_df)
    target_df = pd.DataFrame(data=road_weight, index=index, columns=['Road Weight'])
    #print ('Road Weights : ')
    #print (road_weight)
    #print ('target_df :')
    #print (target_df)
    return traindata_df, target_df

traindata_df, target_df = get_data()


####################################################################################
def get_newdata(newhd, newph, newrf, newtp, newwd):

	newhd_mem = cp.predict_humidity([newhd])
	newph_mem = cp.predict_peekhour([newph])
	newrf_mem = cp.predict_rainfall([newrf])
	newtp_mem = cp.predict_temp([newtp])
	newwd_mem = cp.predict_wind([newwd])

	return newhd_mem, newph_mem, newrf_mem, newtp_mem, newwd_mem


def predict_fuzz_mem(newhd, newph, newrf, newtp, newwd, newroad_st, newroad_ct, newroad_ac, newvehicle_v):
	newhd_mem, newph_mem, newrf_mem, newtp_mem, newwd_mem = get_newdata(newhd, newph, newrf, newtp, newwd)
	interp = np.array([])

	# for humidity interp mf
	hd_level_0 = fuzz.interp_membership(hd, hd_0, newhd_mem)
	hd_level_1 = fuzz.interp_membership(hd, hd_1, newhd_mem)
	hd_level_2 = fuzz.interp_membership(hd, hd_2, newhd_mem)
	hd_level_3 = fuzz.interp_membership(hd, hd_3, newhd_mem)
	hd_level_4 = fuzz.interp_membership(hd, hd_4, newhd_mem)
	interp = (np.append(interp, [hd_level_0, hd_level_1, hd_level_2, 
                                          hd_level_3, hd_level_4]))
    # for peek hour interp mf
	ph_level_0 = fuzz.interp_membership(ph, ph_0, newph_mem)
	ph_level_1 = fuzz.interp_membership(ph, ph_1, newph_mem)
	ph_level_2 = fuzz.interp_membership(ph, ph_2, newph_mem)
	interp = np.append(interp, [ph_level_0, ph_level_1, ph_level_2])

	# rainfall interp mf
	rf_level_0 = fuzz.interp_membership(rf, rf_0, newrf_mem)
	rf_level_1 = fuzz.interp_membership(rf, rf_1, newrf_mem)
	rf_level_2 = fuzz.interp_membership(rf, rf_2, newrf_mem)
	interp = np.append(interp, [rf_level_0, rf_level_1, rf_level_2])

	# temperature interp mf
	tp_level_0 = fuzz.interp_membership(tp, tp_0, newtp_mem)
	tp_level_1 = fuzz.interp_membership(tp, tp_1, newtp_mem)
	tp_level_2 = fuzz.interp_membership(tp, tp_2, newtp_mem)
	interp = np.append(interp, [tp_level_0, tp_level_1, tp_level_2])

	# wind interp mf
	wd_level_0 = fuzz.interp_membership(wd, wd_0, newwd_mem)
	wd_level_1 = fuzz.interp_membership(wd, wd_1, newwd_mem)
	wd_level_2 = fuzz.interp_membership(wd, wd_2, newwd_mem)
	wd_level_3 = fuzz.interp_membership(wd, wd_3, newwd_mem)
	interp = np.append(interp, [wd_level_0, wd_level_1, wd_level_2, wd_level_3])
	# for road status
	rs_level_0 = fuzz.interp_membership(rs, rs_0, newroad_st)
	rs_level_1 = fuzz.interp_membership(rs, rs_1, newroad_st)
	interp = np.append(interp, [rs_level_0, rs_level_1])

	# for road construction
	rc_level_0 = fuzz.interp_membership(rc, rc_0, newroad_ct)
	rc_level_1 = fuzz.interp_membership(rc, rc_1, newroad_ct)
	rc_level_2 = fuzz.interp_membership(rc, rc_2, newroad_ct)
	interp = np.append(interp, [rc_level_0, rc_level_1, rc_level_2])

	# for road accedent
	ra_level_0 = fuzz.interp_membership(ra, ra_0, newroad_ac)
	ra_level_1 = fuzz.interp_membership(ra, ra_1, newroad_ac)
	interp = np.append(interp, [ra_level_0, ra_level_1])

	# for vechile speed
	vhl_level_0 = fuzz.interp_membership(vhl, vhl_0, newvehicle_v)
	vhl_level_1 = fuzz.interp_membership(vhl, vhl_1, newvehicle_v)
	vhl_level_2 = fuzz.interp_membership(vhl, vhl_2, newvehicle_v)
	interp = np.append(interp, [vhl_level_0, vhl_level_1, vhl_level_2])
    
	return interp
