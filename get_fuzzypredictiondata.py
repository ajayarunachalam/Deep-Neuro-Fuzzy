import numpy as np
import skfuzzy as fuzz
import get_predictiondata as gp


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
rw = np.arange(0, 7, 1) # road weaight
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

# vehicle MF
vhl_0 = fuzz.trimf(vhl, [0, 0, 10])
vhl_1 = fuzz.trimf(vhl, [10, 10, 30])
vhl_2 = fuzz.trimf(vhl, [30, 30, 200])

# collect weather feature MF and road features value.
hd_mem, ph_mem, rf_mem, tp_mem, wd_mem, rs_v, rc_v, ra_v, vhl_v = gp.get_newdata()

def get_fuzzy_interp():
	# fuzz.inter_membership
	# activation of membership function
	interp = np.array([])
	# for humidity interp mf
	hd_level_0 = fuzz.interp_membership(hd, hd_0, hd_mem)
	hd_level_1 = fuzz.interp_membership(hd, hd_1, hd_mem)
	hd_level_2 = fuzz.interp_membership(hd, hd_2, hd_mem)
	hd_level_3 = fuzz.interp_membership(hd, hd_3, hd_mem)
	hd_level_4 = fuzz.interp_membership(hd, hd_4, hd_mem)
    
	interp = (np.append(interp, [hd_level_0, hd_level_1, hd_level_2, 
                                    hd_level_3, hd_level_4]))
	# for peek hour interp mf
	ph_level_0 = fuzz.interp_membership(ph, ph_0, ph_mem)
	ph_level_1 = fuzz.interp_membership(ph, ph_1, ph_mem)
	ph_level_2 = fuzz.interp_membership(ph, ph_2, ph_mem)

	interp = np.append(interp, [ph_level_0, ph_level_1, ph_level_2])
	# rainfall interp mf
	rf_level_0 = fuzz.interp_membership(rf, rf_0, rf_mem)
	rf_level_1 = fuzz.interp_membership(rf, rf_1, rf_mem)
	rf_level_2 = fuzz.interp_membership(rf, rf_2, rf_mem)

	interp = np.append(interp, [rf_level_0, rf_level_1, rf_level_2])
	# temperature interp mf
	tp_level_0 = fuzz.interp_membership(tp, tp_0, tp_mem)
	tp_level_1 = fuzz.interp_membership(tp, tp_1, tp_mem)
	tp_level_2 = fuzz.interp_membership(tp, tp_2, tp_mem)
	    
	interp = np.append(interp, [tp_level_0, tp_level_1, tp_level_2])
	# wind interp mf
	wd_level_0 = fuzz.interp_membership(wd, wd_0, wd_mem)
	wd_level_1 = fuzz.interp_membership(wd, wd_1, wd_mem)
	wd_level_2 = fuzz.interp_membership(wd, wd_2, wd_mem)
	wd_level_3 = fuzz.interp_membership(wd, wd_3, wd_mem)

	interp = np.append(interp, [wd_level_0, wd_level_1, wd_level_2, wd_level_3])
	# for road status
	rs_level_0 = fuzz.interp_membership(rs, rs_0, rs_v)
	rs_level_1 = fuzz.interp_membership(rs, rs_1, rs_v)
    
	interp = np.append(interp, [rs_level_0, rs_level_1])

	# for road construction
	rc_level_0 = fuzz.interp_membership(rc, rc_0, rc_v)
	rc_level_1 = fuzz.interp_membership(rc, rc_1, rc_v)
	rc_level_2 = fuzz.interp_membership(rc, rc_2, rc_v)

	interp = np.append(interp, [rc_level_0, rc_level_1, rc_level_2])
	# for road accedent
	ra_level_0 = fuzz.interp_membership(ra, ra_0, ra_v)
	ra_level_1 = fuzz.interp_membership(ra, ra_1, ra_v)

	interp = np.append(interp, [ra_level_0, ra_level_1])
    
	# for vechile speed
	vhl_level_0 = fuzz.interp_membership(vhl, vhl_0, vhl_v)
	vhl_level_1 = fuzz.interp_membership(vhl, vhl_1, vhl_v)
	vhl_level_2 = fuzz.interp_membership(vhl, vhl_2, vhl_v)

	interp = np.append(interp, [vhl_level_0, vhl_level_1, vhl_level_2])

	
	return interp

