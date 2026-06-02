import scipy.io
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np

"""
TODO:
- check values van wittenberghe
- check if gitelson 0.4 corrected is expected
- impl plots
"""

"""
FORMULAS

https://www.sciencedirect.com/science/article/pii/S0034425724003122

SIF -> sun induced chlorophyll fluorescence
F -> Fluorescence emission
APAR -> absorbed photosynthetically active radiation
ΦF -> quantum yield of fluorescence (photosystem level)
f_esc -> escape probability of photons (not absorbed/scattered)
λ -> wavelength
R -> Reflectance
"""



# read data
mat = scipy.io.loadmat('data/Fluowat_2055_n1_result.mat', struct_as_record=False, squeeze_me=True)

n1 = mat['result'].n1
print(list(vars(n1).keys()))
print(list(vars(n1.raw_data).keys()))


print("fluo ", n1.FLUO_spc_photons.shape)
print("wvl_400_800", n1.wvl_400_800.shape)
print("wvl_500_780", n1.wvl_500_780.shape)
print("wvl_650_850", n1.wvl_650_850.shape)
print(n1.wvl_650_850[0], n1.wvl_650_850[-1])

# find closest indices (MATLAB min(abs(...))), use same wavelengths for all variables
wvl = n1.raw_data.wvl[0].T

def get_wvl_ref(arr):
    match len(arr):
        case 2151: return wvl
        case 401: return wvl_400_800
        case 201: return wvl_650_850
        case 281: return wvl_500_780
        case _: print("No valid wavelength refernce!")

def get_wvl_subset(arr, sub):
    ref = get_wvl_ref(arr)
    pos1A = np.argmin(np.abs(ref - sub[0]))
    pos2A = np.argmin(np.abs(ref - sub[-1]))
    arr = np.asarray(arr).squeeze()
    return arr[pos1A:pos2A+1]

# convert variables in watts from raw_data to photons
def to_photons(var):
    wvl = get_wvl_ref(var)
    # ######## FLUORESCENCE ################
    # Conversion of FLUO to micromol photons m-2 s-1
    # Conversion to photon flux units
    h = 6.62607015e-34#Planck constant Joules s
    c = 299792458#speed of light m s-1
    Na = 6.02e23#photons mol-1  Na*1e6 gives photons micromol-1
    # Fup | Input DATA in W/m2/sr/nm 
    wl = np.asarray(wvl).squeeze()[:, None]   # makes (201,1)
    var = np.asarray(var)
    return (1e6/(Na)) * (1e-9/(h*c)) * wl * (var/1000*3.14)

# get variables
wvl_400_800 = n1.wvl_400_800
wvl_500_780 = n1.wvl_500_780
wvl_650_850 = n1.wvl_650_850
irr = n1.irr_PAR_400_800
chla = n1.endmembers_Chla_BB_lsqlin
chlb = n1.endmembers_Chlb_BB_lsqlin
carb = n1.endmembers_Carb_BB_lsqlin
anc = n1.endmembers_Anc_BB_lsqlin
apar = n1.APAR_spc_photons_lsqlin
par = n1.PAR_spc_photons
fluo = n1.FLUO_spc_photons
# transpose variables from raw_data and convert from watts to photons
f_up = to_photons(n1.raw_data.Fup.T)
f_dw = to_photons(n1.raw_data.Fdw.T)
trans = n1.raw_data.trans_real.T
refl = n1.raw_data.refl_real.T
abs = n1.raw_data.abs_real.T


# check data
print("--- Dimensions of data ---")
print("wvl: ", len(wvl))
print("irr: ", len(irr), len(irr[1]))
print("fluo: ", len(fluo), len(fluo[1]))
print("f_up: ", len(f_up), len(f_up[1]))
print("f_dw: ", len(f_dw), len(f_dw[1]))
print("apar: ", len(apar), len(apar[1]))
print("par: ", len(par), len(par[1]))
print("chla: ", len(chla), len(chla[1]))
print("chlb: ", len(chlb), len(chlb[1]))
print("carb: ", len(carb), len(carb[1]))
print("anc: ", len(anc), len(anc[1]))
print("trans: ", len(trans), len(trans[1]))
print("refl: ", len(refl), len(refl[1]))




# Van Wittenberghe Methodology
def van_wittenberghe(f_up, f_dw, refl, trans, wvl):
    f_up = get_wvl_subset(f_up, wvl)
    f_dw = get_wvl_subset(f_dw, wvl)
    refl = get_wvl_subset(refl, wvl)
    trans = get_wvl_subset(trans, wvl)

    F_ps = np.add(np.divide(f_up, refl), np.divide(f_dw, trans))
    f_total = np.add(f_up, f_dw)
    f_esc = np.sqrt(np.divide(f_total, F_ps))
    fluo_corrected = np.divide(f_total, f_esc)
    return (f_total, fluo_corrected, f_esc)


# Gitelson Methodology
def gitelson(fluo, trans, refl, wvl):
    fluo = get_wvl_subset(fluo, wvl)
    trans = get_wvl_subset(trans, wvl)
    refl = get_wvl_subset(refl, wvl)

    fluo_corrected = np.divide(fluo, np.add(trans, refl))
    f_esc = np.divide(fluo, fluo_corrected)
    return (fluo, fluo_corrected, f_esc)


# P Methodology
def p(fluo, apar, a, wvl):
    fluo = get_wvl_subset(fluo, wvl)
    apar = get_wvl_subset(apar, wvl)
    a = get_wvl_subset(a, wvl)

    norm = np.divide(apar, a)
    fluo_corrected = np.divide(fluo, np.subtract(1, norm))
    f_esc = np.divide(fluo, fluo_corrected)
    return (fluo, fluo_corrected, f_esc)


#gitelson(fluo, trans, refl, wvl_650_850)


#van_wittenberghe(f_up, f_dw, refl, trans, wvl_650_850)

abs_p = get_wvl_subset(par, wvl_500_780) * (chla + chlb + carb + anc)
p(fluo, apar, abs_p, np.intersect1d(wvl_500_780, wvl_650_850))
