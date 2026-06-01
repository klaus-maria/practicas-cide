import scipy.io
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np


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

obj = mat['result'].n1
print(list(vars(obj).keys()))
print(list(vars(obj.raw_data).keys()))


print("fluo ", obj.FLUO_spc_photons.shape)
print("wvl_400_800", obj.wvl_400_800.shape)
print("wvl_500_780", obj.wvl_500_780.shape)
print("wvl_650_850", obj.wvl_650_850.shape)
print(obj.wvl_650_850[0], obj.wvl_650_850[-1])

# find closest indices (MATLAB min(abs(...))), use same wavelengths for all variables
wvl = obj.raw_data.wvl[0].T

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

# get variables
wvl_400_800 = obj.wvl_400_800
wvl_500_780 = obj.wvl_500_780
wvl_650_850 = obj.wvl_650_850
irr = obj.irr_PAR_400_800
chla = obj.endmembers_Chla_BB_lsqlin
chlb = obj.endmembers_Chlb_BB_lsqlin
carb = obj.endmembers_Carb_BB_lsqlin
anc = obj.endmembers_Anc_BB_lsqlin
apar = obj.APAR_spc_photons_lsqlin
par = obj.PAR_spc_photons
fluo = obj.FLUO_spc_photons
# transpose variables from raw_data
f_up = obj.raw_data.Fup.T
f_dw = obj.raw_data.Fdw.T
trans = obj.raw_data.trans_real.T
refl = obj.raw_data.refl_real.T
abs = obj.raw_data.abs_real.T


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
    return (1e6/(Na)) * (1e-9/(h*c)) * wl * (var*3.14)

Fup_spc_photons = to_photons(f_up)
Fdw_spc_photons = to_photons(f_dw)

#plt.plot(wvl_650_850, get_wvl_subset(Fdw_spc_photons, wvl_650_850))
#plt.show()

# Van Wittenberghe Methodology
def van_wittenberghe(f_up, f_dw, refl, trans, wvl):
    f_up = get_wvl_subset(f_up, wvl)
    f_dw = get_wvl_subset(f_dw, wvl)
    refl = get_wvl_subset(refl, wvl)
    trans = get_wvl_subset(trans, wvl)

    F_ps = np.add(np.divide(f_dw, refl), np.divide(f_up, trans))
    

    f_total = np.add(f_up, f_dw)
    plt.plot(wvl, f_total)
    plt.title("f total")
    plt.show()
    f_esc = np.sqrt(np.divide(f_total, F_ps))
    plt.plot(wvl, f_esc)
    plt.title("f esc")
    plt.show()
    fluo_corrected = np.divide(f_total, f_esc)
    plt.title("Van Wittenberghe Methodology corrected Fluorescence")
    plt.plot(wvl, fluo_corrected)
    plt.show()

    #phi = np.divide(f_total, apar * f_esc)


# Gitelson Methodology
def gitelson(fluo, trans, refl, wvl):
    fluo = get_wvl_subset(fluo, wvl)
    trans = get_wvl_subset(trans, wvl)
    refl = get_wvl_subset(refl, wvl)
    
    plt.plot(wvl, fluo)
    plt.title("f total")
    plt.show()

    fluo_corrected = np.divide(fluo, np.add(trans, refl)) # esto es fps <-> f esc
    plt.plot(wvl, fluo_corrected)
    plt.title("Gitelson Methodology corrected Fluorescence")
    plt.show()

    f_esc = np.divide(fluo, fluo_corrected)
    plt.title("f esc")
    plt.plot(wvl, f_esc)
    plt.show()

    #phi = np.divide(fluo, apar * f_esc)


# P Methodology
def p(fluo, apar, a, wvl):
    print(wvl)

    fluo = get_wvl_subset(fluo, wvl)
    apar = get_wvl_subset(apar, wvl)
    a = get_wvl_subset(a, wvl)
    
    plt.plot(wvl, fluo)
    plt.title("f total")
    plt.show()

    norm = np.divide(apar, a)
    plt.title("apar norm")
    plt.plot(wvl, norm)
    plt.show()


    f_ps = np.divide(fluo, np.subtract(1, norm))
    plt.title("Fps")
    plt.plot(wvl, f_ps)
    plt.ylim(0,1)
    plt.show()

    f_esc = np.divide(fluo, f_ps)
    plt.title("f esc")
    plt.plot(wvl, f_esc)
    plt.ylim(0,1)
    plt.show()

    fluo_corrected = np.divide(fluo, f_esc)
    plt.title("P Methodology corrected Fluorescence")
    plt.ylim(0,1)
    plt.plot(wvl, fluo_corrected)
    plt.show()

    #phi = np.divide(fluo, apar * f_esc)

def test():
    plt.plot(wvl, abs)
    plt.title("absorbcion total medida")
    plt.ylim(0, 1)
    plt.xlim(400, 800)
    plt.show()

    par1 = get_wvl_subset(par, wvl_500_780)
    irr1 = get_wvl_subset(irr, wvl_500_780)
    abs_p = par1 * (chla + chlb + carb + anc) / irr1
    plt.title("absorbcion suma pigmentos")
    plt.plot(wvl_500_780, abs_p)
    plt.show()


    diff = get_wvl_subset(abs, wvl_500_780) - abs_p
    plt.title("diff abs total - abs suma pigmentos")
    plt.plot(wvl_500_780, diff)
    plt.show()

    norm = np.divide(apar, par)
    f_ps = np.subtract(1, norm)
    plt.title("1 - abs")
    plt.plot(wvl_400_800, f_ps)
    plt.show()

    r_t = refl + trans
    plt.title("R + T")
    plt.plot(wvl, r_t)
    plt.show()

    diff1 = get_wvl_subset(r_t, wvl_400_800) - f_ps
    plt.title("diff R+T - 1-A")
    plt.plot(wvl_400_800, diff1)
    plt.show()


#gitelson(fluo, trans, refl, wvl_650_850)
#van_wittenberghe(Fup_spc_photons, Fdw_spc_photons, refl, trans, wvl_650_850)
#p(fluo, to_photons(apar), to_photons(par), np.intersect1d(wvl_500_780, wvl_650_850))
test()
