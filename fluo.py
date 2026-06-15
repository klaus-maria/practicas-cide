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
n1 = mat['result'].n1

print(list(vars(mat['result']).keys()))
print(list(vars(n1).keys()))
print(list(vars(n1.raw_data).keys()))


# find closest indices (MATLAB min(abs(...))), use same wavelengths for all variables
def get_wvl_ref(arr):
    match len(arr):
        case 2151: return wvl
        case 401: return wvl_400_800
        case 201: return wvl_650_850
        case 281: return wvl_500_780
        case 131: return np.intersect1d(wvl_500_780, wvl_650_850)
        case _: print("No valid wavelength refernce!")

# get subset of spectrum
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


class Result:
    def __init__(self, fluo, escape, corrected, wvl):
        self.fluo = fluo
        self.escape = escape
        self.corrected = corrected
        self.wvl = wvl


# get variables
wvl = n1.raw_data.wvl[0].T
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


# Van Wittenberghe Methodology
def van_wittenberghe(f_up=f_up, f_dw=f_dw, refl=refl, trans=trans, wvl=wvl_650_850):
    f_up = get_wvl_subset(f_up, wvl)
    f_dw = get_wvl_subset(f_dw, wvl)
    refl = get_wvl_subset(refl, wvl)
    trans = get_wvl_subset(trans, wvl)

    F_ps = np.add(np.divide(f_up, refl), np.divide(f_dw, trans))
    f_total = np.add(f_up, f_dw)
    f_esc = np.sqrt(np.divide(f_total, F_ps))
    fluo_corrected = np.divide(f_total, f_esc)
    return Result(fluo, f_esc, fluo_corrected, wvl)

# Gitelson Methodology
def gitelson(fluo=fluo, trans=trans, refl=refl, wvl=wvl_650_850):
    fluo = get_wvl_subset(fluo, wvl)
    trans = get_wvl_subset(trans, wvl)
    refl = get_wvl_subset(refl, wvl)

    fluo_corrected = np.divide(fluo, np.add(trans, refl))
    f_esc = np.divide(fluo, fluo_corrected)
    return Result(fluo, f_esc, fluo_corrected, wvl)

# P Methodology
def p(fluo=fluo, apar=apar, par=par, p=(chla+chlb+anc+carb), wvl=np.intersect1d(wvl_500_780, wvl_650_850)):
    fluo = get_wvl_subset(fluo, wvl)
    apar = get_wvl_subset(apar, wvl)
    abs_p = get_wvl_subset(par, wvl_500_780) * p
    a = get_wvl_subset(abs_p, wvl)


    eps = 1e-1  # avoid division by zero
    

    norm = np.divide(apar, a)

    # correction factor: fraction of excitation NOT lost
    escape_fraction = 1 - norm

    # avoid blow-ups
    escape_fraction = np.clip(escape_fraction, eps, 1)

    # corrected fluorescence
    fluo_corrected = fluo / escape_fraction

    # escape ratio (optional)
    f_esc = fluo / fluo_corrected

    #fluo_corrected = np.divide(fluo, np.subtract(1, norm))
    #f_esc = np.divide(fluo, fluo_corrected)
    return Result(fluo, f_esc, fluo_corrected, wvl)


"""
vars:
nitrogen levels (n1, n3, n5)
days (1..9)
methods
"""


methods = {
    "Gitelson": gitelson(),
    "Van Wittenberghe": van_wittenberghe(),
    "P": p()
}

days = [0, -1]


def diff(a: Result, b: Result):
    corrected =  get_wvl_subset(a.corrected, np.intersect1d(a.wvl,b.wvl)) - get_wvl_subset(b.corrected, np.intersect1d(a.wvl,b.wvl))
    escape = get_wvl_subset(a.escape, np.intersect1d(a.wvl,b.wvl)) - get_wvl_subset(b.escape, np.intersect1d(a.wvl,b.wvl))
    return Result(None, escape, corrected, np.intersect1d(a.wvl, b.wvl))


def plot_data():
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='row')
    keys = list(methods.keys())
    for i in range(len(keys)):
        m = keys[i]
        ax[0, i].set_title(m)
        attr = [methods[m].fluo, methods[m].corrected, methods[m].escape]
        for a in range(len(attr)):
            ax[a, i].plot(methods[m].wvl, attr[a][:, days])
        
    fig.suptitle('n1')
    plt.show()


def plot_diff():
    p_gitelson = diff(p(), gitelson())
    p_vanW = diff(p(), van_wittenberghe())

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].set_title('P - Gitelson Corrected Fluo')
    ax[0, 0].plot(p_gitelson.wvl, p_gitelson.corrected[:, days])

    ax[0, 1].set_title('P - Van Wittenberghe Corrected Fluo')
    ax[0, 1].plot(p_vanW.wvl, p_vanW.corrected[:, days])

    ax[1, 0].set_title('P - Gitelson F Escape')
    ax[1, 0].plot(p_gitelson.wvl, p_gitelson.escape[:, days])

    ax[1, 1].set_title('P - Van Wittenberghe F Escape')
    ax[1, 1].plot(p_vanW.wvl, p_vanW.escape[:, days])
    fig.suptitle('diff')
    plt.show()


def plot_pigments():
    total = chla+chlb+anc+carb
    pigments = [chla, chlb, anc, carb, total]
    names = ['ChlA', 'ChlaB', 'Anc', 'Carb', 'Sum']
    fig, ax = plt.subplots(nrows=3, ncols=5, sharex='col', sharey='all')
    for i in range(len(pigments)):
        res = p(p=pigments[i])
        attr = [res.corrected, res.escape]
        ax[0, i].set_title(names[i])
        ax[0, i].plot(wvl_500_780, pigments[i][:, days])
        for a in range(len(attr)):
            ax[a+1, i].plot(res.wvl, attr[a][:, days])

    fig.suptitle('Pigments')
    plt.show()



#plot_data()
#plot_pigments()
plot_diff()
