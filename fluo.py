import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

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



# read file and determine nitrogen level
def read_file(filename='data/Fluowat_2055_n1_result.mat'):
    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    print(list(vars(mat['result']).keys()))
    if 'n1' in filename:
        return mat['result'].n1, 'n1'
    elif 'n3' in filename:
        return mat['result'].n3, 'n3'
    elif 'n5' in filename:
        return mat['result'].n5, 'n5'
    else:
        raise ValueError('Could not find nitrogen level from filename!')

# find closest indices (MATLAB min(abs(...))), use same wavelengths for all variables
def get_wvl_ref(arr):
    match len(arr):
        case 2151: return wvl
        case 401: return wvl_400_800
        case 201: return wvl_650_850
        case 281: return wvl_500_780
        case 131: return np.intersect1d(wvl_500_780, wvl_650_850)
        case _: raise ValueError("No valid wavelength refernce!")

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


@dataclass
class Result:
    fluo: np.ndarray | None
    escape: np.ndarray | None
    corrected: np.ndarray | None
    wvl: np.ndarray | None

# read data
n, nitro_label = read_file('data/Fluowat_2055_n3_result.mat')
print(list(vars(n).keys()))
print(list(vars(n.raw_data).keys()))

# get variables
wvl = n.raw_data.wvl[0].T
wvl_400_800 = n.wvl_400_800
wvl_500_780 = n.wvl_500_780
wvl_650_850 = n.wvl_650_850
irr = n.irr_PAR_400_800
chla = n.endmembers_Chla_BB_lsqlin
chlb = n.endmembers_Chlb_BB_lsqlin
carb = n.endmembers_Carb_BB_lsqlin
anc = n.endmembers_Anc_BB_lsqlin
apar = n.APAR_spc_photons_lsqlin
par = n.PAR_spc_photons
fluo = n.FLUO_spc_photons
# transpose variables from raw_data and convert from watts to photons
f_up = to_photons(n.raw_data.Fup.T)
f_dw = to_photons(n.raw_data.Fdw.T)
trans = n.raw_data.trans_real.T
refl = n.raw_data.refl_real.T
abs = n.raw_data.abs_real.T


# Van Wittenberghe Methodology
def van_wittenberghe(f_up=f_up, f_dw=f_dw, refl=refl, trans=trans, wvl=wvl_650_850):
    f_up = get_wvl_subset(f_up, wvl)
    f_dw = get_wvl_subset(f_dw, wvl)
    refl = get_wvl_subset(refl, wvl)
    trans = get_wvl_subset(trans, wvl)

    F_ps = (f_up / refl) + (f_dw / trans)
    f_total = f_up + f_dw
    f_esc = np.sqrt((f_total / F_ps))
    fluo_corrected = f_total / f_esc
    return Result(fluo, f_esc, fluo_corrected, wvl)

# Gitelson Methodology
def gitelson(fluo=fluo, trans=trans, refl=refl, wvl=wvl_650_850):
    fluo = get_wvl_subset(fluo, wvl)
    trans = get_wvl_subset(trans, wvl)
    refl = get_wvl_subset(refl, wvl)

    fluo_corrected = fluo / (trans + refl)
    f_esc = fluo / fluo_corrected
    return Result(fluo, f_esc, fluo_corrected, wvl)

# P Methodology
def p(fluo=fluo, apar=apar, par=par, p=(chla+chlb+anc+carb), wvl=np.intersect1d(wvl_500_780, wvl_650_850)):
    fluo = get_wvl_subset(fluo, wvl)
    apar = get_wvl_subset(apar, wvl)
    abs_p = get_wvl_subset(par, wvl_500_780) * p
    a = get_wvl_subset(abs_p, wvl)


    eps = 1e-1  # avoid division by zero
    

    norm = apar / a

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


methods = {
    "Gitelson": gitelson(),
    "Van Wittenberghe": van_wittenberghe(),
    "P": p()
}

days = [0, -1]


def diff(a: Result, b: Result):
    common_wvl = np.intersect1d(a.wvl, b.wvl)
    corrected =  get_wvl_subset(a.corrected, common_wvl) - get_wvl_subset(b.corrected, common_wvl)
    escape = get_wvl_subset(a.escape, common_wvl) - get_wvl_subset(b.escape, common_wvl)
    return Result(None, escape, corrected, np.intersect1d(a.wvl, b.wvl))


def plot_data():
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='row')
    keys = list(methods.keys())
    for i, k in enumerate(keys):
        ax[0, i].set_title(k)
        attr = [methods[k].fluo, methods[k].corrected, methods[k].escape]
        for a in range(len(attr)):
            ax[a, i].plot(methods[k].wvl, attr[a][:, days])
        
    fig.suptitle(nitro_label)
    fig.legend(['Day 1', 'Day 9'])
    fig.tight_layout()
    plt.show()


def plot_diff():
    p_gitelson = diff(methods['P'], methods['Gitelson'])
    p_vanW = diff(methods['P'], methods['Van Wittenberghe'])

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
    fig.legend(['Day 1', 'Day 9'])
    fig.tight_layout()
    plt.show()


def plot_pigments():
    total = chla+chlb+anc+carb
    pigments = [chla, chlb, anc, carb, total]
    names = ['ChlA', 'ChlB', 'Anc', 'Carb', 'Sum']
    fig, ax = plt.subplots(nrows=3, ncols=5, sharex='col', sharey='all')
    for i, pigment in enumerate(pigments):
        res = p(p=pigment)
        attributes = [res.corrected, res.escape]
        ax[0, i].set_title(names[i])
        ax[0, i].plot(wvl_500_780, pigment[:, days])
        for a, attr in enumerate(attributes):
            ax[a+1, i].plot(res.wvl, attr[:, days])

    fig.suptitle('Pigments')
    fig.legend(['Day 1', 'Day 9'])
    fig.tight_layout()
    plt.show()



plot_data()
plot_pigments()
plot_diff()
