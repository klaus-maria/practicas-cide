import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import sys

"""
REFERNCES

Van Wittenberghe
https://www.sciencedirect.com/science/article/pii/S0034425724003122

Gitelson
https://www.sciencedirect.com/science/article/pii/S0273117797011332
"""

"""
TODO:
- y axis label
"""

@dataclass
class Result:
    fluo: np.ndarray | None
    escape: np.ndarray | None
    corrected: np.ndarray | None
    wvl: np.ndarray | None


class Fluo():
    """
    Create a Fluo object. Reads file, retrieves attributes, computes methodologies.
    :param filename: Filepath to .mat file containing data.
    """
    def __init__(self, filename: str):
        self.n, self.n_label = self.read_file(filename)
        print(list(vars(self.n).keys()))
        print(list(vars(self.n.raw_data).keys()))

        # get variables
        self.wvl = self.n.raw_data.wvl[0].T
        self.wvl_400_800 = self.n.wvl_400_800
        self.wvl_500_780 = self.n.wvl_500_780
        self.wvl_650_850 = self.n.wvl_650_850
        self.irr = self.n.irr_PAR_400_800
        self.chla = self.n.endmembers_Chla_BB_lsqlin
        self.chlb = self.n.endmembers_Chlb_BB_lsqlin
        self.carb = self.n.endmembers_Carb_BB_lsqlin
        self.anc = self.n.endmembers_Anc_BB_lsqlin
        self.apar = self.n.APAR_spc_photons_lsqlin
        self.par = self.n.PAR_spc_photons
        self.fluo = self.n.FLUO_spc_photons
        # transpose variables from raw_data and convert from watts to photons
        self.f_up = self.to_photons(self.n.raw_data.Fup.T)
        self.f_dw = self.to_photons(self.n.raw_data.Fdw.T)
        self.trans = self.n.raw_data.trans_real.T
        self.refl = self.n.raw_data.refl_real.T
        self.abs = self.n.raw_data.abs_real.T

        self.methods = {
            "Gitelson": self.gitelson(
                fluo=self.fluo,
                trans=self.trans,
                refl=self.refl,
                wvl=self.wvl_650_850
            ),
            "Van Wittenberghe": self.van_wittenberghe(
                f_up=self.f_up,
                f_dw=self.f_dw,
                refl=self.refl,
                trans=self.trans,
                wvl=self.wvl_650_850
            ),
            "P": self.p(
                fluo=self.fluo,
                apar=self.apar,
                par=self.par,
                p=(self.chla+self.chlb+self.anc+self.carb),
                wvl=np.intersect1d(self.wvl_500_780, self.wvl_650_850)
            )
        }


    """
    Read .mat file, expects a matlab structure with the structure [result].[nitrogen_level].
    Returns matlab structure object and filename.
    :param: filename: Filepath to .mat file.
    """
    def read_file(self, filename: str) -> list[object, str]:
        mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        print(list(vars(mat['result']).keys()))
        if 'n1' in filename:
            return mat['result'].n1, filename
        elif 'n3' in filename:
            return mat['result'].n3, filename
        elif 'n5' in filename:
            return mat['result'].n5, filename
        else:
            raise ValueError('Could not find nitrogen level from filename!')

    """
    Get the wavelength spectrum of a dataset based on its length.
    :param arr: 
    """
    def get_wvl_ref(self, arr: np.array) -> np.array:
        match len(arr):
            case 2151: return self.wvl
            case 401: return self.wvl_400_800
            case 201: return self.wvl_650_850
            case 281: return self.wvl_500_780
            case 131: return np.intersect1d(self.wvl_500_780, self.wvl_650_850)
            case _: raise ValueError("No valid wavelength refernce!")

    """
    Get wvelength spectrum subset of a dataset. Returns input array in given wavelength subset.
    :param arr: input array.
    :param sub: wavelength subset.
    """
    def get_wvl_subset(self, arr: np.array, sub: np.array) -> np.array:
        ref = self.get_wvl_ref(arr)
        pos1A = np.argmin(np.abs(ref - sub[0]))
        pos2A = np.argmin(np.abs(ref - sub[-1]))
        arr = np.asarray(arr).squeeze()
        return arr[pos1A:pos2A+1]

    """
    Convert array from electronic counts to photons.
    :param var: input array.
    """
    def to_photons(self, var: np.array) -> np.array:
        wvl = self.get_wvl_ref(var)
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

    """
    Van Wittenberghe Methodology
    :param f_up: upwards fluorescence.
    :param f_dw: downwards fluorescence.
    :param refl: reflectance.
    :param trans: transmittance.
    :param wvl: wavelength spectrum to use.
    """
    def van_wittenberghe(self, f_up: np.array, f_dw: np.array, refl: np.array, trans: np.array, wvl: np.array) -> Result:
        f_up = self.get_wvl_subset(f_up, wvl)
        f_dw = self.get_wvl_subset(f_dw, wvl)
        refl = self.get_wvl_subset(refl, wvl)
        trans = self.get_wvl_subset(trans, wvl)

        fluo_corrected = (f_up / refl) + (f_dw / trans)
        f_total = f_up + f_dw
        f_esc = f_total / fluo_corrected
        return Result(self.fluo, f_esc, fluo_corrected, wvl)

    """
    Gitelson Methodology
    :param fluo: measured fluorescence.
    :param trans: trasnmittance.
    :param refl: reflectance.
    :param wvl: wavelength spectrum to use.
    """
    def gitelson(self, fluo: np.array, trans: np.array, refl: np.array, wvl: np.array) -> Result:
        fluo = self.get_wvl_subset(fluo, wvl)
        trans = self.get_wvl_subset(trans, wvl)
        refl = self.get_wvl_subset(refl, wvl)

        fluo_corrected = fluo / (trans + refl)
        f_esc = fluo / fluo_corrected
        return Result(fluo, f_esc, fluo_corrected, wvl)

    """
    MaPi Methodology
    :param fluo: measured fluorescence.
    :param apar: total absorption.
    :param par: irradiance.
    :param p: pigment(s).
    :param wvl: wavelength spectrum to use.
    """
    def p(self, fluo: np.array, apar: np.array, par: np.array, p: np.array, wvl: np.array) -> Result:
        fluo = self.get_wvl_subset(fluo, wvl)
        apar = self.get_wvl_subset(apar, wvl)
        apar_pigment = self.get_wvl_subset(par, self.wvl_500_780) * p
        apar_pigment = self.get_wvl_subset(apar_pigment, wvl)

        norm = apar_pigment / self.get_wvl_subset(par, wvl)
        rt = 1 - norm
        fluo_corrected = fluo / rt
        f_esc = fluo / fluo_corrected
        return Result(fluo, f_esc, fluo_corrected, wvl)
    
    """
    Get difference of corrected fluorescence and f_escape between 2 computed results.
    Returns new result with differences.
    :param a: first result.
    :param b: second result.
    """
    def diff(self, a: Result, b: Result) -> Result:
        common_wvl = np.intersect1d(a.wvl, b.wvl)
        corrected =  self.get_wvl_subset(a.corrected, common_wvl) - self.get_wvl_subset(b.corrected, common_wvl)
        escape = self.get_wvl_subset(a.escape, common_wvl) - self.get_wvl_subset(b.escape, common_wvl)
        return Result(None, escape, corrected, np.intersect1d(a.wvl, b.wvl))

    """
    Plot input fluorescence, corrected fluorescence and f escape for each methodology.
    """
    def plot_data(self, startday: int, endday: int):
        fig, ax = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='row')
        keys = list(self.methods.keys())
        for i, k in enumerate(keys):
            ax[0, i].set_title(k)
            attr = [self.methods[k].fluo, self.methods[k].corrected, self.methods[k].escape]
            for a in range(len(attr)):
                ax[a, i].plot(self.methods[k].wvl, attr[a][:, [startday, endday]])
            
        fig.suptitle(self.n_label)
        start_label = 'Day ' + str(startday)
        endday_label = 'Day ' + str(endday)
        fig.legend([start_label, endday_label])
        fig.tight_layout()
        plt.show()

    """
    Plot result differences of MaPi methodology to Gitelson and Van Wittenberghe.
    """
    def plot_diff(self, startday: int, endday: int):
        p_gitelson = self.diff(self.methods['P'], self.methods['Gitelson'])
        p_vanW = self.diff(self.methods['P'], self.methods['Van Wittenberghe'])

        fig, ax = plt.subplots(nrows=2, ncols=2, sharey='row')
        ax[0, 0].set_title('P - Gitelson Corrected Fluo')
        ax[0, 0].plot(p_gitelson.wvl, p_gitelson.corrected[:, [startday, endday]])

        ax[0, 1].set_title('P - Van Wittenberghe Corrected Fluo')
        ax[0, 1].plot(p_vanW.wvl, p_vanW.corrected[:, [startday, endday]])

        ax[1, 0].set_title('P - Gitelson F Escape')
        ax[1, 0].plot(p_gitelson.wvl, p_gitelson.escape[:, [startday, endday]])

        ax[1, 1].set_title('P - Van Wittenberghe F Escape')
        ax[1, 1].plot(p_vanW.wvl, p_vanW.escape[:, [startday, endday]])
        fig.suptitle('Differences Methodologies '+self.n_label)
        start_label = 'Day ' + str(startday)
        endday_label = 'Day ' + str(endday)
        fig.legend([start_label, endday_label])
        fig.tight_layout()
        plt.show()

    """
    Plot MaPi methodology fluorescence correction of each pigment seperately (its contribution) and pigments sum.
    """
    def plot_pigments(self, startday: int, endday: int):
        total = self.chla+self.chlb+self.anc+self.carb
        pigments = [self.chla, self.chlb, self.anc, self.carb, total]
        names = ['ChlA', 'ChlB', 'Anc', 'Carb', 'Sum']
        fig, ax = plt.subplots(nrows=3, ncols=5, sharex='col', sharey='row')
        for i, pigment in enumerate(pigments):
            res = self.p(
                fluo=self.fluo,
                apar=self.apar,
                par=self.par,
                p=pigment,
                wvl=np.intersect1d(self.wvl_500_780, self.wvl_650_850)
                )
            attributes = [res.corrected, res.escape]
            ax[0, i].set_title(names[i])
            ax[0, i].plot(self.wvl_500_780, pigment[:, [startday, endday]])
            for a, attr in enumerate(attributes):
                ax[a+1, i].plot(res.wvl, attr[:, [startday, endday]])
                #ax[a+1, i].set_ylim(0,1)


        fig.suptitle('Pigments MaPi Methodology '+self.n_label)
        start_label = 'Day ' + str(startday)
        endday_label = 'Day ' + str(endday)
        fig.legend([start_label, endday_label])
        fig.tight_layout()
        plt.show()



# get arguments
if len(sys.argv) < 2:
    print("Invalid filename provided! Using default file.")
    filename = "data/Fluowat_2055_n3_result.mat"
    #raise ValueError("No file provided!")
else:
    filename = sys.argv[1]

if len(sys.argv) >= 3:
    startday = int(sys.argv[2])

    if len(sys.argv) >= 4:
        endday = int(sys.argv[3])
    else:
        endday = startday
else:
    startday = 0
    endday = -1


# create Fluo analysis
analysis = Fluo(filename)

# plot analysis, pigments and differences
analysis.plot_data(startday=startday, endday=endday)
analysis.plot_pigments(startday=startday, endday=endday)
analysis.plot_diff(startday=startday, endday=endday)
