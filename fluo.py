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

@dataclass
class Result:
    fluo: np.ndarray | None
    escape: np.ndarray | None
    corrected: np.ndarray | None
    wvl: np.ndarray | None


class Fluo():
    
    def __init__(self, filename):
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

        self.days = [0, -1]

    # read file and determine nitrogen level
    def read_file(self, filename):
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
    def get_wvl_ref(self, arr):
        match len(arr):
            case 2151: return self.wvl
            case 401: return self.wvl_400_800
            case 201: return self.wvl_650_850
            case 281: return self.wvl_500_780
            case 131: return np.intersect1d(self.wvl_500_780, self.wvl_650_850)
            case _: raise ValueError("No valid wavelength refernce!")

    # get subset of spectrum
    def get_wvl_subset(self, arr, sub):
        ref = self.get_wvl_ref(arr)
        pos1A = np.argmin(np.abs(ref - sub[0]))
        pos2A = np.argmin(np.abs(ref - sub[-1]))
        arr = np.asarray(arr).squeeze()
        return arr[pos1A:pos2A+1]

    # convert variables in watts from raw_data to photons
    def to_photons(self, var):
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

    # Van Wittenberghe Methodology
    def van_wittenberghe(self, f_up, f_dw, refl, trans, wvl):
        f_up = self.get_wvl_subset(f_up, wvl)
        f_dw = self.get_wvl_subset(f_dw, wvl)
        refl = self.get_wvl_subset(refl, wvl)
        trans = self.get_wvl_subset(trans, wvl)

        F_ps = (f_up / refl) + (f_dw / trans)
        f_total = f_up + f_dw
        f_esc = np.sqrt((f_total / F_ps))
        fluo_corrected = f_total / f_esc
        return Result(self.fluo, f_esc, fluo_corrected, wvl)

    # Gitelson Methodology
    def gitelson(self, fluo, trans, refl, wvl):
        fluo = self.get_wvl_subset(fluo, wvl)
        trans = self.get_wvl_subset(trans, wvl)
        refl = self.get_wvl_subset(refl, wvl)

        fluo_corrected = fluo / (trans + refl)
        f_esc = fluo / fluo_corrected
        return Result(fluo, f_esc, fluo_corrected, wvl)

    # P Methodology
    def p(self, fluo, apar, par, p, wvl):
        fluo = self.get_wvl_subset(fluo, wvl)
        apar = self.get_wvl_subset(apar, wvl)
        abs_p = self.get_wvl_subset(par, self.wvl_500_780) * p
        a = self.get_wvl_subset(abs_p, wvl)


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
    
    def diff(self, a: Result, b: Result):
        common_wvl = np.intersect1d(a.wvl, b.wvl)
        corrected =  self.get_wvl_subset(a.corrected, common_wvl) - self.get_wvl_subset(b.corrected, common_wvl)
        escape = self.get_wvl_subset(a.escape, common_wvl) - self.get_wvl_subset(b.escape, common_wvl)
        return Result(None, escape, corrected, np.intersect1d(a.wvl, b.wvl))

    def plot_data(self):
        fig, ax = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='row')
        keys = list(self.methods.keys())
        for i, k in enumerate(keys):
            ax[0, i].set_title(k)
            attr = [self.methods[k].fluo, self.methods[k].corrected, self.methods[k].escape]
            for a in range(len(attr)):
                ax[a, i].plot(self.methods[k].wvl, attr[a][:, self.days])
            
        fig.suptitle(self.n_label)
        fig.legend(['Day 1', 'Day 9'])
        fig.tight_layout()
        plt.show()

    def plot_diff(self):
        p_gitelson = self.diff(self.methods['P'], self.methods['Gitelson'])
        p_vanW = self.diff(self.methods['P'], self.methods['Van Wittenberghe'])

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].set_title('P - Gitelson Corrected Fluo')
        ax[0, 0].plot(p_gitelson.wvl, p_gitelson.corrected[:, self.days])

        ax[0, 1].set_title('P - Van Wittenberghe Corrected Fluo')
        ax[0, 1].plot(p_vanW.wvl, p_vanW.corrected[:, self.days])

        ax[1, 0].set_title('P - Gitelson F Escape')
        ax[1, 0].plot(p_gitelson.wvl, p_gitelson.escape[:, self.days])

        ax[1, 1].set_title('P - Van Wittenberghe F Escape')
        ax[1, 1].plot(p_vanW.wvl, p_vanW.escape[:, self.days])
        fig.suptitle('diff')
        fig.legend(['Day 1', 'Day 9'])
        fig.tight_layout()
        plt.show()

    def plot_pigments(self):
        total = self.chla+self.chlb+self.anc+self.carb
        pigments = [self.chla, self.chlb, self.anc, self.carb, total]
        names = ['ChlA', 'ChlB', 'Anc', 'Carb', 'Sum']
        fig, ax = plt.subplots(nrows=3, ncols=5, sharex='col', sharey='all')
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
            ax[0, i].plot(self.wvl_500_780, pigment[:, self.days])
            for a, attr in enumerate(attributes):
                ax[a+1, i].plot(res.wvl, attr[:, self.days])

        fig.suptitle('Pigments')
        fig.legend(['Day 1', 'Day 9'])
        fig.tight_layout()
        plt.show()


# create Fluo analysis
n3 = Fluo('data/Fluowat_2055_n3_result.mat')
# plot analysis, pigments and differences
n3.plot_data()
n3.plot_pigments()
n3.plot_diff()
