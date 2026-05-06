import scipy.io
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np

mat = scipy.io.loadmat('data/Fluowat_2055_n1_result.mat', struct_as_record=False, squeeze_me=True)

obj = mat['result'].n1

wvl = obj.wvl_500_780
irr = obj.irr_PAR_400_800
chla = obj.endmembers_Chla_BB_lsqlin
chlb = obj.endmembers_Chlb_BB_lsqlin
carb = obj.endmembers_Carb_BB_lsqlin
anc = obj.endmembers_Anc_BB_lsqlin
apar = obj.APAR_Chl_lsqlin
fluo = obj.FLUO_photons
f_up = obj.
f_dw = obj.

"""

1. APAR calculado con NDVI
2. APAR de datos total
3. APAR solo chla
4. APAR solo chlb
5. APAR solo chla+b


questions: which vars from MATLAB exactly? photons version? lsqlin or lsqnonneg?
    where is f_up, f_dw?

https://www.sciencedirect.com/science/article/pii/S0034425724003122

SIF -> sun induced chlorophyll fluorescence
F -> Fluorescence emission
APAR -> absorbed photosynthetically active radiation
ΦF -> quantum yield of fluorescence (photosystem level)
f_esc -> escape probability of photons (not absorbed/scattered)
λ -> wavelength
R -> Reflectance

Sun Induced Fluorescence can be calculated with:
    SIF = APAR * ΦF * f_esc


Fluorescence Quantum Yield can be calculated with (incorporating wavelength effects):
    ΦF(λ) = SIF(λ) / (APAR * f_esc(λ))

    
SIF normalized by APAR or apparent SIF: ???
    FY_λ = SIF(λ) / APAR


SIF at Photosystem level:
    F_λ,PS = F_λ / f_esc,λ


APAR (if no measurements) can be calculated with:
    APAR = 1.37 * NDVI_re - 0.17  // (NDVI_re = R750, R705)

    
Escape Probability (leaf level) can be calulated with:
    f_esc = sqrt( (F_up + F_dw) / F )

    
Fluorescence Quantum Yield at leaf level can be calculated with:
    ΦF = F / PAR * APAR * f_esc
"""



# find closest indices (MATLAB min(abs(...)))
pos1A = np.argmin(np.abs(wvl - 500))
pos2A = np.argmin(np.abs(wvl - 780))

# IMPORTANT: Python slicing excludes the end → +1
APARChla = irr[pos1A:pos2A+1] * chla
APARChlb = irr[pos1A:pos2A+1] * chlb

plt.plot(APARChla)
plt.plot(APARChlb)
plt.show()

plt.plot(chla)
plt.plot(chlb)
plt.show()