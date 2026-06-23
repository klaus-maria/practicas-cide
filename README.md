# Fluo Analysis Tool

A Python-based fluorescence analysis tool for processing Fluowat MATLAB (`.mat`) output files and comparing different fluorescence escape correction methodologies.

## Overview

This tool loads Fluowat simulation results and computes fluorescence correction and fluorescence escape (`F_escape`) using three methodologies:

1. **Gitelson**
2. **Van Wittenberghe**
3. **MaPi (Pigment-based correction)**

The tool provides visualization of:

* Measured fluorescence
* Corrected fluorescence
* Fluorescence escape factors
* Differences between methodologies
* Individual pigment contributions

## References

### Van Wittenberghe et al.

https://www.sciencedirect.com/science/article/pii/S0034425724003122

### Gitelson et al.

https://www.sciencedirect.com/science/article/pii/S0273117797011332

---

## Requirements

### Python Packages

```bash
pip install scipy matplotlib numpy
```

### Dependencies

```python
scipy
matplotlib
numpy
dataclasses
```

---

## Input Data

The script expects a MATLAB `.mat` file containing a structure:

```text
result
 └── [n1 or n3 or n5]
```

The nitrogen level is automatically selected based on the filename:

| Filename contains | Dataset used |
| ----------------- | ------------ |
| `n1`              | `result.n1`  |
| `n3`              | `result.n3`  |
| `n5`              | `result.n5`  |

Example:

```text
Fluowat_2055_n3_result.mat
```

loads:

```python
result.n3
```

---

## Usage

### Default

```bash
python fluo.py
```

Uses:

```text
data/Fluowat_2055_n3_result.mat
```

### Specify File

```bash
python fluo.py my_result_n5.mat
```

### Plot Single Day

```bash
python fluo.py my_result_n5.mat 25
```

Equivalent to:

```python
startday = 25
endday = 25
```

### Plot Multiple Days

```bash
python fluo.py my_result_n5.mat 25 50
```

Equivalent to:

```python
startday = 25
endday = 50
```

---

## Methodologies

### 1. Gitelson

Computes corrected fluorescence using reflectance and transmittance:

```text
F_corrected = F / (R + T)
```

Fluorescence escape:

```text
F_escape = F / F_corrected
```

Inputs:

* Fluorescence
* Reflectance
* Transmittance

---

### 2. Van Wittenberghe

Uses upward and downward fluorescence measurements:

```text
F_corrected = (Fup / R) + (Fdw / T)
```

```text
F_total = Fup + Fdw
```

```text
F_escape = F_total / F_corrected
```

Inputs:

* Upward fluorescence
* Downward fluorescence
* Reflectance
* Transmittance

---

### 3. MaPi (Pigment-Based)

Uses pigment concentrations and absorbed PAR.

Pigments:

* Chlorophyll A
* Chlorophyll B
* Anthocyanins
* Carotenoids

Pigment absorption:

```text
APAR_pigment = PAR × Pigment
```

Normalization:

```text
Norm = APAR_pigment / PAR
```

Radiative transfer term:

```text
RT = 1 - Norm
```

Correction:

```text
F_corrected = F / RT
```

Escape factor:

```text
F_escape = F / F_corrected
```

---

## Wavelength Ranges

The code works with several predefined wavelength subsets:

| Variable      | Range (nm)    |
| ------------- | ------------- |
| `wvl`         | Full spectrum |
| `wvl_400_800` | 400–800       |
| `wvl_500_780` | 500–780       |
| `wvl_650_850` | 650–850       |
| Intersection  | 650–780       |

The appropriate wavelength reference is selected automatically from array size.

---

## Unit Conversion

Raw fluorescence data (`Fup`, `Fdw`) are converted from radiometric units:

```text
W m⁻² sr⁻¹ nm⁻¹
```

to:

```text
µmol photons m⁻² s⁻¹
```

using:

* Planck constant
* Speed of light
* Avogadro constant

---

## Generated Figures

### 1. Data Comparison

```python
plot_data()
```

Produces a 3 × 3 figure showing:

Rows:

1. Measured fluorescence
2. Corrected fluorescence
3. Fluorescence escape

Columns:

1. Gitelson
2. Van Wittenberghe
3. MaPi

---

### 2. Pigment Contributions

```python
plot_pigments()
```

Shows:

Columns:

* Chlorophyll A
* Chlorophyll B
* Anthocyanins
* Carotenoids
* Total pigment sum

Rows:

1. Pigment abundance
2. Corrected fluorescence
3. Fluorescence escape

---

### 3. Method Differences

```python
plot_diff()
```

Compares:

```text
MaPi − Gitelson
```

and

```text
MaPi − Van Wittenberghe
```

for:

* Corrected fluorescence
* Fluorescence escape

---

## Main Classes

### Result

Container class storing methodology results.

```python
Result(
    fluo,
    escape,
    corrected,
    wvl
)
```

Attributes:

| Attribute   | Description                |
| ----------- | -------------------------- |
| `fluo`      | Original fluorescence      |
| `escape`    | Fluorescence escape factor |
| `corrected` | Corrected fluorescence     |
| `wvl`       | Wavelength axis            |

---

### Fluo

Main analysis class.

```python
analysis = Fluo(filename)
```

Available methods:

```python
plot_data(startday, endday)
plot_pigments(startday, endday)
plot_diff(startday, endday)
```

---

## Example

```python
analysis = Fluo("data/Fluowat_2055_n3_result.mat")

analysis.plot_data(startday=20, endday=100)
analysis.plot_pigments(startday=20, endday=100)
analysis.plot_diff(startday=20, endday=100)
```

---

## Things that will break the code
* Changes in the .mat naming schema, structure and variable names
* Changes in wavelength and dataset array lengths
