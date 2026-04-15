import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


folder = "C:/Users/klaus/Desktop/uni_es/practicas_cide/2026_Klaus"
"""
wh - white reference
dc - dark current
wh2 - white reference after experiments
dc2 - dark current after experiments
fp - flower purple
pd - pine dead
pg - plant green
qu - quercus
ro - rock
sd - soil dry
sw - soil wet
"""
types = ["wh", "dc", "wh2", "dc2", "fp", "pd", "pg", "qu", "ro", "sd", "sw"]


def find_files(folder, prefix):
    file_list = []
    for file in os.listdir(folder):
        filename = Path(file).name
        if filename.startswith(prefix):
                file_list.append(file)
    return file_list


def parse_with_pandas(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    integration_time = float(
        [l for l in lines if "Integration Time" in l][0].split(":")[1]
    )
    start = lines.index(">>>>>Begin Spectral Data<<<<<\n") + 1
    df = pd.read_csv(
        file_path,
        skiprows=start,
        sep=r"\s+",
        names=["wavelength", "intensity"]
    )
    return integration_time, df


def get_normalised_intensities(files):
    dataframes = []
    
    for file in files:
        integration_time, data = parse_with_pandas(folder+"/"+file)
        dataframes.append(data)
    
    df_concat = pd.concat(dataframes)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_normalised = df_means.copy()
    df_normalised["intensity"] = df_normalised["intensity"] / integration_time
    
    print(integration_time)
    return df_normalised


def subtract_df(df, sub):
    diff = df.copy()
    diff["intensity"] = diff["intensity"] - sub["intensity"]
    return diff


def get_signature(folder, prefix):
    files = find_files(folder, prefix)
    df = get_normalised_intensities(files)
    print(df)
    return df


def get_reflectance(folder, prefix, diff_white_ref, dark_current):
    radiance = get_signature(folder, prefix)
    plt.subplot(1,3,1)
    plt.plot(radiance["wavelength"], radiance["intensity"])
    plt.xlim(400, 800)
    plt.title(f"{prefix} (total radiance)")

    radiance_corrected = subtract_df(radiance, dark_current)
    plt.subplot(1,3,2)
    plt.plot(radiance_corrected["wavelength"], radiance_corrected["intensity"])
    plt.xlim(400, 800)
    plt.title(f"{prefix} corrected for dark current")

    reflectance = radiance_corrected.copy()
    reflectance["intensity"] = reflectance["intensity"] / diff_white_ref["intensity"]
    print(reflectance)
    plt.subplot(1,3,3)
    plt.plot(reflectance["wavelength"], reflectance["intensity"])
    plt.ylim(0, 1)
    plt.xlim(400, 800)
    plt.title(f"{prefix} signature (reflectance)")
    plt.show()



# get white reference
white_ref = get_signature(folder, "wh")
plt.subplot(1,3,1)
plt.plot(white_ref["wavelength"], white_ref["intensity"])
plt.xlim(400, 800)
plt.title("white reference (total)")

# get dark current
dark_current_ref = get_signature(folder, "dc")
plt.subplot(1,3,2)
plt.plot(dark_current_ref["wavelength"], dark_current_ref["intensity"])
plt.xlim(400, 800)
plt.title("dark current")

# get difference white reference and dark current
diff_wh_dc = subtract_df(white_ref, dark_current_ref)
plt.subplot(1,3,3)
plt.plot(diff_wh_dc["wavelength"], diff_wh_dc["intensity"])
plt.xlim(400, 800)
plt.title("white reference minus dark current")
plt.show()


# get materials
get_reflectance(folder, "sd", diff_wh_dc, dark_current_ref) # soil dry