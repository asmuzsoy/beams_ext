#!/usr/bin/env python3
"""
Fit ZTF Bright Transient Survey light curves with both Ia (SALT2) and Ibc
(Arnett) models, for a chosen redshift treatment.

Usage:
    python3 fit_real_sne_new.py --z=spec
    python3 fit_real_sne_new.py --z=phot
    python3 fit_real_sne_new.py --z=photo  # alias for phot
    python3 fit_real_sne_new.py --z=none

This script is mechanically derived from fit_real_sne_new.ipynb. The fitting
code is unchanged from the notebook; only a CLI wrapper, output-path
plumbing, and per-z-type filenames have been added.
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime

# ---------- CLI ----------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--z", required=True,
    help="Redshift treatment: 'spec' (fixed to spec-z), 'phot' or 'photo' (free with photo-z prior), or 'none' (free, no prior).",
)
parser.add_argument(
    "--name", default=None,
    help="Output name prefix used in saved filenames (e.g. '041026'). Defaults to today's date YYMMDD.",
)
args = parser.parse_args()

# Normalize 'photo' -> 'phot'
Z_RAW = args.z.lower()
if Z_RAW == "photo":
    Z_TYPE = "phot"
elif Z_RAW in ("spec", "phot", "none"):
    Z_TYPE = Z_RAW
else:
    sys.exit(f"--z must be one of: spec, phot, photo, none (got '{args.z}')")

OUT_NAME = args.name if args.name is not None else datetime.now().strftime("%m%d%y")

# Output paths
FIG_DIR = os.path.join("figures", Z_TYPE)
os.makedirs(FIG_DIR, exist_ok=True)
OBJECTS_NPZ = f"fitted_sne_real_{OUT_NAME}_{Z_TYPE}.npz"
FAILED_IBC_TXT = f"failed_ibc_fits_{Z_TYPE}.txt"
FAILED_IA_TXT = f"failed_ia_fits_{Z_TYPE}.txt"

print(f"Redshift treatment: {Z_TYPE}")
print(f"Output objects file: {OBJECTS_NPZ}")
print(f"Figure directory: {FIG_DIR}")

# Use a non-interactive matplotlib backend so plots can save without a display
import matplotlib
matplotlib.use("Agg")

# ---------- Imports (from notebook cell 2) ----------
import sncosmo
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from pandas import read_csv, read_feather
from scipy.integrate import cumulative_trapezoid as cumtrapz
import astropy.constants as co
import astropy.units as u
import astropy.cosmology.units as cu
from scipy import interpolate
from astropy.cosmology import WMAP9 as cosmo
import extinction
import math
import scipy.stats as stats
from scipy.interpolate import CubicSpline
import time

DAY_CGS = u.day
M_SUN_CGS = co.M_sun
C_CGS = co.c
beta = 13.7
KM_CGS = u.km

STEF_CONST = 4. * np.pi * co.sigma_sb
ANG_CGS = u.Angstrom
MPC_CGS = u.Mpc

DIFF_CONST = 2.0 * M_SUN_CGS / (beta * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (4. * np.pi * KM_CGS ** 2)
FLUX_CONST = 4.0 * np.pi * (
        2.0 * co.h * co.c ** 2 * np.pi) * u.Angstrom
X_CONST = (co.h * co.c / co.k_B)


# ---------- Helpers (cells 3, 4) ----------
def guess_texp(lc, sigma=1):
    times = np.array(lc['time'])
    fluxes = np.array(lc['flux'])
    mean = np.mean(fluxes)
    std = np.std(fluxes)
    mask = fluxes < (mean + sigma * std)
    if np.any(mask):
        peak_idx = np.argmax(fluxes[mask])
        max_time = times[mask][peak_idx]
    else:
        max_time = times[np.argmax(fluxes)]
    if max_time < 15:
        texp_guess = 0
    elif max_time > 35:
        texp_guess = 20
    else:
        texp_guess = max_time - 15
    return texp_guess


def find_t_peak(lc, band='ztfr'):
    mask = lc['band'] == band
    times = np.array(lc['time'][mask])
    fluxes = np.array(lc['flux'][mask])
    t_peak = times[np.argmax(fluxes)]
    return t_peak


# ---------- ArnettSource (cell 5) ----------
class ArnettSource(sncosmo.Source):

    _param_names = ['redshift', 'texp', 'mej', 'fni', 'vej']
    param_names = ['redshift', 'texp', 'mej', 'fni', 'vej']

    param_names_latex = ['z', 't_{exp}', 'M_{ej}', 'f_{Ni}', 'v_{ej}']

    def __init__(self, phase, wave, params=None, name=None, version=None):

        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._tfloor = 3000 * u.K
        if params is not None:
            self._parameters = params
        else:
            self._parameters = [0, 0, 0, 0, 0]

    def _blackbody_flux(self, temperature, radius, wavelength):
        all_fluxes = np.zeros((len(temperature), len(wavelength))) * (u.erg / (u.s * u.AA)).decompose()
        for i in range(len(temperature)):
            temp = temperature[i]
            rad = radius[i]
            numerator = (2 * co.h * co.c**2 / wavelength**5).decompose()
            exponent = (co.h * co.c / (wavelength * co.k_B * temp)).decompose()
            denominator = np.exp(exponent.value) - 1
            flux_density = numerator / denominator
            flux_final = flux_density * (4 * np.pi * rad**2)
            all_fluxes[i, :] = flux_final
        return all_fluxes

    def _gen_arnett_model(self, t, wvs, theta):
        z, texp, mej, fni, vej = theta
        mej = (mej * u.Msun).to(u.g)
        vej = vej * u.km/u.s
        t = t * u.day
        wvs = wvs * u.AA
        tfloor = self._tfloor
        mni = mej * fni
        vej = vej.to(u.cm / u.s)

        tni = 8.8 * u.day
        tco = 111.3 * u.day
        epco = 6.8e9 * u.erg / u.g / u.s
        epni = 3.9e10 * u.erg / u.g / u.s
        opac = 0.1 * u.cm * u.cm / u.g
        texp = texp * u.day

        td = np.sqrt(2 * opac * mej / (13.7 * co.c * vej)).to(u.day)

        t_to_integrate = np.linspace(0, np.max(t - texp), 1000)

        integrand1 = (t_to_integrate / td) * np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tni)
        integrand2 = (t_to_integrate / td) * np.exp(t_to_integrate**2 / td**2 - t_to_integrate / tco)

        dense_luminosities = 2 * mni / (td) * np.exp(-t_to_integrate**2 / td**2) * \
              (((epni - epco) * cumtrapz(integrand1, t_to_integrate, initial=0) +
               epco * cumtrapz(integrand2, t_to_integrate, initial=0))) * u.day

        spline = CubicSpline(t_to_integrate, dense_luminosities, extrapolate=False)
        luminosities = spline(t - texp) * u.erg / u.s

        radius = (vej * ((t - texp) * ((t - texp) >= 0))).to(u.cm)

        temperature = ((luminosities / (STEF_CONST * radius**2))**0.25).to(u.K)
        temperature = np.maximum(temperature, tfloor)

        radius = np.sqrt(luminosities / (STEF_CONST * temperature**4))
        radius = radius.to(u.cm)

        fluxes = self._blackbody_flux(temperature, radius, wvs / (1 + z))

        fluxes[t < texp, :] = 0. * u.kg * u.m / u.s**3
        fluxes[np.isnan(fluxes)] = 0. * u.kg * u.m / u.s**3

        z = z * cu.redshift
        d_cm = z.to(u.cm, cu.redshift_distance(cosmo, kind="luminosity"))
        flux_density = fluxes / (4 * np.pi * d_cm**2)
        return flux_density / (1 + z.value)

    def _flux(self, phase, wave):
        return self._gen_arnett_model(phase, wave, self._parameters)


# ---------- Globals (cell 6) ----------
times = np.linspace(0.1, 100, 100)
wavelengths = np.linspace(2000, 12000, 10)

num_points = 10
time_points = np.linspace(0.01, 100, num_points)
num_points = len(time_points)


# ---------- Ibc metadata (cell 11) ----------
ibc_data = pd.read_csv("ibc_data.csv")

ibc_lookup = {}
for _, row in ibc_data.iterrows():
    try:
        if "pec" in row['type']:
            continue
        redshift = float(row['redshift'])
        av = float(row['A_V'])
        ibc_lookup[row['ZTFID']] = {'redshift': redshift, 'A_V': av}
    except (ValueError, TypeError):
        pass

def get_object_data_from_directory(directory_path):
    redshift_dict = {}
    av_dict = {}
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('_bts.csv')]
    for filename in csv_files:
        object_name = filename.split('_')[0]
        if object_name in ibc_lookup:
            redshift_dict[filename] = ibc_lookup[object_name]['redshift']
            av_dict[filename] = ibc_lookup[object_name]['A_V']
        else:
            print(f"Warning: {object_name} from {filename} not found in ibc_data.csv")
    return redshift_dict, av_dict

directory_path = "ibc_sims_to_fit"
redshift_dict, av_dict = get_object_data_from_directory(directory_path)
print(f"Matched {len(redshift_dict)} files to ibc_data.csv")


# ---------- Flux conversion + Ibc reader (cell 13) ----------
ZP = 25.0
ZPSYS = 'ab'

def mag_to_bandflux(mag, zp=ZP):
    return 10**(-0.4 * (mag - zp))

def magerr_to_fluxerr(mag, magerr, zp=ZP):
    f = mag_to_bandflux(mag, zp)
    return f * np.log(10) * 0.4 * magerr

def read_ibc_csv(filename, min_points_per_band=5):
    df = pd.read_csv("ibc_sims_to_fit/" + filename)
    tab = Table.from_pandas(df)
    magtype_mask = tab['magtype'] > 0
    tab = tab[magtype_mask]

    tab = tab[(tab['filter'] != 'i') & (tab['filter'] != 'I')]

    g_count = np.sum(tab['filter'] == 'g')
    r_count = np.sum(tab['filter'] == 'r')
    if g_count < min_points_per_band or r_count < min_points_per_band:
        return None

    tab['flux'] = mag_to_bandflux(tab['mag'])
    tab['fluxerr'] = magerr_to_fluxerr(tab['mag'], tab['dmag'])
    tab['zp'] = ZP * np.ones_like(tab['flux'])
    tab['zpsys'] = np.array([ZPSYS] * len(tab), dtype=str)
    tab['band'] = np.array(['ztf' + f for f in tab['filter']], dtype=str)
    tab['time'] = tab['mjd'] - np.min(tab['mjd'])

    tab.remove_column('mjd')
    tab.remove_column('filter')
    return tab


# ---------- Ia/Ibc fitting functions (cells 16, 21) ----------
def fit_ia_w_dust(true_z, lcs, true_av=None, true_mwebv=None, bands=None, redshift='spec'):
    ia_source = sncosmo.get_source('salt2-extended', version='2.0')
    dust = sncosmo.CCM89Dust()
    R_V = 3.1
    if true_mwebv is None:
        if true_av is None:
            raise ValueError("Either true_av or true_mwebv must be provided")
        true_mwebv = true_av / R_V

    this_ia_model = sncosmo.Model(source=ia_source,
                                  effects=[dust],
                                  effect_names=['mw'],
                                  effect_frames=['obs'])
    if redshift == 'spec':
        this_ia_model.set(z=true_z, mwebv=true_mwebv)
        params_to_fit = ['t0', 'x0', 'x1', 'c']
        guess_z_value = False
        bounds = {'x0': (0, 0.1), 'x1': (-5, 50), 'c': (-1, 5), 't0': (0, 100)}
    elif redshift == 'phot':
        this_ia_model.set(mwebv=true_mwebv)
        sigma = 0.5 * true_z
        params_to_fit = ['z', 't0', 'x0', 'x1', 'c']
        guess_z_value = True
        zlims = (true_z - sigma, true_z + sigma)
        bounds = {'z': zlims, 'x0': (0, 0.1), 'x1': (-5, 50), 'c': (-1, 5), 't0': (0, 100)}
    elif redshift == 'none':
        this_ia_model.set(mwebv=true_mwebv)
        params_to_fit = ['z', 't0', 'x0', 'x1', 'c']
        guess_z_value = True
        bounds = {'z': (0, 0.2), 'x0': (0, 0.1), 'x1': (-5, 50), 'c': (-1, 5), 't0': (0, 100)}
    else:
        raise ValueError("redshift parameter must be 'spec', 'phot', or 'none'")

    result, fitted_model = sncosmo.fit_lc(lcs[0], this_ia_model,
        params_to_fit, minsnr=5., guess_z=guess_z_value, bounds=bounds)

    return result, fitted_model


def fit_ibc_w_dust(true_z, lcs, true_av=None, true_mwebv=None, texp_guess=None, mej=3, fni=0.05, vej=15000, redshift='spec'):
    if texp_guess is None:
        texp_guess = 0

    arnett_source = ArnettSource(times, wavelengths, params=[true_z, texp_guess, mej, fni, vej])

    dust = sncosmo.CCM89Dust()
    R_V = 3.1
    if true_mwebv is None:
        if true_av is None:
            raise ValueError("Either true_av or true_mwebv must be provided")
        true_mwebv = true_av / R_V

    arnett_model = sncosmo.Model(source=arnett_source,
                                 effects=[dust],
                                 effect_names=['mw'],
                                 effect_frames=['obs'])

    arnett_model.set(mwebv=true_mwebv)

    if redshift == 'spec':
        arnett_model.set(redshift=true_z)
        params_to_fit = ['texp', 'mej', 'fni', 'vej']
        bounds = {'texp': (-30, 30), 'mej': (0.01, 30), 'fni': (0.01, 50), 'vej': (2000, 40000)}
    elif redshift == 'phot':
        sigma = 0.5 * true_z
        zlims = (true_z - sigma, true_z + sigma)
        params_to_fit = ['redshift', 'texp', 'mej', 'fni', 'vej']
        bounds = {'redshift': zlims, 'texp': (-30, 30), 'mej': (0.01, 30), 'fni': (0.01, 50), 'vej': (2000, 40000)}
    elif redshift == 'none':
        zlims = (0, 0.2)
        params_to_fit = ['redshift', 'texp', 'mej', 'fni', 'vej']
        bounds = {'redshift': zlims, 'texp': (-30, 30), 'mej': (0.01, 30), 'fni': (0.01, 50), 'vej': (2000, 40000)}
    else:
        raise ValueError("redshift parameter must be 'spec', 'phot', or 'none'")
    try:
        result, fitted_model = sncosmo.fit_lc(lcs[0], arnett_model, params_to_fit,
                                          bounds=bounds,
                                          minsnr=5.0, guess_t0=True, guess_z=False, guess_amplitude=False)
    except Exception as e:
        print("Trying again...")
        arnett_source = ArnettSource(times, wavelengths, params=[true_z, 10, 0.5, 0.1, 20000])
        arnett_model = sncosmo.Model(source=arnett_source,
                                     effects=[dust],
                                     effect_names=['mw'],
                                     effect_frames=['obs'])
        arnett_model.set(mwebv=true_mwebv)
        if redshift == 'spec':
            arnett_model.set(redshift=true_z)
        result, fitted_model = sncosmo.fit_lc(lcs[0], arnett_model, params_to_fit,
                                            bounds=bounds,
                                            minsnr=5.0, guess_t0=True, guess_z=False, guess_amplitude=False)
    return result, fitted_model


# ---------- real_SN class (cell 28) ----------
class real_SN:
    ia_keys = ['z', 't0', 'x0', 'x1', 'c', 'chisq']
    ibc_keys = ['z', 'texp', 'mej', 'fni', 'vej', 'chisq']

    def __init__(self, true_class, true_redshift, lc=None, name=None):
        self.true_class = true_class
        self.true_redshift = true_redshift
        self.lc = lc
        self.name = name

    def fit_as_ia(self, results_dict):
        self.ia_fit = results_dict

    def fit_as_ibc(self, results_dict):
        self.ibc_fit = results_dict


# ---------- Ibc fitting loop (cell 29) ----------
warnings.filterwarnings("ignore")

sn_objects = []
failed_files = []
skipped_files = []

for filename in redshift_dict.keys():
    try:
        lc_table = read_ibc_csv(filename)
        if lc_table is None:
            print(f"Skipped {filename}: insufficient points in g or r band")
            skipped_files.append(filename)
            continue

        this_redshift = redshift_dict[filename]
        this_av = av_dict[filename]

        parts = filename.split('_')
        ztfid = parts[0]
        true_class = parts[1].replace('-', ' ')

        ia_result, ia_fitted_model = fit_ia_w_dust(this_redshift, [lc_table], true_av=this_av, redshift=Z_TYPE)
        ibc_result, ibc_fitted_model = fit_ibc_w_dust(this_redshift, [lc_table], true_av=this_av, redshift=Z_TYPE)

        sn_obj = real_SN(true_class, this_redshift, lc=lc_table, name=ztfid)
        sn_obj.fit_as_ia(sncosmo.flatten_result(ia_result))
        sn_obj.fit_as_ibc(sncosmo.flatten_result(ibc_result))
        sn_objects.append(sn_obj)

        print(f"Successfully fit {filename}")

    except Exception as e:
        print(f"Failed to fit {filename}: {e}")
        failed_files.append((filename, str(e)))

print(f"\nSuccessfully fit {len(sn_objects)} out of {len(redshift_dict)} light curves")
print(f"Skipped (insufficient data): {len(skipped_files)}")
print(f"Failed: {len(failed_files)}")

ibc_failed_files = list(failed_files)
ibc_skipped_files = list(skipped_files)
with open(FAILED_IBC_TXT, "w") as f:
    f.write("# filename\terror\n")
    for fname, err in ibc_failed_files:
        f.write(f"{fname}\t{err}\n")

print(f"\nSaved failed Ibc fits to {FAILED_IBC_TXT} ({len(ibc_failed_files)} entries)")
print(f"Number files skipped due to insufficient data: {len(ibc_skipped_files)}")


# ---------- Ibc diagnostic plots (cells 30, 34, 36) ----------
plt.figure()
plt.hist([np.log10(sn.ibc_fit['chisq']) for sn in sn_objects], bins=20, alpha=0.5, label='Ibc fits')
plt.hist([np.log10(sn.ia_fit['chisq']) for sn in sn_objects], bins=20, alpha=0.5, label='Ia fits')
plt.xlabel("Chi-squared")
plt.ylabel("Number of SNe")
plt.legend()
plt.savefig(os.path.join(FIG_DIR, "ibc_chisq_hist.pdf"), bbox_inches='tight')
plt.close()

if len(sn_objects) > 1:
    ia_param_names = sn_objects[1].ia_keys[1:]
    ibc_param_names = sn_objects[1].ibc_keys[1:]

    fig, axes = plt.subplots(2, len(ibc_param_names), figsize=(20, 8))
    fig.suptitle("Histograms of Fitted Parameters for True Ibc SNe")

    for i, param in enumerate(ia_param_names):
        values = [sn.ia_fit[param] for sn in sn_objects if sn.ia_fit[param] is not None]
        axes[0, i].hist(values, bins=20, color='tab:blue', alpha=0.7)
        axes[0, i].set_title(f"Ia: {param}")
        axes[0, i].set_xlabel(param)
        axes[0, i].set_ylabel("Count")

    for i, param in enumerate(ibc_param_names):
        values = [sn.ibc_fit[param] for sn in sn_objects if sn.ibc_fit[param] is not None]
        if param == 'vej':
            values = [np.log10(v) for v in values if v > 0]
            axes[1, i].hist(values, bins=20, color='tab:orange', alpha=0.7)
            axes[1, i].set_title(f"Ibc: log10({param})")
            axes[1, i].set_xlabel(f"log10({param})")
        else:
            axes[1, i].hist(values, bins=20, color='tab:orange', alpha=0.7)
            axes[1, i].set_title(f"Ibc: {param}")
            axes[1, i].set_xlabel(param)
        axes[1, i].set_ylabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIG_DIR, "ibc_param_hists.pdf"), bbox_inches='tight')
    plt.close()

plt.figure()
param = 'fni'
plt.hist(np.log10(np.array([sn.ibc_fit[param] for sn in sn_objects if sn.ibc_fit[param] is not None])))
plt.xlabel(f"log10({param})")
plt.ylabel("Count")
plt.savefig(os.path.join(FIG_DIR, "ibc_log_fni_hist.pdf"), bbox_inches='tight')
plt.close()


# ---------- Ia metadata (cell 41) ----------
transient_table = pd.read_csv("ZTFBTS/ZTFBTS_TransientTable.csv")

ia_mask = transient_table['type'].str.fullmatch('SN Ia')
ia_transients = transient_table[ia_mask]
print(f"Found {len(ia_transients)} Type Ia supernovae in transient table")
print(f"Subtypes: {ia_transients['type'].value_counts().to_dict()}")

ia_data = pd.read_csv("ia_data.csv")

num_pec = ia_data['type'].str.contains('pec', na=False).sum()
print(f"Found {num_pec} entries in ia_data.csv with 'pec' in type")

ia_lookup = {}
for _, row in ia_data.iterrows():
    try:
        if "pec" in row['type']:
            continue
        redshift = float(row['redshift'])
        av = float(row['A_V'])
        ia_lookup[row['ZTFID']] = {'redshift': redshift, 'A_V': av}
    except (ValueError, TypeError):
        pass

print(f"\nFound {len(ia_lookup)} entries in ia_data.csv with valid redshift and A_V")

lc_dir = "ZTFBTS/light-curves"
available_lcs = set([f.replace('.csv', '') for f in os.listdir(lc_dir) if f.endswith('.csv')])
print(f"Found {len(available_lcs)} light curve files in {lc_dir}")

ia_ztfids = set(ia_transients['ZTFID'])
ia_with_data = ia_ztfids & set(ia_lookup.keys()) & available_lcs

print(f"\nMatched {len(ia_with_data)} Type Ia SNe with redshift, A_V, and light curve data")


# ---------- Ia reader and valid-list (cell 43) ----------
def read_ia_lc(ztfid, min_points_per_band=5):
    filepath = f"ZTFBTS/light-curves/{ztfid}.csv"
    df = pd.read_csv(filepath)

    df = df[~df['band'].isin(['i', 'I'])]

    g_count = (df['band'] == 'g').sum()
    r_count = df['band'].isin(['r', 'R']).sum()
    if g_count < min_points_per_band or r_count < min_points_per_band:
        return None

    tab = Table.from_pandas(df)

    tab['flux'] = mag_to_bandflux(tab['mag'])
    tab['fluxerr'] = magerr_to_fluxerr(tab['mag'], tab['magerr'])
    tab['zp'] = ZP * np.ones_like(tab['flux'])
    tab['zpsys'] = np.array([ZPSYS] * len(tab), dtype=str)

    band_map = {'g': 'ztfg', 'r': 'ztfr', 'R': 'ztfr'}
    tab['band'] = np.array([band_map.get(b, 'ztf' + b.lower()) for b in tab['band']], dtype=str)

    tab['time'] = tab['time'] - np.min(tab['time'])

    return tab

ia_redshift_dict = {ztfid: ia_lookup[ztfid]['redshift'] for ztfid in ia_with_data}
ia_av_dict = {ztfid: ia_lookup[ztfid]['A_V'] for ztfid in ia_with_data}

ia_subtype_dict = {}
for ztfid in ia_with_data:
    subtype = ia_transients[ia_transients['ZTFID'] == ztfid]['type'].values[0]
    ia_subtype_dict[ztfid] = subtype

valid_ia_ztfids = []
skipped_count = 0
for ztfid in ia_with_data:
    lc = read_ia_lc(ztfid)
    if lc is not None:
        valid_ia_ztfids.append(ztfid)
    else:
        skipped_count += 1

print(f"Created dictionaries for {len(ia_redshift_dict)} Type Ia SNe")
print(f"After filtering for >=5 points per band (g, r only; i-band dropped): {len(valid_ia_ztfids)} valid, {skipped_count} skipped")


# ---------- Redshift distribution plot (cell 57) ----------
plt.figure()
plt.hist(list(ia_redshift_dict.values()), density=True, label="Ia redshifts", alpha=0.7)
plt.hist(list(redshift_dict.values()), density=True, label="Ibc redshifts", alpha=0.7)
plt.legend()
plt.xlabel("Redshift")
plt.ylabel("Density")
plt.savefig(os.path.join(FIG_DIR, "redshift_distributions.pdf"), bbox_inches='tight')
plt.close()


# ---------- Ia fitting loop (cell 58) ----------
sn_ia_objects = []
failed_files = []

for ztfid in valid_ia_ztfids[:2500]:
    try:
        lc_table = read_ia_lc(ztfid)

        this_redshift = ia_redshift_dict[ztfid]
        this_av = ia_av_dict[ztfid]

        true_class = ia_subtype_dict[ztfid]

        ia_result, ia_fitted_model = fit_ia_w_dust(this_redshift, [lc_table], true_av=this_av, redshift=Z_TYPE)
        ibc_result, ibc_fitted_model = fit_ibc_w_dust(this_redshift, [lc_table], true_av=this_av, redshift=Z_TYPE)

        sn_obj = real_SN(true_class, this_redshift, lc=lc_table, name=ztfid)
        sn_obj.fit_as_ia(sncosmo.flatten_result(ia_result))
        sn_obj.fit_as_ibc(sncosmo.flatten_result(ibc_result))
        sn_ia_objects.append(sn_obj)

        print(f"Successfully fit {ztfid}")

    except Exception as e:
        print(f"Failed to fit {ztfid}: {e}")
        failed_files.append((ztfid, str(e)))

print(f"\nSuccessfully fit {len(sn_ia_objects)} out of {len(valid_ia_ztfids)} light curves")
print(f"Failed: {len(failed_files)}")

ia_failed_files = list(failed_files)
with open(FAILED_IA_TXT, "w") as f:
    f.write("# ztfid\terror\n")
    for ztfid, err in ia_failed_files:
        f.write(f"{ztfid}\t{err}\n")
print(f"\nSaved failed Ia fits to {FAILED_IA_TXT} ({len(ia_failed_files)} entries)")


# ---------- Combine and save (cells 61, 66) ----------
all_sn_objects = sn_objects + sn_ia_objects

np.savez(
    OBJECTS_NPZ,
    ibc=[sn for sn in all_sn_objects if sn.true_class != 'SN Ia'],
    ia=[sn for sn in all_sn_objects if sn.true_class == 'SN Ia'],
)
print(f"\nSaved {len(all_sn_objects)} fitted SN objects to {OBJECTS_NPZ}")
