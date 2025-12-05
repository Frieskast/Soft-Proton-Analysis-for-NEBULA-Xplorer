import os
import re
import glob
import math
import argparse
import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator

# apply style once
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "figure.labelsize": 18,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

# color palette (use provided colors / related)
COLOR_ORBIT = {
    "45-deg": "#4169E1",   # royal blue
    "98-deg": "#DC143C",   # crimson red (use red/blue combination)
}

COLOR_MIRROR = {
    "SP":  "#4169E1",   # royal blue
    "DCC": "#2ca02c",   # green
    "DPH": "#DC143C",   # crimson
}

MAJOR_TICKS = [100, 250, 500, 1000]
XTICKS_FULL = np.arange(50, 1051, 50)

# helper to produce nice orbit labels (uses unicode degree sign)
ORBIT_LABEL = {"45-deg": "45\u00b0", "98-deg": "98\u00b0"}
def orbit_label(orbit):
    return ORBIT_LABEL.get(orbit, orbit.replace("-deg", "\u00b0"))


# --- Correction factors for dumbbell opening ---
CORRECTION_FACTORS = {
    "withOpening": 0.92131,
    "withoutOpening": 0.92022
}


def parse_N_from_name(fname):
    """Parses the number of simulated events (N) from a filename."""
    m = re.search(r"_N(\d+)([kmKM])?", fname)
    if not m:
        return None
    n = int(m.group(1))
    suf = m.group(2)
    if not suf:
        return n
    suf = suf.lower()
    if suf == "k":
        return n * 1000
    if suf == "m":
        return n * 1000000
    return n

def parse_metadata_from_name(fname):
    """Parses mirror type, filter status, and energy from a filename."""
    base = os.path.basename(fname)
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else ""
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else ""
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    N = parse_N_from_name(base)
    # --- Detect hole status from the full path, not just the basename ---
    hole = "withOpening" if "withhole" in fname.lower() else ("withoutOpening" if "withouthole" in fname.lower() else "")
    return mirror, filter_flag, hole, energy, N

def _calculate_binomial_error(k, N):
    """Computes efficiency and binomial error, returns (eff, err)."""
    if N is None or N <= 0 or k is None:
        return 0.0, 0.0
    k = max(0, k)
    eff = k / float(N)
    # Binomial error: sqrt(p*(1-p)/N)
    err = math.sqrt(max(0, eff * (1.0 - eff) / N))
    return eff, err

def count_unique_events_in_tree(rootfile, treename, col="EventID"):
    """Counts unique events in a TTree to determine the number of hits."""
    try:
        tree = rootfile[treename]
        arr = tree.arrays([col], library="np")
        evts = np.asarray(arr[col]).astype(int)
        return int(np.unique(evts).size)
    except Exception:
        return 0

def calculate_flux_data(dirs, fluence_map, orbit_name):
    """
    Scans ROOT files and calculates all data points for flux calculation for a given orbit.
    Returns a list of dictionaries.
    """
    # --- Constants based on your request ---
    r_out_mm = 47.0  # Aperture radius in mm
    aperture_area_cm2 = math.pi * (r_out_mm / 10.0)**2

    x_aperture_map = { "DPH": 901.0, "DCC": 901.0, "SP": 851.0 }

    # --- Energy bin widths in MeV for fluence correction ---
    energy_bin_widths_MeV = {
        100: 0.108,
        250: 0.196,
        500: 0.353,
        1000: 0.707
    }

    entries = []
    for d in dirs:
        if not os.path.isdir(d): continue
        pattern = os.path.join(d, "**", "output_*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    mirror, filter_flag, hole, energy, N = parse_metadata_from_name(fpath)
                    
                    # --- Get correction factor based on hole status ---
                    correction_factor = CORRECTION_FACTORS.get(hole, 1.0) # Default to 1.0 if hole status is unknown

                    hits = 0
                    target_tree_name = "SmallDetSummary"
                    if target_tree_name in f:
                        hits = count_unique_events_in_tree(f, target_tree_name, col="EventID")
                    
                    if N is None or N <= 0 or hits is None: continue

                    funnelling_eff, funnelling_eff_err = _calculate_binomial_error(hits, N)
                    
                    # --- FIX: Define x_aperture_mm before using it ---
                    x_aperture_mm = x_aperture_map.get(mirror)
                    if x_aperture_mm is None: continue

                    mission_fluence_differential = fluence_map.get(energy, 0.0)
                    
                    # --- Correct the fluence by multiplying with the energy bin width ---
                    bin_width_MeV = energy_bin_widths_MeV.get(energy, 0.0)
                    mission_fluence_integral = mission_fluence_differential * bin_width_MeV

                    theta_max = math.atan(r_out_mm / x_aperture_mm)
                    a_eff_cm2 = aperture_area_cm2 * 0.25 * (math.sin(theta_max)**2)
                    
                    # --- FIX: Use the corrected integral fluence in calculations ---
                    total_particles_on_detector = mission_fluence_integral * a_eff_cm2 * funnelling_eff
                    total_particles_err = mission_fluence_integral * a_eff_cm2 * funnelling_eff_err

                    # --- Apply correction factor ---
                    total_particles_on_detector *= correction_factor
                    total_particles_err *= correction_factor

                    entries.append({
                        "basename": os.path.basename(fpath), "orbit": orbit_name, "mirror": mirror,
                        "filter": filter_flag, "hole": hole, "energy_keV": energy,
                        "funnelling_eff": funnelling_eff, "funnelling_eff_err": funnelling_eff_err,
                        "geom_factor_cm2": a_eff_cm2, # Renaming for clarity
                        "total_particles": total_particles_on_detector,
                        "total_particles_err": total_particles_err
                    })
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
    return entries

def plot_results(df):
    """Legacy single-plot function (kept for reference)."""
    if df.empty:
        print("DataFrame is empty, cannot generate plots.")
        return
    # kept minimal to avoid accidental use
    print("Use the new comparison plot functions: plot_filter_comparison, plot_hole_comparison, plot_concentrator_comparison")

# --- comparison plots requested by user ---

def _ensure_figures_dir(dirname="figures/flux_analysis"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def _prepare_df(df):
    d = df.copy()
    if 'orbit' not in d.columns:
        d['orbit'] = '45-deg'
    d = d[d['energy_keV'].notna() & (d['energy_keV'] != 50)].copy()
    return d

def _normalize_series(s):
    mx = s.max() if len(s) else 0
    return s / mx if mx and mx > 0 else s * 0

# --- Ensure a minimum visible y-error when plotting so tiny errors are readable ---
def _ensure_min_yerr(y, yerr, min_frac=1e-3, min_abs=None):
    """
    Enforce a minimum y-error for plotting.
    - y, yerr: arrays (or lists) of same length
    - min_frac: minimum fraction of |y| to use as error (default 0.1%)
    - min_abs: optional absolute minimum error (overrides computed floor if provided)
    Returns adjusted yerr array.

    NOTE: If min_frac <= 0 and min_abs is None, return yerr unchanged (no enforced floor).
    """
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    # replace NaNs / inf with zero so np.maximum works
    yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)

    # If caller explicitly disabled min_frac (<=0) and didn't provide min_abs, return raw errors.
    if (min_frac is None or min_frac <= 0.0) and (min_abs is None):
        return yerr

    if min_abs is None:
        maxy = np.nanmax(np.abs(y)) if y.size else 0.0
        min_abs = maxy * (min_frac if min_frac is not None else 0.0)
    # elementwise minimum based on value magnitude, but always at least min_abs
    min_per_point = np.maximum((min_frac if min_frac is not None else 0.0) * np.abs(y), min_abs)
    return np.maximum(yerr, min_per_point)

def compute_percent_reduction_by_group(dfp, group_col, high_value, low_value):
    """Compute percent reduction (high - low)/high *100 per orbit and energy.
    group_col: 'filter' or 'hole'
    high_value: baseline value name (e.g. 'filterOff' or 'withoutHole')
    low_value: compared value name (e.g. 'filterOn' or 'withHole')
    Returns dict orbit -> (energies_array, percent_array)
    """
    # Aggregate sums for values and squared errors for propagation
    agg = dfp.groupby(['orbit', group_col, 'energy_keV'], as_index=False).agg(
        total_particles=('total_particles', 'sum'),
        total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
    )
    agg['total_particles_err'] = np.sqrt(agg['total_particles_err_sq'])

    if agg.empty:
        return {}
    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', group_col], values='total_particles')
    orbits = sorted(agg['orbit'].unique())
    result = {}
    for orbit in orbits:
        col_high = (orbit, high_value)
        col_low = (orbit, low_value)
        s_high = pivot[col_high].dropna() if col_high in pivot.columns else pd.Series(dtype=float)
        s_low  = pivot[col_low].dropna()  if col_low  in pivot.columns else pd.Series(dtype=float)
        energies = sorted(set(s_high.index).union(s_low.index))
        if not energies:
            result[orbit] = (np.array([]), np.array([]))
            continue
        s_high_re = s_high.reindex(energies).fillna(0.0)
        s_low_re  = s_low.reindex(energies).fillna(0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            # --- Calculate INCREASE for 'hole', REDUCTION for others ---
            if group_col == 'hole':
                # Increase = (with - without) / without * 100
                percent = (s_low_re - s_high_re) / s_high_re * 100.0
            else:
                # Reduction = (off - on) / off * 100
                percent = (s_high_re - s_low_re) / s_high_re * 100.0
        percent = percent.replace([np.inf, -np.inf], np.nan).values
        result[orbit] = (np.array(energies), percent)
    return result

def plot_filter_comparison(df, cl_label, figures_dir=None, reduction_by_orbit=None):
    """Flux vs energy for filter on/off with reduction plotted underneath (shared x)."""
    figures_dir = _ensure_figures_dir(figures_dir or "/home/frisoe/Desktop/Thesis/figures/flux_analysis/")
    dfp = _prepare_df(df)

    agg = dfp.groupby(['orbit', 'filter', 'energy_keV'], as_index=False).agg(
        total_particles=('total_particles', 'sum'),
        total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
    )
    agg['total_particles_err'] = np.sqrt(agg['total_particles_err_sq'])
    if agg.empty:
        print("No data for filter comparison.")
        return

    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', 'filter'], values='total_particles')
    pivot_err = agg.pivot_table(index='energy_keV', columns=['orbit', 'filter'], values='total_particles_err')
    orbits = sorted(agg['orbit'].unique())

    if reduction_by_orbit is None:
        reduction_by_orbit = compute_percent_reduction_by_group(dfp, 'filter', 'filterOff', 'filterOn')

    # create two stacked axes: top = flux, bottom = reduction (%)
    fig, (ax_flux, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                          gridspec_kw={'height_ratios': [3, 1]})

    # --- FINAL CORRECTED TICK SETUP ---
    # xticks = np.arange(50, 1051, 50)
    # define the major ticks we want to show
    # major_ticks = [100, 250, 500, 1000]

    # 1. Apply basic styling to ALL axes and set the SAME major ticks
    for ax in [ax_flux, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    # 2. Configure the shared X-AXIS properties ONLY on the BOTTOM plot
    # set labels that match the major tick locations
    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_red.yaxis.set_minor_locator(MultipleLocator(10))
    ax_flux.set_xticklabels([])

    ax_flux.set_yscale('log')
    # --- FIX: Adjust y-axis limits depending on confidence level ---
    # For CL50 and CL75 use a wider dynamic range (1e7..1e11), otherwise keep default

    # ax_flux.set_ylim(1e6, 1e10)(1e7, 1e11)
    # if cl_label in ('CL50', 'CL75'):
    #     # ax_flux.set_ylim(1e6, 1e10)(1e7, 1e11)
    # else:
    #     # ax_flux.set_ylim(1e6, 1e10)(1e8, 1e10)
    # --- Move x-axis label to the bottom ---
    fig.supxlabel('Proton energy [keV]')
    ax_flux.set_ylabel('Mission Lifetime Proton Fluence [#]')
    # ax_flux.set_title(f'Thermal Optical Filter Comparison ({cl_label})')

    scatter_s = 160
    flux_lw = 2.5
    reduction_marker = 'D'
    reduction_color = "#2ca02c"
    reduction_lw = 2.5
    reduction_ms = 8
    linestyles = {"filterOn": "-", "filterOff": "--"}
    color_map = COLOR_ORBIT

    # plot fluxes (no twin axis)
    for orbit in orbits:
        color = color_map.get(orbit, "#000000")
        col_on = (orbit, 'filterOn')
        col_off = (orbit, 'filterOff')

        s_on = pivot[col_on].dropna() if col_on in pivot.columns else pd.Series(dtype=float)
        s_off = pivot[col_off].dropna() if col_off in pivot.columns else pd.Series(dtype=float)
        err_on = pivot_err[col_on].dropna() if col_on in pivot_err.columns else pd.Series(dtype=float)
        err_off = pivot_err[col_off].dropna() if col_off in pivot_err.columns else pd.Series(dtype=float)

        energies = sorted(set(s_on.index).union(s_off.index))
        if not energies:
            continue

        s_on_re = s_on.reindex(energies).fillna(0.0)
        s_off_re = s_off.reindex(energies).fillna(0.0)
        err_on_re = err_on.reindex(energies).fillna(0.0)
        err_off_re = err_off.reindex(energies).fillna(0.0)

        # --- Ensure tiny errors are visible ---
        err_off_plot = _ensure_min_yerr(s_off_re.values, err_off_re.values, min_frac=1e-3)
        err_on_plot  = _ensure_min_yerr(s_on_re.values,  err_on_re.values,  min_frac=1e-3)

        # --- plot error bands (shaded) instead of errorbars ---
        # Off
        lower_off = np.maximum(s_off_re.values - err_off_plot, s_off_re.values * 1e-8)
        upper_off = s_off_re.values + err_off_plot
        if s_off_re.size:
            ax_flux.fill_between(energies, lower_off, upper_off, color=color, alpha=0.18, zorder=1)
            ax_flux.plot(energies, s_off_re.values, linestyle=linestyles['filterOff'],
                         color=color, alpha=0.85, lw=flux_lw, zorder=2, label=f"{orbit_label(orbit)} Filter Off",
                         marker='o', markerfacecolor='none', markeredgecolor=color)

        # On
        lower_on = np.maximum(s_on_re.values - err_on_plot, s_on_re.values * 1e-8)
        upper_on = s_on_re.values + err_on_plot
        if s_on_re.size:
            ax_flux.fill_between(energies, lower_on, upper_on, color=color, alpha=0.18, zorder=2)
            ax_flux.plot(energies, s_on_re.values, linestyle=linestyles['filterOn'],
                         color=color, alpha=0.95, lw=flux_lw, zorder=3, label=f"{orbit_label(orbit)} Filter On",
                         marker='o', markerfacecolor=color, markeredgecolor='k')

    # compute and plot combined (average) reduction on bottom axis
    if reduction_by_orbit:
        dfs = [pd.DataFrame({'energy': e, f'red_{o}': p}) for o, (e, p) in reduction_by_orbit.items() if e.size > 0]
        if len(dfs) > 1:
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on='energy', how='outer')
            red_cols = [c for c in merged_df.columns if c.startswith('red_')]
            merged_df['avg_reduction'] = merged_df[red_cols].mean(axis=1)
            ax_red.plot(merged_df['energy'], merged_df['avg_reduction'],
                        linestyle='-', marker=reduction_marker, color=reduction_color,
                        markersize=reduction_ms, linewidth=reduction_lw, alpha=0.95, label='Average Reduction', zorder=5)
        elif dfs:
            ax_red.plot(dfs[0]['energy'], dfs[0].iloc[:,1],
                        linestyle='-', marker=reduction_marker, color=reduction_color,
                        markersize=reduction_ms, linewidth=reduction_lw, alpha=0.95, label='Average Reduction', zorder=5)

    ax_red.set_ylabel('Reduction (%)')
    ax_red.set_ylim(-25, 75)

    ax_flux.legend(loc='upper right')
    ax_flux.legend(title='Orbit / Filter')
    ax_red.legend(loc='upper right')

    plt.tight_layout()
    # save figure
    try:
        os.makedirs(figures_dir, exist_ok=True)
        fname_base = os.path.join(figures_dir, f"filter_comparison_{cl_label}")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to save filter comparison plot: {e}")
    #plt.show()
    plt.close()

def plot_hole_comparison(df, cl_label, figures_dir=None, reduction_by_orbit=None):
    """Flux vs energy comparing withHole / withoutHole with reduction underneath."""
    figures_dir = _ensure_figures_dir(figures_dir or "/home/frisoe/Desktop/Thesis/figures/flux_analysis/")
    dfp = _prepare_df(df)

    agg = dfp.groupby(['orbit', 'hole', 'energy_keV'], as_index=False).agg(
        total_particles=('total_particles', 'sum'),
        total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
    )
    agg['total_particles_err'] = np.sqrt(agg['total_particles_err_sq'])
    if agg.empty:
        print("No data for hole comparison.")
        return

    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', 'hole'], values='total_particles')
    pivot_err = agg.pivot_table(index='energy_keV', columns=['orbit', 'hole'], values='total_particles_err')
    orbits = sorted(agg['orbit'].unique())

    # compute hole-based reduction (without relative to with)
    if reduction_by_orbit is None:
        # --- Correct the arguments to calculate increase relative to 'withoutOpening' ---
        reduction_by_orbit = compute_percent_reduction_by_group(dfp, 'hole', 'withoutOpening', 'withOpening')

    # stacked plot: flux (top) and reduction (bottom)
    fig, (ax_flux, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                          gridspec_kw={'height_ratios': [3, 1]})

    # --- FINAL CORRECTED TICK SETUP ---
    # xticks = np.arange(50, 1051, 50)
    # define the major ticks we want to show
    # major_ticks = [100, 250, 500, 1000]

    # 1. Apply basic styling to ALL axes and set the SAME major ticks
    for ax in [ax_flux, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    # 2. Configure the shared X-AXIS properties ONLY on the BOTTOM plot
    # set labels that match the major tick locations
    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_red.yaxis.set_major_locator(MultipleLocator(5))
    ax_red.yaxis.set_minor_locator(MultipleLocator(1))
    ax_flux.set_xticklabels([])

    # --- FIX: Remove energy labels from the main flux plot ---
    ax_flux.set_xticklabels([])

    ax_flux.set_yscale('log')
    # --- FIX: Apply same y-axis limits as other plots ---

    # ax_flux.set_ylim(1e6, 1e10)(1e7, 1e11)
      # Use wider y-range for lower confidence levels to capture signal spread
    # if cl_label in ('CL50', 'CL75'):
    #     # ax_flux.set_ylim(1e6, 1e10)(1e7, 1e11)
    # else:
    #     # ax_flux.set_ylim(1e6, 1e10)(1e8, 1e10)

    # --- Move x-axis label to the bottom ---
    fig.supxlabel('Proton energy [keV]')
    ax_flux.set_ylabel('Mission Lifetime Proton Fluence [#]')
    # --- TITLE CHANGE requested ---
    # ax_flux.set_title(f'Dumbbell Opening Comparison ({cl_label})')

    # --- FIX: Ensure major tick at 5% intervals on reduction plot ---
    ax_red.yaxis.set_major_locator(MultipleLocator(5))

    scatter_s = 160
    flux_lw = 2.5
    hole_marker = 'o'
    reduction_marker = 'D'
    reduction_color = "#2ca02c"
    reduction_lw = 2.5
    reduction_ms = 8
    flux_linestyles = {"withOpening": "-", "withoutOpening": "--"}

    for orbit in orbits:
        color = COLOR_ORBIT.get(orbit, "#000000")
        col_with = (orbit, 'withOpening')
        col_without = (orbit, 'withoutOpening')

        s_with = pivot[col_with].dropna() if col_with in pivot.columns else pd.Series(dtype=float)
        s_without = pivot[col_without].dropna() if col_without in pivot.columns else pd.Series(dtype=float)
        err_with = pivot_err[col_with].dropna() if col_with in pivot_err.columns else pd.Series(dtype=float)
        err_without = pivot_err[col_without].dropna() if col_without in pivot_err.columns else pd.Series(dtype=float)

        energies = sorted(set(s_with.index).union(s_without.index))
        if not energies:
            continue

        s_with_re = s_with.reindex(energies).fillna(0.0)
        s_without_re = s_without.reindex(energies).fillna(0.0)
        err_with_re = err_with.reindex(energies).fillna(0.0)
        err_without_re = err_without.reindex(energies).fillna(0.0)

        # --- Ensure tiny errors are visible ---
        err_with_plot = _ensure_min_yerr(s_with_re.values, err_with_re.values, min_frac=1e-3)
        err_without_plot = _ensure_min_yerr(s_without_re.values, err_without_re.values, min_frac=1e-3)

        # With Hole -> shaded band + line
        if not s_with_re.empty:
            lower_with = np.maximum(s_with_re.values - err_with_plot, s_with_re.values * 1e-8)
            upper_with = s_with_re.values + err_with_plot
            ax_flux.fill_between(energies, lower_with, upper_with, color=color, alpha=0.18, zorder=2)
            ax_flux.plot(energies, s_with_re.values, color=color, linestyle=flux_linestyles['withOpening'],
                         alpha=0.95, lw=flux_lw, zorder=3, label=f"{orbit_label(orbit)} With Opening",
                         marker=hole_marker, markerfacecolor=color, markeredgecolor='k')

        # Without Hole -> shaded band + line
        if not s_without_re.empty:
            lower_without = np.maximum(s_without_re.values - err_without_plot, s_without_re.values * 1e-8)
            upper_without = s_without_re.values + err_without_plot
            ax_flux.fill_between(energies, lower_without, upper_without, color=color, alpha=0.12, zorder=1)
            ax_flux.plot(energies, s_without_re.values, color=color, linestyle=flux_linestyles['withoutOpening'],
                         alpha=0.75, lw=flux_lw, zorder=2, label=f"{orbit_label(orbit)} Without Opening",
                         marker=hole_marker, markerfacecolor='none', markeredgecolor=color)

    # compute and plot combined (average) reduction on bottom axis
    if reduction_by_orbit:
        dfs = [pd.DataFrame({'energy': e, f'red_{o}': p}) for o, (e, p) in reduction_by_orbit.items() if e.size > 0]
        if len(dfs) > 1:
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on='energy', how='outer')
            red_cols = [c for c in merged_df.columns if c.startswith('red_')]
            merged_df['avg_reduction'] = merged_df[red_cols].mean(axis=1)
            ax_red.plot(merged_df['energy'], merged_df['avg_reduction'],
                        linestyle='-', marker=reduction_marker, color=reduction_color,
                        markersize=reduction_ms, linewidth=reduction_lw, alpha=0.95, label='Average Increase', zorder=5)
        elif dfs:
            ax_red.plot(dfs[0]['energy'], dfs[0].iloc[:,1],
                        linestyle='-', marker=reduction_marker, color=reduction_color,
                        markersize=reduction_ms, linewidth=reduction_lw, alpha=0.95, label='Average Increase', zorder=5)

    ax_red.set_ylabel('Increase (%)')
    # --- Set y-limit back to 0-20% for the increase plot ---
    ax_red.set_ylim(0, 20)

    ax_flux.legend(loc='upper right')
    ax_flux.legend(title='Orbit / Opening')
    ax_red.legend(loc='upper right')

    plt.tight_layout()
    # save figure
    try:
        os.makedirs(figures_dir, exist_ok=True)
        fname_base = os.path.join(figures_dir, f"hole_comparison_{cl_label}")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to save hole comparison plot: {e}")
    # #plt.show()
    plt.close()

def plot_orbit_comparison(df, cl_label, figures_dir=None):
    """
    Compares proton fluence between 45-deg and 98-deg orbits for each mirror type on a single plot.
    - Top plot: Absolute fluence for a baseline configuration (filterOn, withHole).
    - Bottom plot: Percent reduction of 98-deg relative to 45-deg for each mirror.
    """
    figures_dir = _ensure_figures_dir(figures_dir or "/home/frisoe/Desktop/Thesis/figures/flux_analysis/")
    dfp = _prepare_df(df)
    
    # --- Define baseline configurations for orbit-to-orbit comparison ---
    baseline_mirrors = ['SP', 'DCC', 'DPH']
    baseline_filter = 'filterOn'
    baseline_hole = 'withOpening'

    dfp = dfp[(dfp['mirror'].isin(baseline_mirrors)) & 
             (dfp['filter'] == baseline_filter) & 
             (dfp['hole'] == baseline_hole)].copy()

    if dfp.empty:
        print(f"No data for orbit comparison with baseline config (Mirrors: {baseline_mirrors}, Filter: {baseline_filter}, Hole: {baseline_hole}).")
        return

    # Aggregate data for the selected configuration, now including mirror
    agg = dfp.groupby(['mirror', 'orbit', 'energy_keV'], as_index=False).agg(
        total_particles=('total_particles', 'sum'),
        total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
    )
    agg['total_particles_err'] = np.sqrt(agg['total_particles_err_sq'])
    
    pivot = agg.pivot_table(index='energy_keV', columns=['mirror', 'orbit'], values='total_particles')
    pivot_err = agg.pivot_table(index='energy_keV', columns=['mirror', 'orbit'], values='total_particles_err')

    # create two stacked axes: top = flux, bottom = reduction (%)
    fig, (ax_flux, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                          gridspec_kw={'height_ratios': [3, 1]})

    # --- Configure Axes (similar to other plots) ---
    for ax in [ax_flux, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_flux.set_xticklabels([])

    fig.supxlabel('Proton energy [keV]')
    ax_flux.set_ylabel('Mission Lifetime Proton Fluence [#]')
    # ax_flux.set_title(f'Orbit Comparison (Filter On, With Opening) ({cl_label})')

    flux_lw = 2.5
    linestyles = {"45-deg": "-", "98-deg": "--"}
    reduction_marker = 'D'
    # --- markers for orbits (match other plots) ---
    markers = {"45-deg": "o", "98-deg": "s"}

    # Use log scale and fixed y-limits requested
    ax_flux.set_yscale('log')
    
    # ax_flux.set_ylim(1e6, 1e10)(1e6, 1e10)

    # --- Plot Flux and Reduction Data ---
    # collect per-mirror reduction series to compute a single average reduction line
    reduction_dfs = []
    for mirror in baseline_mirrors:
        color = COLOR_MIRROR.get(mirror)

        # --- Plot Flux Data (Top Plot) ---
        for orbit in ['45-deg', '98-deg']:
            col_id = (mirror, orbit)
            if col_id not in pivot.columns:
                continue

            s = pivot[col_id].dropna()
            err = pivot_err[col_id].reindex(s.index).fillna(0)
            energies = s.index.values

            err_plot = _ensure_min_yerr(s.values, err.values, min_frac=1e-3)
            lower = np.maximum(s.values - err_plot, s.values * 1e-8)
            upper = s.values + err_plot

            ax_flux.fill_between(energies, lower, upper, color=color, alpha=0.12, zorder=1)
            ax_flux.plot(
                energies,
                s.values,
                color=color,
                linestyle=linestyles.get(orbit),
                alpha=0.95,
                lw=flux_lw,
                zorder=3,
                marker=markers.get(orbit),
                markerfacecolor=color,
                markeredgecolor='k',
                label=f"{mirror} / {orbit_label(orbit)}"
            )

        # --- Calculate and collect Reduction (Bottom Plot) ---
        s_98 = pivot.get((mirror, '98-deg'), pd.Series(dtype=float)).dropna()
        s_45 = pivot.get((mirror, '45-deg'), pd.Series(dtype=float)).dropna()

        if not s_98.empty and not s_45.empty:
            energies = sorted(set(s_98.index).union(s_45.index))
            s_98_re = s_98.reindex(energies).fillna(0)
            s_45_re = s_45.reindex(energies).fillna(0)

            with np.errstate(divide='ignore', invalid='ignore'):
                reduction = (s_98_re - s_45_re) / s_98_re * 100.0
            reduction = reduction.replace([np.inf, -np.inf], np.nan)
            reduction_dfs.append(pd.DataFrame({'energy': energies, f'red_{mirror}': reduction.values}))
 
    # merge per-mirror reductions and plot a single average reduction line
    if reduction_dfs:
        merged_red = reduction_dfs[0]
        for next_df in reduction_dfs[1:]:
            merged_red = pd.merge(merged_red, next_df, on='energy', how='outer')
        red_cols = [c for c in merged_red.columns if c.startswith('red_')]
        merged_red['avg_reduction'] = merged_red[red_cols].mean(axis=1, skipna=True)
        ax_red.plot(merged_red['energy'], merged_red['avg_reduction'],
                    linestyle='-', marker=reduction_marker, color='#2ca02c',
                    linewidth=flux_lw, label='Average Reduction (all mirrors)', zorder=5)

    ax_red.set_ylabel('Reduction (%)')
    ax_red.set_ylim(0, 100)
    ax_red.yaxis.set_major_locator(MultipleLocator(20))
    ax_red.yaxis.set_minor_locator(MultipleLocator(10))

    ax_flux.legend(loc='upper right', ncol=2)
    ax_flux.legend(title='Concentrator / Orbit')
    ax_red.legend(loc='upper right')

    plt.tight_layout()
    # save figure
    try:
        os.makedirs(figures_dir, exist_ok=True)
        fname_base = os.path.join(figures_dir, f"orbit_comparison_all-mirrors_{cl_label}")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to save orbit comparison plot: {e}")
    # #plt.show()
    plt.close()


def plot_concentrator_comparison(df, cl_label, figures_dir=None):
    """Absolute flux vs energy for SP / DCC / DPH with percent gain vs SP subplot.
    Percent gain = (mirror - SP) / SP * 100.
    """
    figures_dir = _ensure_figures_dir(figures_dir or "/home/frisoe/Desktop/Thesis/figures/flux_analysis/")
    dfp = _prepare_df(df)

    agg = dfp.groupby(['orbit', 'mirror', 'energy_keV'], as_index=False).agg(
        total_particles=('total_particles', 'sum'),
        total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
    )
    agg['total_particles_err'] = np.sqrt(agg['total_particles_err_sq'])
    if agg.empty:
        print("No data for concentrator comparison.")
        return

    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', 'mirror'], values='total_particles')
    pivot_err = agg.pivot_table(index='energy_keV', columns=['orbit', 'mirror'], values='total_particles_err')

    orbits = sorted(agg['orbit'].unique())
    mirrors = ['SP', 'DCC', 'DPH']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 10), sharex=False,
                                         gridspec_kw={'height_ratios': [3, 1]})

    # --- STANDARD X-TICK SETUP (major ticks and minor ticks) ---
    for a in (ax_top, ax_bot):
        a.set_xlim(50, 1050)
        a.set_xticks(MAJOR_TICKS)
        a.xaxis.set_minor_locator(MultipleLocator(50))
        a.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        a.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        a.set_axisbelow(True)
        a.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        a.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax_bot.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_bot.yaxis.set_minor_locator(MultipleLocator(10))
    ax_top.set_xticklabels([])

    fig.supxlabel('Proton energy [keV]')
    ax_top.set_ylabel('Mission Lifetime Proton Fluence [#]')
    # ax_top.set_title(f'Orbit Comparison (Filter On, With Opening) ({cl_label})')

    # --- Set the y-axis to a logarithmic scale ---
    ax_top.set_yscale('log')

    ax_bot.set_ylabel('Percentage gain vs SP (%)')
    ax_bot.axhline(0, color='k', linewidth=0.6, alpha=0.6)

    # unify sizes/lines to match the first two plots
    scatter_s = 160
    flux_lw = 2.5
    reduction_marker = 'D'
    reduction_ms = 8
    markers = {"45-deg": "o", "98-deg": "s"}
    linestyles = {"45-deg": "-", "98-deg": "--"}

    # Plot absolute flux on top plot
    for orbit in orbits:
        for mirror in mirrors:
            col_m = (orbit, mirror)
            s_m = pivot[col_m].dropna() if col_m in pivot.columns else pd.Series(dtype=float)
            err_m = pivot_err[col_m].dropna() if col_m in pivot_err.columns else pd.Series(dtype=float)
            if not s_m.empty:
                color = COLOR_MIRROR.get(mirror, COLOR_ORBIT.get(orbit, "#000000"))
                # --- Ensure tiny errors are visible ---
                err_plot = _ensure_min_yerr(s_m.values, err_m.reindex(s_m.index).fillna(0).values, min_frac=1e-3)

                # --- REPLACE: use shaded error band (fill_between) instead of errorbar ---
                lower = np.maximum(s_m.values - err_plot, s_m.values * 1e-8)
                upper = s_m.values + err_plot
                ax_top.fill_between(s_m.index.values, lower, upper, color=color, alpha=0.18, zorder=2)

                ax_top.plot(s_m.index.values, s_m.values, linestyle=linestyles.get(orbit,"-"),
                            color=color, alpha=0.9, lw=flux_lw, zorder=3,
                            marker=markers.get(orbit), markerfacecolor=color, markeredgecolor='k',
                            label=f"{orbit_label(orbit)} / {mirror}")

    # Calculate and plot combined percent gain on bottom plot
    for mirror in ['DCC', 'DPH']:
        gain_dfs = []
        for orbit in orbits:
            col_sp = (orbit, 'SP')
            col_m = (orbit, mirror)
            s_sp = pivot[col_sp].dropna() if col_sp in pivot.columns else pd.Series(dtype=float)
            s_m = pivot[col_m].dropna() if col_m in pivot.columns else pd.Series(dtype=float)
            
            energies = sorted(set(s_sp.index).union(s_m.index))
            if not energies or s_sp.empty:
                continue

            s_sp_re = s_sp.reindex(energies).fillna(0.0)
            s_m_re = s_m.reindex(energies).fillna(0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                percent_gain = (s_m_re - s_sp_re) / s_sp_re * 100.0
            gain_dfs.append(pd.DataFrame({'energy': energies, f'gain_{orbit_label(orbit)}': percent_gain.values}))

        if len(gain_dfs) > 1:
            merged_df = pd.merge(gain_dfs[0], gain_dfs[1], on='energy', how='outer')
            gain_cols = [c for c in merged_df.columns if c.startswith('gain_')]
            merged_df['avg_gain'] = merged_df[gain_cols].mean(axis=1)
            ax_bot.plot(merged_df['energy'], merged_df['avg_gain'], marker=reduction_marker,
                        color=COLOR_MIRROR.get(mirror), label=f"Average Gain {mirror} vs SP", markersize=reduction_ms, linewidth=flux_lw)
        elif gain_dfs:
            ax_bot.plot(gain_dfs[0]['energy'], gain_dfs[0].iloc[:,1], marker=reduction_marker,
                        color=COLOR_MIRROR.get(mirror), label=f"Average Gain {mirror} vs SP", markersize=reduction_ms, linewidth=flux_lw)

    ax_bot.set_ylim(0, 75)

    # --- FIX: Create legends after plotting artists ---
    ax_top.legend(loc='upper right')
    ax_top.legend(title='Orbit / Concentrator')
    ax_bot.legend(loc='lower right')

    plt.tight_layout()
    # save figure
    try:
        os.makedirs(figures_dir, exist_ok=True)
        fname_base = os.path.join(figures_dir, f"concentrator_comparison_{cl_label}")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"Failed to save concentrator comparison plot: {e}")
    # #plt.show()
    plt.close()

def plot_final_dcc_fluence(df, figures_dir=None):
    """
    Generates a dedicated plot for the final recommended configuration (DCC, filterOn, withOpening)
    and includes data for all available confidence levels.
    """
    figures_dir = _ensure_figures_dir(figures_dir or "/home/frisoe/Desktop/Thesis/figures/flux_analysis/")
    dfp = _prepare_df(df)

    # Filter for the specific configuration
    df_final = dfp[
        (dfp['mirror'] == 'DCC') &
        (dfp['filter'] == 'filterOn') &
        (dfp['hole'] == 'withOpening')
    ].copy()

    if df_final.empty:
        print("No data found for the final DCC configuration (DCC, filterOn, withOpening).")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # --- STANDARD X-TICK SETUP (major ticks and minor ticks) ---
    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])

    # --- Set titles and labels ---
    # ax.set_title('Proton Fluence for DCC (Filter On, With Opening)')
    ax.set_xlabel('Proton energy [keV]')
    ax.set_ylabel('Mission Lifetime Proton Fluence [#]')
    ax.set_yscale('log')

    # --- Plot data for each orbit and confidence level ---
    flux_lw = 2.5
    cl_linestyles = {'CL95': '-', 'CL75': '--', 'CL50': ':'}

    for (orbit, cl), group_df in df_final.groupby(['orbit', 'cl']):
        orbit_df = group_df.sort_values('energy_keV')
        if orbit_df.empty:
            continue

        energies = orbit_df['energy_keV'].values
        particles = orbit_df['total_particles'].values
        errors = orbit_df['total_particles_err'].values
        
        color = COLOR_ORBIT.get(orbit, 'black')
        linestyle = cl_linestyles.get(cl, '-')
        
        # --- Plot error band for ALL confidence levels ---
        err_plot = _ensure_min_yerr(particles, errors, min_frac=1e-3)
        lower = np.maximum(particles - err_plot, 1e-8 * particles)
        upper = particles + err_plot
        ax.fill_between(energies, lower, upper, color=color, alpha=0.18, zorder=1)

        # --- Use filled markers for CL95 and open markers for others ---
        mfc = color if cl == 'CL95' else 'none'
        mec = color
        ax.plot(energies, particles, color=color, linestyle=linestyle, lw=flux_lw, marker='o',
                markerfacecolor=mfc, markeredgecolor=mec,
                label=f'{orbit_label(orbit)} / {cl}', zorder=2)

    ax.legend(loc='upper right', ncol=2)
    ax.legend(title='Orbit / Confidence Level')
    plt.tight_layout()

    # --- Save the figure ---
    try:
        fname_base = os.path.join(figures_dir, "final_dcc_fluence_all_cls")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
        print(f"Saved final DCC plot to {fname_base}.pdf")
    except Exception as e:
        print(f"Failed to save final DCC fluence plot: {e}")
    #plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot detector fluence based on simulation efficiency.")
    parser.add_argument("--dirs", "-d", nargs="+", default=[
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ], help="Directories to scan for ROOT files.")
    parser.add_argument("--cl", choices=['CL50', 'CL75', 'CL95'], default='CL95', help="Confidence level for fluence.")
    args = parser.parse_args()

    # --- Print geometric factor calculations for verification ---
    print("\n" + "="*80)
    print("Verifying Geometric Factor Calculations")
    print("="*80)
    r_out_mm = 47.0
    aperture_area_cm2 = math.pi * (r_out_mm / 10.0)**2
    x_aperture_map = {"DPH": 901.0, "DCC": 901.0, "SP": 851.0}
    print(f"Aperture Radius (r_out): {r_out_mm} mm")
    print(f"Total Aperture Area: {aperture_area_cm2:.4f} cm^2")
    print("-" * 80)
    for mirror, x_aperture_mm in x_aperture_map.items():
        theta_max_rad = math.atan(r_out_mm / x_aperture_mm)
        theta_max_deg = math.degrees(theta_max_rad)
        a_eff_cm2 = aperture_area_cm2 * 0.25 * (math.sin(theta_max_rad)**2)
        print(f"For {mirror} (x_aperture = {x_aperture_mm} mm):")
        print(f"  - Theta_max: {theta_max_deg:.4f} degrees")
        print(f"  - Effective Area (a_eff): {a_eff_cm2:.4f} cm^2")
    print("="*80 + "\n")


    # --- Define Fluence Maps for Different Orbits ---
    fluence_maps_all = {
        'CL95': {
            "45-deg": {100: 2.69746044e11, 250: 2.05480469e11, 500: 1.41679924e11, 1000: 6.26134501e10},
            "98-deg-AP9": {100: 6.92936322e11, 250: 4.52721202e11, 500: 2.29025363e11, 1000: 5.24018357e10},
            "98-deg-SP": {100: 2.321100e12, 250: 5.304100e11, 500: 1.702800e11, 1000: 5.484100e10}
        },
        'CL75': {
            "45-deg": {100: 1.575964e11, 250: 1.220416e11, 500: 7.381142e10, 1000: 3.236780e10},
            "98-deg-AP9": {100: 3.113434e11, 250: 2.047505e11, 500: 1.081152e11, 1000: 2.757775e10},
            "98-deg-SP": {100: 3.747300e12, 250: 6.560000e11, 500: 1.171120e11, 1000: 4.482700e10}
        },
        'CL50': {
            "45-deg": {100: 9.373315e10, 250: 7.207972e10, 500: 5.209968e10, 1000: 2.478172e10},
            "98-deg-AP9": {100: 2.163474e11, 250: 1.382338e11, 500: 6.787237e10, 1000: 1.903805e10},
            "98-deg-SP": {100: 4.887900e12, 250: 7.113000e11, 500: 1.606100e11, 1000: 3.644600e10}
        }
    }

    # --- Select Fluence based on CL ---
    CONFIDENCE_LEVEL = args.cl
    print(f"Using {CONFIDENCE_LEVEL} confidence level for comparison plots.")
    
    # --- Calculate data for ALL confidence levels for the final plot ---
    all_entries_all_cls = []
    for cl_level, fluence_maps in fluence_maps_all.items():
        print(f"Calculating data for {cl_level}...")
        fluence_map_45_tot = fluence_maps["45-deg"]
        fluence_map_98_ap9 = fluence_maps["98-deg-AP9"]
        fluence_map_98_sp = fluence_maps["98-deg-SP"]
        fluence_map_98_tot = {energy: fluence_map_98_ap9.get(energy, 0) + fluence_map_98_sp.get(energy, 0) for energy in set(fluence_map_98_ap9) | set(fluence_map_98_sp)}

        data_45 = calculate_flux_data(args.dirs, fluence_map_45_tot, "45-deg")
        data_98 = calculate_flux_data(args.dirs, fluence_map_98_tot, "98-deg")
        
        current_cl_entries = data_45 + data_98
        for entry in current_cl_entries:
            entry['cl'] = cl_level # Add CL identifier
        all_entries_all_cls.extend(current_cl_entries)

    if not all_entries_all_cls:
        print("No valid files found; nothing to calculate or plot.")
    else:
        # Create a master DataFrame with all CLs
        df_all = pd.DataFrame(all_entries_all_cls)
        
        # Create a DataFrame for the user-selected CL for the comparison plots
        df_single_cl = df_all[df_all['cl'] == CONFIDENCE_LEVEL].copy()

        df_single_cl.sort_values(by=['orbit','mirror', 'hole', 'filter', 'energy_keV'], inplace=True)
        
        # --- START: Added print sections from _cu script ---
        print("\n" + "="*140)
        print(f"Mission Lifetime Flux Calculation ({CONFIDENCE_LEVEL})")
        print("="*140)
        header = f"{'Basename':<50} | {'Orbit':<7} | {'Mirror':<6} | {'Filter':<9} | {'Hole':<11} | {'Energy':>6} | {'Funnelling Eff':>14} | {'Total Particles':>15} | {'Error':>15}"
        print(header)
        print("-" * len(header))

        for _, e in df_single_cl.iterrows():
            print(f"{e['basename']:<50} | {e['orbit']:<7} | {e['mirror']:<6} | {e['filter']:<9} | {e['hole']:<11} | {int(e['energy_keV']):>6} | {e['funnelling_eff']:>14.4e} | {e['total_particles']:>15.3e} | {e['total_particles_err']:>15.3e}")
        
        print("\n" + "="*140)
        print(f"Total Fluence for DCC, filterOn, withOpening ({CONFIDENCE_LEVEL})")
        print("="*140)
        
        df_filtered = df_single_cl[(df_single_cl['mirror'] == 'DCC') & (df_single_cl['filter'] == 'filterOn') & (df_single_cl['hole'] == 'withOpening')].copy()
        
        df_grouped_all_orbits = df_filtered.groupby('energy_keV')[['total_particles', 'total_particles_err']].sum()
        
        if not df_grouped_all_orbits.empty:
            header = f"{'Energy':>6} | {'Total Particles':>15} | {'Error':>15}"
            print(header)
            print("-" * len(header))
            for energy, row in df_grouped_all_orbits.iterrows():
                print(f"{int(energy):>6} | {row['total_particles']:>15.3e} | {row['total_particles_err']:>15.3e}")
        else:
            print("No data found for DCC, filterOn, withOpening configuration.")

        for selected_orbit in df_filtered['orbit'].unique():
            print(f"\n...for orbit: {selected_orbit}")
            df_filtered_orbit = df_filtered[df_filtered['orbit'] == selected_orbit].copy()
            df_grouped_orbit = df_filtered_orbit.groupby('energy_keV')[['total_particles', 'total_particles_err']].sum()

            if not df_grouped_orbit.empty:
                header = f"{'Energy':>6} | {'Total Particles':>15} | {'Error':>15}"
                print(header)
                print("-" * len(header))
                for energy, row in df_grouped_orbit.iterrows():
                    print(f"{int(energy):>6} | {row['total_particles']:>15.3e} | {row['total_particles_err']:>15.3e}")
            else:
                print(f"No data found for DCC, filterOn, withOpening configuration in orbit {selected_orbit}.")
        
        mission_seconds = 5.0 * 365.25 * 24 * 3600
        df_dcc_all = df_all[
            (df_all['mirror'] == 'DCC') &
            (df_all['filter'] == 'filterOn') &
            (df_all['hole'] == 'withOpening')
        ].copy()

        print("\n" + "="*140)
        print("Background rate (per second) for DCC, filterOn, withOpening — per CL & orbit")
        print("="*140)
        if df_dcc_all.empty:
            print("No data available across confidence levels for DCC/filterOn/withOpening.")
        else:
            agg_rates = df_dcc_all.groupby(['cl', 'orbit', 'energy_keV'], as_index=False).agg(
                total_particles=('total_particles', 'sum'),
                total_particles_err_sq=('total_particles_err', lambda x: (x**2).sum())
            )
            agg_rates['total_particles_err'] = np.sqrt(agg_rates['total_particles_err_sq'])
            agg_rates['rate_per_s'] = agg_rates['total_particles'] / mission_seconds
            agg_rates['rate_err_per_s'] = agg_rates['total_particles_err'] / mission_seconds

            for cl_level in sorted(agg_rates['cl'].unique()):
                print(f"\nConfidence level: {cl_level}")
                for orbit in sorted(agg_rates[agg_rates['cl'] == cl_level]['orbit'].unique()):
                    sub = agg_rates[(agg_rates['cl'] == cl_level) & (agg_rates['orbit'] == orbit)].sort_values('energy_keV')
                    if sub.empty:
                        continue
                    hdr = f"{'Energy(keV)':>12} | {'Total Particles':>15} | {'Err':>12} | {'Rate [s^-1]':>13} | {'Rate Err [s^-1]':>15}"
                    print(f"\n... orbit: {orbit}")
                    print(hdr)
                    print("-" * len(hdr))
                    for _, r in sub.iterrows():
                        print(f"{int(r['energy_keV']):>12} | {r['total_particles']:>15.3e} | {r['total_particles_err']:>12.3e} | {r['rate_per_s']:>13.3e} | {r['rate_err_per_s']:>15.3e}")
        print("\n")
        # --- END: Added print sections from _cu script ---


        # --- Calculate and print delta fluence table ---
        print("\n" + "="*140)
        print(f"Delta Fluence Comparison Table at {CONFIDENCE_LEVEL}")
        print(f"Baseline: DCC, filterOn, withOpening (relative to each orbit)")
        print("="*140)

        # Group by each unique configuration and sum the total particles across all energies
        config_totals = df_single_cl.groupby(['orbit', 'mirror', 'filter', 'hole'])['total_particles'].sum().reset_index()

        # --- Calculate percentage change relative to the baseline WITHIN EACH ORBIT ---
        
        # Find the baseline fluence for each orbit
        baseline_45_deg = config_totals[
            (config_totals['orbit'] == '45-deg') & (config_totals['mirror'] == 'DCC') &
            (config_totals['filter'] == 'filterOn') & (config_totals['hole'] == 'withOpening')
        ]
        baseline_98_deg = config_totals[
            (config_totals['orbit'] == '98-deg') & (config_totals['mirror'] == 'DCC') &
            (config_totals['filter'] == 'filterOn') & (config_totals['hole'] == 'withOpening')
        ]

        if baseline_45_deg.empty or baseline_98_deg.empty:
            print("One or more baseline configurations not found. Cannot calculate full delta table.")
        else:
            fluence_45_base = baseline_45_deg['total_particles'].iloc[0]
            fluence_98_base = baseline_98_deg['total_particles'].iloc[0]

            # Define a function to apply the correct baseline
            def calculate_pct_change(row):
                if row['orbit'] == '45-deg':
                    if fluence_45_base > 0:
                        return ((row['total_particles'] - fluence_45_base) / fluence_45_base) * 100
                elif row['orbit'] == '98-deg':
                    if fluence_98_base > 0:
                        return ((row['total_particles'] - fluence_98_base) / fluence_98_base) * 100
                return 0.0

            config_totals['fluence_change_pct'] = config_totals.apply(calculate_pct_change, axis=1)
            
            # --- Sort by orbit, then by percentage change descending ---
            config_totals.sort_values(by=['orbit', 'fluence_change_pct'], ascending=[True, False], inplace=True)

            # Print the formatted table
            header = f"{'Orbit':<7} | {'Mirror':<6} | {'Filter':<9} | {'Hole':<14} | {'Total Fluence':>15} | {'Fluence Change (%)':>20}"
            print(header)
            print("-" * len(header))
            for _, row in config_totals.iterrows():
                # --- Conditionally format the percentage to remove '+' from zero ---
                pct_str = f"{row['fluence_change_pct']:>20.2f}" if abs(row['fluence_change_pct']) < 1e-9 else f"{row['fluence_change_pct']:>+20.2f}"
                print(f"{row['orbit']:<7} | {row['mirror']:<6} | {row['filter']:<9} | {row['hole']:<14} | {row['total_particles']:>15.3e} | {pct_str}")


        # --- Call the plotting functions to generate and save the plots ---
        print("\n" + "="*140)
        print("Generating and saving plots...")
        # --- FINAL PLOT: Dedicated DCC plot with all confidence levels ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # --- STANDARD X-TICK SETUP (major ticks and minor ticks) ---
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
        ax.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])

        # --- Set titles and labels ---
        ax.set_xlabel('Proton energy [keV]')
        ax.set_ylabel('Mission Lifetime Proton Fluence [#]')
        ax.set_yscale('log')

        # --- Plot data for each orbit and confidence level ---
        flux_lw = 2.5
        cl_linestyles = {'CL95': '-', 'CL75': '--', 'CL50': ':'}

        for (orbit, cl), group_df in df_all.groupby(['orbit', 'cl']):
            orbit_df = group_df.sort_values('energy_keV')
            if orbit_df.empty:
                continue

            energies = orbit_df['energy_keV'].values
            particles = orbit_df['total_particles'].values
            errors = orbit_df['total_particles_err'].values
            
            color = COLOR_ORBIT.get(orbit, 'black')
            linestyle = cl_linestyles.get(cl, '-')
            
            # --- Plot error band for ALL confidence levels ---
            err_plot = _ensure_min_yerr(particles, errors, min_frac=1e-3)
            lower = np.maximum(particles - err_plot, 1e-8 * particles)
            upper = particles + err_plot
            ax.fill_between(energies, lower, upper, color=color, alpha=0.18, zorder=1)

            # --- Use filled markers for CL95 and open markers for others ---
            mfc = color if cl == 'CL95' else 'none'
            mec = color
            ax.plot(energies, particles, color=color, linestyle=linestyle, lw=flux_lw, marker='o',
                    markerfacecolor=mfc, markeredgecolor=mec,
                    label=f'{orbit_label(orbit)} / {cl}', zorder=2)

        ax.legend(loc='upper right', ncol=2)
        ax.legend(title='Orbit / Confidence Level')
        plt.tight_layout()

        # --- Save the figure ---
        try:
            fname_base = os.path.join(figures_dir, "final_dcc_fluence_all_cls")
            fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{fname_base}.pdf", dpi=200, bbox_inches='tight')
            print(f"Saved final DCC plot to {fname_base}.pdf")
        except Exception as e:
            print(f"Failed to save final DCC fluence plot: {e}")
        #plt.show()
        plt.close()