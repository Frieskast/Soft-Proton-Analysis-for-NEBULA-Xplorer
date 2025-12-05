import os
import re
import glob
import math
import argparse
import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# --- Matplotlib Style (consistent with calculate_flux.py) ---
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

# --- Correction factors for dumbbell opening ---
CORRECTION_FACTORS = {
    "withOpening": 0.92131,
    "withoutOpening": 0.92022
}

COLOR_MIRROR = {
    "SP":  "#4169E1", "DCC": "#2ca02c", "DPH": "#DC143C",
}
# --- Define colors for orbits as requested ---
COLOR_ORBIT = {
    "45-deg": "#4169E1", # Blue
    "98-deg": "#DC143C", # Red
}

MAJOR_TICKS = [100, 250, 500, 1000]
XTICKS_FULL = np.arange(50, 1051, 50)

# helper to produce nice orbit labels (uses unicode degree sign)
ORBIT_LABEL = {"45-deg": "45\u00b0", "98-deg": "98\u00b0"}
def orbit_label(orbit):
    return ORBIT_LABEL.get(orbit, orbit.replace("-deg", "\u00b0"))

def parse_N_from_name(fname):
    """Parses the number of simulated events (N) from a filename."""
    m = re.search(r"_N(\d+)([kmKM])?", fname)
    if not m: return None
    n = int(m.group(1))
    suf = m.group(2)
    if not suf: return n
    suf = suf.lower()
    if suf == "k": return n * 1000
    if suf == "m": return n * 1000000
    return n

def parse_metadata_from_name(fname):
    """Parses mirror type, filter status, and energy from a filepath."""
    base = os.path.basename(fname)
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else ""
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else ""
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    N = parse_N_from_name(base)
    hole = "withOpening" if "withhole" in fname.lower() else ("withoutOpening" if "withouthole" in fname.lower() else "")
    return mirror, filter_flag, hole, energy, N

def get_tid_and_error_from_file(rootfile, tid_key):
    """
    Read TID data and return (total_sim_tid, err_on_sum).
    err_on_sum is computed using a Poisson-like uncertainty on the number of
    contributing events: err = total_sim_tid / sqrt(N_hits), where N_hits is
    the number of events with non‑zero per‑event TID. Falls back to SmallDetSummary
    count or per-event count if needed.
    """
    try:
        # read TID array (could be step-level deposits)
        t_tree = rootfile["TID"]
        tid_array = t_tree[tid_key].array(library="np")
        if tid_array.size == 0:
            return 0.0, 0.0

        # aggregate per-event totals if EventID exists in TID tree
        N_events_from_tid = None
        per_event = tid_array
        if "EventID" in t_tree.keys():
            evt = t_tree["EventID"].array(library="np")
            if evt.size == tid_array.size:
                df_tmp = pd.DataFrame({"evt": evt, "tid": tid_array})
                per_event = df_tmp.groupby("evt", sort=False)["tid"].sum().values
                N_events_from_tid = per_event.size

        total_sim_tid = float(per_event.sum())

        # Count number of events that actually deposited non-zero TID
        N_hits = int(np.count_nonzero(per_event > 0))

        # Prefer the number of entries from SmallDetSummary if present (but treat as hits if branch lists only hits)
        N_small = None
        if "SmallDetSummary" in rootfile.keys():
            sd = rootfile["SmallDetSummary"]
            try:
                N_small = int(sd.num_entries)
            except Exception:
                keys = sd.keys()
                if keys:
                    arr = sd[keys[0]].array(library="np")
                    N_small = int(arr.size)

        # Choose N to use for Poisson-like error:
        # prefer N_hits (events with non-zero deposit), else SmallDetSummary, else aggregated events from TID tree
        N_use = N_hits if N_hits > 0 else (N_small if (N_small is not None and N_small > 0) else (N_events_from_tid if N_events_from_tid is not None else 0))

        if N_use <= 0:
            # no reliable denominator -> return total with zero error (caller may log)
            return total_sim_tid, 0.0

        # Poisson-like error on the *sum* assuming fluctuations in the number of contributing events:
        # err(S) = mean_event * sqrt(N_use) = (S / N_use) * sqrt(N_use) = S / sqrt(N_use)
        err_on_sum = total_sim_tid / math.sqrt(N_use)
        return total_sim_tid, float(err_on_sum)

    except Exception:
        return 0.0, 0.0

def calculate_tid_data(dirs, fluence_map, orbit_name):
    """
    Scans ROOT files, reads simulated TID, and scales to mission lifetime for a given orbit.
    """
    r_out_mm = 47.0
    aperture_area_cm2 = math.pi * (r_out_mm / 10.0)**2
    x_aperture_map = {"DPH": 901.0, "DCC": 901.0, "SP": 851.0}

    # --- Energy bin widths in MeV for fluence correction (from calculate_flux.py) ---
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
                    if N is None or N <= 0 or energy is None: continue

                    # --- Get correction factor based on hole status ---
                    correction_factor = CORRECTION_FACTORS.get(hole, 1.0) # Default to 1.0 if not found

                    sim_tid_det, sim_tid_det_err = get_tid_and_error_from_file(f, "TID_krad_det")
                    sim_tid_focal, sim_tid_focal_err = get_tid_and_error_from_file(f, "TID_krad_focal")
                    
                    if sim_tid_det <= 0 and sim_tid_focal <= 0: continue

                    x_aperture_mm = x_aperture_map.get(mirror)
                    if x_aperture_mm is None: continue
                    
                    # --- FIX: Define mission_fluence before using it ---
                    mission_fluence_differential = fluence_map.get(energy, 0.0)

                    # --- Correct the fluence by multiplying with the energy bin width ---
                    bin_width_MeV = energy_bin_widths_MeV.get(energy, 0.0)
                    mission_fluence_integral = mission_fluence_differential * bin_width_MeV

                    theta_max = math.atan(r_out_mm / x_aperture_mm)
                    a_eff_cm2 = aperture_area_cm2 * 0.25 * (math.sin(theta_max)**2)
                    
                    # --- FIX: Use the corrected integral fluence ---
                    total_mission_particles = mission_fluence_integral * a_eff_cm2
                    scaling_factor = total_mission_particles / float(N) if N > 0 else 0.0
                    print(sim_tid_det)
                    # Scale TID values to mission lifetime
                    mission_tid_det = sim_tid_det * scaling_factor
                    mission_tid_focal = sim_tid_focal * scaling_factor
                    
                    # --- FIX: Correct error propagation ---
                    # Use the file-level error returned by get_tid_and_error_from_file.
                    # Scale the simulated-file error to mission lifetime with the same factor.
                    err_mission_tid_det = scaling_factor * sim_tid_det_err if sim_tid_det_err else 0.0
                    err_mission_tid_focal = scaling_factor * sim_tid_focal_err if sim_tid_focal_err else 0.0

                    # --- Apply correction factor ---
                    mission_tid_det *= correction_factor
                    
                    err_mission_tid_det *= correction_factor
                    mission_tid_focal *= correction_factor
                    err_mission_tid_focal *= correction_factor

                    entries.append({
                        "orbit": orbit_name, "mirror": mirror, "filter": filter_flag, "hole": hole, "energy_keV": energy,
                        "mission_tid_det": mission_tid_det, "err_mission_tid_det_sq": err_mission_tid_det**2,
                        "mission_tid_focal": mission_tid_focal, "err_mission_tid_focal_sq": err_mission_tid_focal**2
                    })
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
    return pd.DataFrame(entries)

def compute_tid_reduction(df, group_col, high_val, low_val):
    """
    Computes percent reduction for TID values, similar to the flux reduction function.
    Returns a dict: orbit -> (energies, percent_reduction)
    """
    agg = df.groupby(['orbit', group_col, 'energy_keV'], as_index=False).agg(
        mission_tid_det=('mission_tid_det', 'sum')
    )
    if agg.empty:
        return {}
    
    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', group_col], values='mission_tid_det')
    orbits = sorted(agg['orbit'].unique())
    result = {}

    for orbit in orbits:
        # --- FIX: Use high_val and low_val for generic comparison ---
        s_high = pivot.get((orbit, high_val), pd.Series(dtype=float)).dropna()
        s_low = pivot.get((orbit, low_val), pd.Series(dtype=float)).dropna()
        
        energies = sorted(set(s_high.index).union(s_low.index))
        if not energies:
            result[orbit] = (np.array([]), np.array([]))
            continue

        s_high_re = s_high.reindex(energies).fillna(0.0)
        s_low_re = s_low.reindex(energies).fillna(0.0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if group_col == 'hole':
                # Increase = (with - without) / without * 100
                percent = (s_low_re - s_high_re) / s_high_re * 100.0
            else:
                # Reduction = (off - on) / off * 100
                percent = (s_high_re - s_low_re) / s_high_re * 100.0
        
        percent = percent.replace([np.inf, -np.inf], np.nan)
        result[orbit] = (np.array(energies), percent.values)
    return result

def plot_tid_concentrator_comparison(df, cl_label, baseline_config):
    """Plots TID for different concentrators with a percent gain subplot."""
    df_filtered = df.copy()
    if baseline_config:
        for col, val in baseline_config.items():
            df_filtered = df_filtered[df_filtered[col] == val]

    if df_filtered.empty:
        print(f"No data for Concentrator Comparison plot with baseline {baseline_config}.")
        return

    fig, (ax_tid, ax_gain) = plt.subplots(2, 1, figsize=(12, 10), sharex=False, gridspec_kw={'height_ratios': [3, 1]})

    # --- Axis Styling (from flux plots) ---
    for ax in [ax_tid, ax_gain]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        # set x minor locator explicitly (50 keV grid)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        # configure tick appearance (no ax.minorticks_on)
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    # set major tick labels explicitly (avoid set_ticklabels warning)
    ax_gain.set_xticks(MAJOR_TICKS)
    ax_gain.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_gain.set_ylim(0,100)
    ax_tid.set_xticklabels([])

    # Add minor ticks on y-axis of the lower (gain) axis (every 10 percentage points)
    ax_gain.yaxis.set_minor_locator(MultipleLocator(10))
    ax_gain.tick_params(axis="y", which="minor", direction="in", length=4, width=1.0)

    # --- AGGREGATE over filter and hole to isolate concentrator differences ---
    # sum mission TID and sum error-squares across all filter/hole variants
    agg = df_filtered.groupby(['orbit', 'mirror', 'energy_keV'], as_index=False).agg(
        mission_tid_det=('mission_tid_det', 'sum'),
        # --- FIX: Correctly sum the SQUARED errors before taking the root ---
        err_mission_tid_det_sq=('err_mission_tid_det_sq', 'sum')
    )
    if agg.empty:
        print("No data after aggregating over filter/hole for concentrator comparison.")
        return
    # compute combined error
    agg['err_mission_tid_det'] = np.sqrt(agg['err_mission_tid_det_sq'].fillna(0.0))

    # pivot for gain calculation
    pivot = agg.pivot_table(index='energy_keV', columns=['orbit', 'mirror'], values='mission_tid_det')

    # --- Define linestyles and markers for orbits ---
    linestyles = {"45-deg": "-", "98-deg": "--"}
    markers = {"45-deg": "o", "98-deg": "s"}

    # iterate over aggregated groups for plotting
    for (orbit, mirror), s_df in agg.groupby(['orbit', 'mirror']):
        s_df = s_df.sort_values('energy_keV')
        energies = s_df['energy_keV'].values
        data = s_df['mission_tid_det'].values
        err = s_df['err_mission_tid_det'].values

        if data.size == 0:
            continue

        color = COLOR_MIRROR.get(mirror)
        linestyle = linestyles.get(orbit, '-')
        marker = markers.get(orbit, 'o')
        
        # --- Use different marker fill for 98-deg orbit ---
        mfc = color if orbit == '45-deg' else 'none'

        lower = np.maximum(data - err, 1e-12) # prevent negative/log issues
        upper = data + err
        ax_tid.fill_between(energies, lower, upper, color=color, alpha=0.2)
        ax_tid.plot(energies, data, label=f"{orbit_label(orbit)} / {mirror}", color=color, 
                    linestyle=linestyle, marker=marker, markerfacecolor=mfc, markeredgecolor=color)

    # --- Calculate and plot percent gain vs SP ---
    gain_plotted = False
    for mirror_comp in ['DCC', 'DPH']:
        gain_dfs = []
        for orbit in df_filtered['orbit'].unique():
            s_sp = pivot.get((orbit, 'SP'), pd.Series(dtype=float)).dropna()
            s_comp = pivot.get((orbit, mirror_comp), pd.Series(dtype=float)).dropna()
            
            if s_sp.empty or s_comp.empty: continue
            
            energies = sorted(set(s_sp.index).union(s_comp.index))
            s_sp_re = s_sp.reindex(energies).fillna(0)
            s_comp_re = s_comp.reindex(energies).fillna(0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                gain = (s_comp_re - s_sp_re) / s_sp_re * 100.0
            gain = gain.replace([np.inf, -np.inf], np.nan)
            gain_dfs.append(pd.DataFrame({'energy': energies, f'gain_{orbit_label(orbit)}': gain.values}))
        
        if gain_dfs:
            merged_df = gain_dfs[0]
            for next_df in gain_dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on='energy', how='outer')
            gain_cols = [c for c in merged_df.columns if c.startswith('gain_')]
            merged_df['avg_gain'] = merged_df[gain_cols].mean(axis=1)
            ax_gain.plot(merged_df['energy'], merged_df['avg_gain'], marker='D', color=COLOR_MIRROR.get(mirror_comp), label=f"Average Gain {mirror_comp} vs SP")
            gain_plotted = True

    # baseline_str = ", ".join([f"{k}={v}" for k, v in baseline_config.items()])
    # ax_tid.set_title(f'Concentrator Comparison (Filter On, With Opening) ({cl_label})')
    ax_tid.set_ylabel('Total Ionizing Dose [krad]')
    ax_tid.set_yscale('log')
    ax_gain.set_ylabel('Percentage Gain vs SP (%)')
    ax_gain.axhline(0, color='k', linewidth=0.8, linestyle='--')
    fig.supxlabel('Proton Energy [keV]')
    ax_tid.legend(title="Orbit / Mirror")
    # --- FIX: Only show legend if something was plotted ---
    if gain_plotted:
        ax_gain.legend()
    plt.tight_layout()
    # --- Save high-quality PNG and PDF ---
    figures_dir = "/home/frisoe/Desktop/Thesis/figures/tid_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    fname_base = os.path.join(figures_dir, f"tid_concentrator_comparison_{cl_label}")
    try:
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Plot saved to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save concentrator comparison plot: {e}")
    plt.close()

def plot_final_dcc_tid(df, figures_dir=None):
    """
    Generates a dedicated plot for the final recommended configuration (DCC, filterOn, withOpening)
    and includes data for all available confidence levels.
    """
    figures_dir = "/home/frisoe/Desktop/Thesis/figures/tid_analysis"
    os.makedirs(figures_dir, exist_ok=True)

    # Filter for the specific configuration
    df_final = df[
        (df['mirror'] == 'DCC') &
        (df['filter'] == 'filterOn') &
        (df['hole'] == 'withOpening')
    ].copy()

    if df_final.empty:
        print("No data found for the final DCC configuration (DCC, filterOn, withOpening).")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # --- Apply standard axis styling ---
    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    # ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])

    # --- Set titles and labels ---
    # ax.set_title('Total Ionizing Dose for DCC (Filter On, With Opening)')
    ax.set_xlabel('Proton energy [keV]')
    ax.set_ylabel('Mission Lifetime TID [krad]')
    ax.set_yscale('log')

    # --- Plot data for each orbit and confidence level with error bands ---
    flux_lw = 2.5
    cl_linestyles = {'CL95': '-', 'CL75': '--', 'CL50': ':'}

    for (orbit, cl), group_df in df_final.groupby(['orbit', 'cl']):
        orbit_df = group_df.sort_values('energy_keV')
        if orbit_df.empty:
            continue

        energies = orbit_df['energy_keV'].values
        tid = orbit_df['mission_tid_det'].values
        errors = orbit_df['err_mission_tid_det'].values

        color = COLOR_ORBIT.get(orbit, 'black')
        linestyle = cl_linestyles.get(cl, '-')

        # Plot error band (mission_det) before the line so line is on top
        lower = np.maximum(tid - errors, 1e-12)
        upper = tid + errors
        ax.fill_between(energies, lower, upper, color=color, alpha=0.18, zorder=1)

        # Plot main line with appropriate marker
        mfc = color if cl == 'CL95' else 'none'
        mec = color
        ax.plot(energies, tid, color=color, linestyle=linestyle, lw=flux_lw, marker='o',
                markerfacecolor=mfc, markeredgecolor=mec,
                label=f'{orbit_label(orbit)} / {cl}', zorder=2)

    ax.legend(loc='lower right', ncol=2)
    ax.legend(title='Orbit / Confidence Level')
    plt.tight_layout()

    # --- Save high-quality PNG and PDF ---
    try:
        fname_base = os.path.join(figures_dir, "final_dcc_tid_all_cls")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Saved final DCC plot to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save final DCC TID plot: {e}")
    plt.close()

def plot_tid_orbit_comparison(df, cl_label):
    """
    Compares TID between 45-deg and 98-deg orbits for each mirror type.
    - Top plot: Absolute TID for a baseline configuration (filterOn, withOpening).
    - Bottom plot: Percent reduction of 45-deg relative to 98-deg.
    """
    figures_dir = "/home/frisoe/Desktop/Thesis/figures/tid_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Define baseline configurations for orbit-to-orbit comparison ---
    baseline_mirrors = ['SP', 'DCC', 'DPH']
    baseline_filter = 'filterOn'
    baseline_hole = 'withOpening'

    dfp = df[(df['mirror'].isin(baseline_mirrors)) & 
             (df['filter'] == baseline_filter) & 
             (df['hole'] == baseline_hole)].copy()

    if dfp.empty:
        print(f"No data for TID orbit comparison with baseline config (Mirrors: {baseline_mirrors}, Filter: {baseline_filter}, Hole: {baseline_hole}).")
        return

    # Pivot data for easy access
    pivot = dfp.pivot_table(index='energy_keV', columns=['mirror', 'orbit'], values='mission_tid_det')
    pivot_err = dfp.pivot_table(index='energy_keV', columns=['mirror', 'orbit'], values='err_mission_tid_det')

    # create two stacked axes: top = tid, bottom = reduction (%)
    fig, (ax_tid, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                          gridspec_kw={'height_ratios': [3, 1]})

    # --- Configure Axes (similar to other plots) ---
    for ax in [ax_tid, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_tid.set_xticklabels([])

    fig.supxlabel('Proton energy [keV]')
    ax_tid.set_ylabel('Total Ionizing Dose [krad]')
    ax_tid.set_yscale('log')

    # --- Plot TID and Reduction Data ---
    reduction_dfs = []
    flux_lw = 2.5
    linestyles = {"45-deg": "-", "98-deg": "--"}
    markers = {"45-deg": "o", "98-deg": "s"}
    reduction_marker = 'D'

    for mirror in baseline_mirrors:
        color = COLOR_MIRROR.get(mirror)

        # --- Plot TID Data (Top Plot) ---
        for orbit in ['45-deg', '98-deg']:
            col_id = (mirror, orbit)
            if col_id not in pivot.columns:
                continue

            s = pivot[col_id].dropna()
            err = pivot_err[col_id].reindex(s.index).fillna(0)
            energies = s.index.values

            lower = np.maximum(s.values - err.values, 1e-9)
            upper = s.values + err.values

            ax_tid.fill_between(energies, lower, upper, color=color, alpha=0.12, zorder=1)
            ax_tid.plot(
                energies, s.values, color=color, linestyle=linestyles.get(orbit),
                alpha=0.95, lw=flux_lw, zorder=3, marker=markers.get(orbit),
                markerfacecolor=color if orbit == '45-deg' else 'none', markeredgecolor=color,
                label=f"{mirror} / {orbit_label(orbit)}"
            )

        # --- Calculate and collect Reduction (for Bottom Plot) ---
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

    ax_tid.legend(loc='lower right', ncol=2, title='Concentrator / Orbit')
    ax_red.legend(loc='lower right')

    plt.tight_layout()
    # --- Save high-quality PNG and PDF ---
    try:
        fname_base = os.path.join(figures_dir, f"tid_orbit_comparison_{cl_label}")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Plot saved to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save TID orbit comparison plot: {e}")
    plt.close()

def plot_tid_filter_comparison(df, cl_label, baseline_config):
    """Plots TID for filter on/off with a reduction subplot."""
    df_filtered = df.copy()
    for col, val in baseline_config.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    if df_filtered.empty:
        print(f"No data for Filter Comparison plot with baseline {baseline_config}.")
        return

    fig, (ax_tid, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [3, 1]})
    
    reduction_by_orbit = compute_tid_reduction(df_filtered, 'filter', 'filterOff', 'filterOn')

    # --- Axis Styling (from flux plots) ---
    xticks = np.arange(50, 1051, 50)
    for ax in [ax_tid, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        # x minor locator only
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    

    ax_red.set_xticks(MAJOR_TICKS)
    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_red.set_ylim(0,100)
    ax_tid.set_xticklabels([])

    # y minor ticks on reduction axis only (10%)
    ax_red.yaxis.set_minor_locator(MultipleLocator(10))
    ax_red.tick_params(axis="y", which="minor", direction="in", length=4, width=1.0)

    # --- Iterate over the grouped data directly to use pre-calculated errors ---
    for orbit, s_df in df_filtered.groupby('orbit'):
        # --- FIX: Use pivot_table on the already filtered s_df ---
        pivot = s_df.pivot_table(index='energy_keV', columns='filter', values='mission_tid_det')
        pivot_err = s_df.pivot_table(index='energy_keV', columns='filter', values='err_mission_tid_det')

        for status in ['filterOn', 'filterOff']:
            if status not in pivot.columns: continue
            
            data = pivot[status].dropna()
            if data.empty: continue
            
            data = data.sort_index()
            err = pivot_err[status].reindex(data.index).fillna(0)
            
            # --- Use orbit for color, status for style ---
            color = COLOR_ORBIT.get(orbit, 'black')
            label = {"filterOn": "Filter On", "filterOff": "Filter Off"}[status]
            mfc = 'none' if status == 'filterOff' else color
            mec = color
            linestyle = '--' if status == 'filterOff' else '-'
            marker = 'o'

            lower = np.maximum(data.values - err.values, 1e-9) # Prevent negative values in log scale
            upper = data.values + err.values
            ax_tid.fill_between(data.index, lower, upper, color=color, alpha=0.2)
            ax_tid.plot(data.index, data.values, label=f"{orbit_label(orbit)} / {label}", color=color,
                        marker=marker, markerfacecolor=mfc, markeredgecolor=mec, linestyle=linestyle)

    # Plot average reduction
    reduction_plotted = False
    red_dfs = [pd.DataFrame({'energy': e, f'red_{o}': p}) for o, (e, p) in reduction_by_orbit.items() if e.size > 0]
    if red_dfs:
        merged_df = red_dfs[0]
        for next_df in red_dfs[1:]:
            merged_df = pd.merge(merged_df, next_df, on='energy', how='outer')
        red_cols = [c for c in merged_df.columns if c.startswith('red_')]
        merged_df['avg_reduction'] = merged_df[red_cols].mean(axis=1)
        ax_red.plot(merged_df['energy'], merged_df['avg_reduction'], marker='D', color='#2ca02c', label='Average Reduction')
        reduction_plotted = True

    # baseline_str = ", ".join([f"{k}={v}" for k, v in baseline_config.items()])
    # ax_tid.set_title(f'Filter Comparison ({cl_label})')
    ax_tid.set_ylabel('Total Ionizing Dose [krad]')
    ax_tid.set_yscale('log')
    ax_red.set_ylabel('Reduction (%)')
    fig.supxlabel('Proton Energy [keV]')
    ax_tid.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_red.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_tid.legend(title="Orbit / Filter")
    # --- FIX: Only show legend if something was plotted ---
    if reduction_plotted:
        ax_red.legend()
    plt.tight_layout()
    # --- Save high-quality PNG and PDF ---
    figures_dir = "/home/frisoe/Desktop/Thesis/figures/tid_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    fname_base = os.path.join(figures_dir, f"tid_filter_comparison_{cl_label}")
    try:
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Plot saved to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save filter comparison plot: {e}")
    plt.close()

def plot_tid_hole_comparison(df, cl_label, baseline_config):
    """Plots TID for with/without opening with a reduction subplot."""
    df_filtered = df.copy()
    for col, val in baseline_config.items():
        df_filtered = df_filtered[df_filtered[col] == val]

    if df_filtered.empty:
        print(f"No data for Opening Comparison plot with baseline {baseline_config}.")
        return

    fig, (ax_tid, ax_red) = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [3, 1]})
    
    reduction_by_orbit = compute_tid_reduction(df_filtered, 'hole', 'withoutOpening', 'withOpening')

    for ax in [ax_tid, ax_red]:
        ax.set_xlim(50, 1050)
        ax.set_xticks(MAJOR_TICKS)
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax_red.set_xticks(MAJOR_TICKS)
    ax_red.set_xticklabels([str(int(t)) for t in MAJOR_TICKS])
    ax_red.set_ylim(0,20)
    ax_tid.set_xticklabels([])

    # y minor ticks on increase axis only (2%)
    ax_red.yaxis.set_minor_locator(MultipleLocator(2))
    ax_red.tick_params(axis="y", which="minor", direction="in", length=4, width=1.0)

    # --- Iterate over the grouped data directly to use pre-calculated errors ---
    for orbit, s_df in df_filtered.groupby('orbit'):
        # --- FIX: Use pivot_table on the already filtered s_df ---
        pivot = s_df.pivot_table(index='energy_keV', columns='hole', values='mission_tid_det')
        pivot_err = s_df.pivot_table(index='energy_keV', columns='hole', values='err_mission_tid_det')

        for status in ['withOpening', 'withoutOpening']:
            if status not in pivot.columns: continue
            
            data = pivot[status].dropna()
            if data.empty: continue

            data = data.sort_index()
            err = pivot_err[status].reindex(data.index).fillna(0)
            
            # --- Use orbit for color, status for style ---
            color = COLOR_ORBIT.get(orbit, 'black')
            label = {"withOpening": "With Opening", "withoutOpening": "Without Opening"}[status]
            # --- FIX: Ensure dashed line ('withOpening') has an unfilled marker ---
            mfc = 'none' if status == 'withOpening' else color
            mec = color
            linestyle = '--' if status == 'withOpening' else '-'
            marker = 'o'

            lower = np.maximum(data.values - err.values, 1e-9) # Prevent negative values in log scale
            upper = data.values + err.values
            ax_tid.fill_between(data.index, lower, upper, color=color, alpha=0.2)
            ax_tid.plot(data.index, data.values, label=f"{orbit_label(orbit)} / {label}", color=color,
                        marker=marker, markerfacecolor=mfc, markeredgecolor=mec, linestyle=linestyle)

    # Plot average reduction
    increase_plotted = False
    red_dfs = [pd.DataFrame({'energy': e, f'red_{o}': p}) for o, (e, p) in reduction_by_orbit.items() if e.size > 0]
    if red_dfs:
        merged_df = red_dfs[0]
        for next_df in red_dfs[1:]:
            merged_df = pd.merge(merged_df, next_df, on='energy', how='outer')
        red_cols = [c for c in merged_df.columns if c.startswith('red_')]
        merged_df['avg_reduction'] = merged_df[red_cols].mean(axis=1)
        ax_red.plot(merged_df['energy'], merged_df['avg_reduction'], marker='D', color='#2ca02c', label='Average Increase')
        increase_plotted = True

    # baseline_str = ", ".join([f"{k}={v}" for k, v in baseline_config.items()])
    # ax_tid.set_title(f'Dumbbell Opening Comparison ({cl_label})')
    ax_tid.set_ylabel('Total Ionizing Dose [krad]')
    ax_tid.set_yscale('log')
    ax_red.set_ylabel('Increase (%)')
    fig.supxlabel('Proton Energy [keV]')
    ax_tid.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_red.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_tid.legend(title="Orbit / Opening")
    # --- FIX: Only show legend if something was plotted ---
    if increase_plotted:
        ax_red.legend()
    plt.tight_layout()
    # --- Save high-quality PNG and PDF ---
    figures_dir = "/home/frisoe/Desktop/Thesis/figures/tid_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    fname_base = os.path.join(figures_dir, f"tid_opening_comparison_{cl_label}")
    try:
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Plot saved to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save opening comparison plot: {e}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot mission TID based on simulation results.")
    parser.add_argument("--dirs", "-d", nargs="+", default=["/home/frisoe/Desktop/Root/withhole/", "/home/frisoe/Desktop/Root/withouthole/"], help="Directories to scan for ROOT files.")
    parser.add_argument("--cl", choices=['CL50', 'CL75', 'CL95'], default='CL95', help="Confidence level for fluence.")
    args = parser.parse_args()

    fluence_maps_all = {
        'CL95': {"45-deg": {100: 2.697e11, 250: 2.054e11, 500: 1.416e11, 1000: 6.261e10}, "98-deg-AP9": {100: 6.929e11, 250: 4.527e11, 500: 2.290e11, 1000: 5.240e10}, "98-deg-SP": {100: 2.321e12, 250: 5.304e11, 500: 1.702e11, 1000: 5.484e10}},
        'CL75': {"45-deg": {100: 1.575e11, 250: 1.220e11, 500: 7.381e10, 1000: 3.236e10}, "98-deg-AP9": {100: 3.113e11, 250: 2.047e11, 500: 1.081e11, 1000: 2.757e10}, "98-deg-SP": {100: 3.747e12, 250: 6.560e11, 500: 1.171e11, 1000: 4.482e10}},
        'CL50': {"45-deg": {100: 9.373e10, 250: 7.207e10, 500: 5.209e10, 1000: 2.478e10}, "98-deg-AP9": {100: 2.163e11, 250: 1.382e11, 500: 6.787e10, 1000: 1.903e10}, "98-deg-SP": {100: 4.887e12, 250: 7.113e11, 500: 1.606e11, 1000: 3.644e10}}
    }

    CONFIDENCE_LEVEL = args.cl
    print(f"Using {CONFIDENCE_LEVEL} confidence level for comparison plots.")
    
    # --- Calculate data for ALL confidence levels ---
    all_dfs = []
    for cl_level, fluence_maps in fluence_maps_all.items():
        print(f"Calculating data for {cl_level}...")
        fluence_map_45 = fluence_maps["45-deg"]
        fluence_map_98_ap9 = fluence_maps["98-deg-AP9"]
        fluence_map_98_sp = fluence_maps["98-deg-SP"]
        fluence_map_98 = {e: fluence_map_98_ap9.get(e, 0) + fluence_map_98_sp.get(e, 0) for e in set(fluence_map_98_ap9) | set(fluence_map_98_sp)}

        df_45 = calculate_tid_data(args.dirs, fluence_map_45, "45-deg")
        df_98 = calculate_tid_data(args.dirs, fluence_map_98, "98-deg")
        
        df_cl = pd.concat([df_45, df_98], ignore_index=True)
        df_cl['cl'] = cl_level # Add CL identifier
        all_dfs.append(df_cl)

    df_all_cls = pd.concat(all_dfs, ignore_index=True)

    if df_all_cls.empty:
        print("No valid data found; nothing to calculate or plot.")
    else:
        # Group by all configuration parameters and sum the TID and errors
        group_cols = ['orbit', 'mirror', 'energy_keV', 'hole', 'filter', 'cl']
        agg_dict = {
            'mission_tid_det': 'sum', 'err_mission_tid_det_sq': 'sum',
            'mission_tid_focal': 'sum', 'err_mission_tid_focal_sq': 'sum',
        }
        grouped_all_cls = df_all_cls.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Final error is the sqrt of the sum of squares
        grouped_all_cls['err_mission_tid_det'] = np.sqrt(grouped_all_cls['err_mission_tid_det_sq'])
        grouped_all_cls['err_mission_tid_focal'] = np.sqrt(grouped_all_cls['err_mission_tid_focal_sq'])

        # --- FIX: Filter out 50 keV data point to prevent plotting artifacts ---
        grouped_all_cls = grouped_all_cls[grouped_all_cls['energy_keV'] > 50].copy()

        # Create a DataFrame for the user-selected CL for the comparison plots
        grouped_single_cl = grouped_all_cls[grouped_all_cls['cl'] == CONFIDENCE_LEVEL].copy()

        print("\n" + "="*120)
        print(f"Summed Projected Mission TID ({CONFIDENCE_LEVEL})")
        print("="*120)
        header = f"{'Orbit':<7} | {'Mirror':<6} | {'Filter':<9} | {'Opening':<11} | {'Energy':>6} | {'TID Det (krad)':>18} | {'TID Focal (krad)':>20}"
        print(header)
        print("-"*len(header))

        for _, e in grouped_single_cl.sort_values(by=[c for c in group_cols if c != 'cl']).iterrows():
            print(f"{e['orbit']:<7} | {e['mirror']:<6} | {e['filter']:<9} | {e['hole']:<11} | {int(e['energy_keV']):>6} | {e['mission_tid_det']:>18.3e} | {e['mission_tid_focal']:>20.3e}")

        # --- Print out TID for the specific DCC, filterOn, withOpening, 45-deg orbit configuration ---
        print("\n" + "="*120)
        print(f"Projected TID for DCC, filterOn, withOpening, 45-deg Orbit ({CONFIDENCE_LEVEL})")
        print("="*120)
        
        df_specific = grouped_single_cl[
            (grouped_single_cl['mirror'] == 'DCC') &
            (grouped_single_cl['filter'] == 'filterOn') &
            (grouped_single_cl['hole'] == 'withOpening') &
            (grouped_single_cl['orbit'] == '45-deg')
        ].copy()

        if not df_specific.empty:
            header = f"{'Energy':>6} | {'TID Det (krad)':>18} | {'Error':>18}"
            print(header)
            print("-" * len(header))
            for _, row in df_specific.sort_values(by='energy_keV').iterrows():
                print(f"{int(row['energy_keV']):>6} | {row['mission_tid_det']:>18.3e} | {row['err_mission_tid_det']:>18.3e}")
        else:
            print("No data found for the specified configuration (DCC, filterOn, withOpening, 45-deg).")


        # --- Print TID for DCC / filterOn / withOpening for all CL levels and orbits ---
        df_dcc_allcls = grouped_all_cls[
            (grouped_all_cls['mirror'] == 'DCC') &
            (grouped_all_cls['filter'] == 'filterOn') &
            (grouped_all_cls['hole'] == 'withOpening')
        ].copy()

        print("\n" + "="*120)
        print("Projected TID for DCC, filterOn, withOpening — all CL levels and orbits")
        print("="*120)
        if df_dcc_allcls.empty:
            print("No DCC / filterOn / withOpening data available across CLs.")
        else:
            for cl_level in sorted(df_dcc_allcls['cl'].unique()):
                print(f"\nConfidence level: {cl_level}")
                df_cl = df_dcc_allcls[df_dcc_allcls['cl'] == cl_level]
                for orbit in sorted(df_cl['orbit'].unique()):
                    sub = df_cl[df_cl['orbit'] == orbit].sort_values('energy_keV')
                    if sub.empty:
                        continue
                    hdr = f"{'Energy(keV)':>10} | {'TID Det [krad]':>15} | {'Err Det [krad]':>15} | {'TID Focal [krad]':>17} | {'Err Focal [krad]':>15}"
                    print(f"\n... orbit: {orbit}")
                    print(hdr)
                    print("-" * len(hdr))
                    for _, r in sub.iterrows():
                        print(f"{int(r['energy_keV']):>10} | {r['mission_tid_det']:>15.3e} | {r['err_mission_tid_det']:>15.3e} | {r['mission_tid_focal']:>17.3e} | {r['err_mission_tid_focal']:>15.3e}")
        print("\n")

        # --- Calculate and print delta TID table ---
        print("\n" + "="*140)
        print(f"Delta TID Comparison Table at {CONFIDENCE_LEVEL}")
        print(f"Baseline: DCC, filterOn, withOpening (relative to each orbit)")
        print("="*140)

        # Group by each unique configuration and sum the total TID across all energies
        config_totals_tid = grouped_single_cl.groupby(['orbit', 'mirror', 'filter', 'hole'])['mission_tid_det'].sum().reset_index()

        # Find the baseline TID for each orbit
        baseline_45_deg = config_totals_tid[
            (config_totals_tid['orbit'] == '45-deg') & (config_totals_tid['mirror'] == 'DCC') &
            (config_totals_tid['filter'] == 'filterOn') & (config_totals_tid['hole'] == 'withOpening')
        ]
        baseline_98_deg = config_totals_tid[
            (config_totals_tid['orbit'] == '98-deg') & (config_totals_tid['mirror'] == 'DCC') &
            (config_totals_tid['filter'] == 'filterOn') & (config_totals_tid['hole'] == 'withOpening')
        ]

        if baseline_45_deg.empty or baseline_98_deg.empty:
            
            print("One or more baseline configurations not found. Cannot calculate full delta table.")
        else:
            tid_45_base = baseline_45_deg['mission_tid_det'].iloc[0]
            tid_98_base = baseline_98_deg['mission_tid_det'].iloc[0]

            # Define a function to apply the correct baseline
            def calculate_pct_change(row):
                if row['orbit'] == '45-deg':
                    if tid_45_base > 0:
                        return ((row['mission_tid_det'] - tid_45_base) / tid_45_base) * 100
                elif row['orbit'] == '98-deg':
                    if tid_98_base > 0:
                        return ((row['mission_tid_det'] - tid_98_base) / tid_98_base) * 100
                return 0.0

            config_totals_tid['tid_change_pct'] = config_totals_tid.apply(calculate_pct_change, axis=1)
            
            # Sort by orbit, then by percentage change descending
            config_totals_tid.sort_values(by=['orbit', 'tid_change_pct'], ascending=[True, False], inplace=True)

            # Print the formatted table
            header = f"{'Orbit':<7} | {'Mirror':<6} | {'Filter':<9} | {'Hole':<14} | {'Total TID [krad]':>18} | {'TID Change (%)':>16}"
            print(header)
            print("-" * len(header))
            for _, row in config_totals_tid.iterrows():
                # Conditionally format the percentage to remove '+' from zero
                pct_str = f"{row['tid_change_pct']:>16.2f}" if abs(row['tid_change_pct']) < 1e-9 else f"{row['tid_change_pct']:>+16.2f}"
                print(f"{row['orbit']:<7} | {row['mirror']:<6} | {row['filter']:<9} | {row['hole']:<14} | {row['mission_tid_det']:>18.3e} | {pct_str}")


         # --- Generate the three requested comparison plots ---
         # 1. Filter comparison (baseline: SP mirror, with opening)
        plot_tid_filter_comparison(grouped_single_cl, CONFIDENCE_LEVEL, 
                                    baseline_config={'mirror': 'SP', 'hole': 'withOpening'})

        # 2. Opening comparison (baseline: SP mirror, with filter)
        plot_tid_hole_comparison(grouped_single_cl, CONFIDENCE_LEVEL,
                                 baseline_config={'mirror': 'SP', 'filter': 'filterOn'})

        # 3. Concentrator comparison (baseline: with filter, with opening)
        plot_tid_concentrator_comparison(grouped_single_cl, CONFIDENCE_LEVEL,
                                         baseline_config=None)
        
        # --- Add call to the new orbit comparison plot ---
        plot_tid_orbit_comparison(grouped_single_cl, CONFIDENCE_LEVEL)

        # --- Plot the final DCC TID with all confidence levels ---
        plot_final_dcc_tid(grouped_all_cls)