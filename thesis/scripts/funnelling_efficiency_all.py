import os
import re
import glob
import math
#import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter


# --- UPDATED: Match plotting style to funnelling_efficiency.py ---
plt.rcParams.update({
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

MAJOR_TICKS = [100, 250, 500, 1000]

COLOR_FILTER = {"filterOn": "#d62728", "filterOff": "#1f77b4"}
COLOR_HOLE   = {"withHole": "#d62728", "withoutHole": "#1f77b4"}

# Add concentrator colors (same palette)
COLOR_MIRROR = {"DCC": "#2ca02c", "DPH": "#DC143C", "SP": "#4169E1"}

# Scale factor for concentrator error bands (increase for readability)
ERROR_BAND_FACTOR_CONC = 1.0

# --- Systematic Correction Factors from Validation ---
SYSTEMATIC_CORRECTIONS = {
    'Low_Energy': {  # For <= 250 keV (uses Single Scattering model for correction)
        'C_bar_eta': 0.7556,
        'sigma_C': 0.2102
    },
    'High_Energy': { # For > 250 keV (uses MS opt 4 for correction)
        'C_bar_eta': 0.8142,
        'sigma_C': 0.2139
    }
}

def get_systematic_correction(energy):
    """Returns the appropriate systematic correction dictionary for a given energy."""
    if energy is None: return None
    if energy <= 250:
        return SYSTEMATIC_CORRECTIONS['Low_Energy']
    else:
        return SYSTEMATIC_CORRECTIONS['High_Energy']


# Systematic Error
CORRECTION_FACTORS = {
    "withHole": 0.92131,
    "withoutHole": 0.92022
}

# helper to produce figure directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "eff_analysis")
os.makedirs(FIGURES_DIR, exist_ok=True)

def parse_N_from_name(fname):
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

def parse_metadata_from_name(path):
    base = os.path.basename(path)
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else ""
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else ""
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    N = parse_N_from_name(base)
    hole = "withHole" if "withhole" in path.lower() else ("withoutHole" if "withouthole" in path.lower() else "")
    return mirror, filter_flag, hole, energy, N

def compute_binomial(k, N):
    if N is None or N <= 0:
        return None, None
    k = int(max(0, k))
    p = k / float(N)
    err = math.sqrt(max(0.0, p * (1.0 - p) / N))
    return p, err

def unique_eventid_union(f, tree_names=("SmallDetSummary", "FocalDet", "SmallDet", "FocalDet")):
    ids = None
    for t in tree_names:
        if t in f:
            try:
                arrs = f[t].arrays(["EventID"], library="np")
                ev = np.asarray(arrs["EventID"]).astype(int).ravel()
                if ev.size == 0:
                    continue
                if ids is None:
                    ids = np.unique(ev)
                else:
                    ids = np.unique(np.concatenate((ids, np.unique(ev))))
            except Exception:
                continue
    if ids is None:
        return 0
    return int(ids.size)

def collect_entries(dirs):
    entries = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        pattern = os.path.join(d, "**", "output_*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        for fpath in files:
            try:
                f = uproot.open(fpath)
            except Exception:
                continue
            mirror, filter_flag, hole, energy, N = parse_metadata_from_name(fpath)
            if N is None:
                # skip files without documented N
                continue
            
            # --- Get correction factor based on hole status ---
            correction_factor = CORRECTION_FACTORS.get(hole, 1.0)

            # union of unique EventID that appear in either small detector or focal plane trees
            hits_union = unique_eventid_union(f)
            eff, err = compute_binomial(hits_union, N)
            
            # --- Apply correction factor ---
            if eff is not None: eff *= correction_factor
            if err is not None: err *= correction_factor

            # focal-plane only: count EventID occurrences in FocalDet / FocalDetSummary
            hits_focal = unique_eventid_union(f, tree_names=("FocalDet", "FocalDetSummary"))
            eff_focal, err_focal = compute_binomial(hits_focal, N)

            # --- Apply correction factor ---
            if eff_focal is not None: eff_focal *= correction_factor
            if err_focal is not None: err_focal *= correction_factor

            # --- Small-detector only hits ---
            hits_smalldet = unique_eventid_union(f, tree_names=("SmallDet", "SmallDetSummary"))
            eff_smalldet, err_smalldet = compute_binomial(hits_smalldet, N)

            # --- Apply correction factor to smalldet efficiency ---
            if eff_smalldet is not None: eff_smalldet *= correction_factor
            if err_smalldet is not None: err_smalldet *= correction_factor

            # --- Calculate and store systematic corrections for all efficiency types ---
            syst = get_systematic_correction(energy)
            
            def apply_syst(eff, err):
                if syst and eff is not None and err is not None:
                    c_bar_eta = syst['C_bar_eta']
                    sigma_c = syst['sigma_C']
                    eff_syst = eff * c_bar_eta
                    err_stat_syst = err * c_bar_eta
                    rel_syst_err = (sigma_c / c_bar_eta) if c_bar_eta != 0 else 0
                    err_syst = eff_syst * rel_syst_err
                    return eff_syst, err_stat_syst, err_syst
                return None, None, None

            eff_syst, err_stat_syst, err_syst = apply_syst(eff, err)
            eff_focal_syst, err_focal_stat_syst, err_focal_syst = apply_syst(eff_focal, err_focal)
            eff_smalldet_syst, err_smalldet_stat_syst, err_smalldet_syst = apply_syst(eff_smalldet, err_smalldet)

            entries.append({
                "path": fpath,
                "basename": os.path.basename(fpath),
                "mirror": mirror,
                "filter": filter_flag,
                "hole": hole,
                "energy_keV": energy,
                "N": N,
                "hits_union": hits_union,
                "eff": eff,
                "err": err,
                "hits_focal": hits_focal,
                "eff_focal": eff_focal,
                "err_focal": err_focal,
                "eff_smalldet": eff_smalldet,
                "err_smalldet": err_smalldet,
                # Add systematically corrected values
                "eff_syst": eff_syst, "err_stat_syst": err_stat_syst, "err_syst": err_syst,
                "eff_focal_syst": eff_focal_syst, "err_focal_stat_syst": err_focal_stat_syst, "err_focal_syst": err_focal_syst,
                "eff_smalldet_syst": eff_smalldet_syst, "err_smalldet_stat_syst": err_smalldet_stat_syst, "err_smalldet_syst": err_smalldet_syst,
            })
    return entries

# remove saving helper - we will save using FIGURES_DIR
# def _ensure_dir(d):
#     os.makedirs(d, exist_ok=True)
#     return d

def plot_comparison(entries, comparison_col, fixed_col, fixed_val, concentrator, colors, legend_title):
    # exclude 50 keV and require valid energy
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val and e.get("mirror") == concentrator and e.get("energy_keV") is not None and e.get("energy_keV") != 50]
    if not filtered:
        return
    groups = {}
    for e in filtered:
        key = e.get(comparison_col) or "unknown"
        groups.setdefault(key, []).append(e)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xlabel("Energy [keV]")
    # --- REVERT: Use original y-axis label ---
    ax.set_ylabel("Funnelling Efficiency [hits / N]")
    label_map = {"filterOn": "With Filter", "filterOff": "Without Filter", "withHole": "With Opening", "withoutHole": "Without Opening"}
    
    handles, labels = [], []

    for key, grp in sorted(groups.items()):
        energies = np.array([g["energy_keV"] for g in grp])
        # Use systematically corrected SmallDet values
        effs = np.array([g.get("eff_smalldet", 0.0) for g in grp], dtype=float)
        errs = np.array([g.get("err_smalldet", 0.0) for g in grp], dtype=float)
        order = np.argsort(energies)
        energies = energies[order]
        effs = effs[order]
        errs = errs[order]
        color = colors.get(key, "k")
        # plot line and shaded error band around focal-plane efficiency
        line, = ax.plot(energies, effs, "-o", color=color, label=label_map.get(key, key))
        # --- FIX: Use 'errs' (statistical error) for the non-syst plot ---
        ax.fill_between(energies, np.maximum(0.0, effs - errs), effs + errs, color=color, alpha=0.2)
        handles.append(line)
        labels.append(label_map.get(key, key))

    # Ticks, limits and grid (match style)
    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    # set minor ticks every 50 keV (use a Locator object, not AutoLocator with an argument)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', alpha=0.6)
    # --- Set y-axis limit and let formatter be default ---
    ax.set_ylim(1e-4, 1e-3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Ensure scientific notation is used for the offset
    ax.legend(handles, labels, title=legend_title, ncol=1, loc="lower right")
    plt.tight_layout()

    # Save plot
    fname = os.path.join(FIGURES_DIR, f"{comparison_col}_{fixed_col}_{fixed_val}_{concentrator}.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)


def plot_comparison_with_syst(entries, comparison_col, fixed_col, fixed_val, concentrator, colors, legend_title):
    """Plots comparison with layered statistical and systematic error bands."""
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val and e.get("mirror") == concentrator and e.get("energy_keV") is not None and e.get("energy_keV") != 50]
    if not filtered:
        return
    groups = {}
    for e in filtered:
        key = e.get(comparison_col) or "unknown"
        groups.setdefault(key, []).append(e)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Funnelling Efficiency [hits / N]")
    label_map = {"filterOn": "With Filter", "filterOff": "Without Filter", "withHole": "With Opening", "withoutHole": "Without Opening"}
    
    handles, labels = [], []

    for key, grp in sorted(groups.items()):
        energies = np.array([g["energy_keV"] for g in grp])
        # Use systematically corrected SmallDet values
        effs = np.array([g.get("eff_smalldet_syst", 0.0) for g in grp], dtype=float)
        stat_errs = np.array([g.get("err_smalldet_stat_syst", 0.0) for g in grp], dtype=float)
        syst_errs = np.array([g.get("err_smalldet_syst", 0.0) for g in grp], dtype=float)
        total_errs = np.sqrt(stat_errs**2 + syst_errs**2)

        order = np.argsort(energies)
        energies, effs, stat_errs, total_errs = energies[order], effs[order], stat_errs[order], total_errs[order]
        
        color = colors.get(key, "k")
        
        # --- Plot only the total error band ---
        ax.fill_between(energies, np.maximum(0.0, effs - total_errs), effs + total_errs, color=color, alpha=0.2)
        
        line, = ax.plot(energies, effs, "-o", color=color, label=label_map.get(key, key))
        handles.append(line)
        labels.append(label_map.get(key, key))

    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax.set_ylim(1e-4, 1e-3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(handles=handles, labels=labels, title=legend_title, ncol=1, loc="lower right")
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"{comparison_col}_{fixed_col}_{fixed_val}_{concentrator}_syst.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)


def plot_concentrator_comparison(entries, fixed_col, fixed_val, colors, legend_title):
    """
    Compare concentrators (DCC/DPH/SP) on the same axes for the given fixed condition
    (e.g. hole == withHole or filter == filterOff). Saves plot and adds a reduction subplot
    underneath the main comparison (percent reduction relative to a reference concentrator).
    """
    # Filter entries by the fixed condition
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val]
    if not filtered:
        print(f"No data for condition {fixed_col}={fixed_val}; skipping concentrator comparison.")
        return

    # Group by mirror
    groups = {}
    for e in filtered:
        key = e.get("mirror") or "Unknown"
        groups.setdefault(key, []).append(e)

    # Aggregate to a single point per energy per concentrator (exclude 50 keV)
    conc_data = {}
    for key in sorted(groups.keys()):
        grp = [g for g in groups[key] if g.get("energy_keV") is not None and g.get("energy_keV") != 50]
        if not grp:
            continue
        by_energy = {}
        for g in grp:
            e = int(g["energy_keV"])
            # --- use small-detector efficiency only ---
            eff_val = g.get("eff_smalldet", None)
            err_val = g.get("err_smalldet", None)
            # if smalldet info missing fall back to NaN so it gets ignored in aggregation
            if eff_val is None:
                continue
            by_energy.setdefault(e, []).append((eff_val, err_val if err_val is not None else 0.0))
        energies = []
        effs_agg = []
        errs_agg = []
        for e in sorted(by_energy.keys()):
            vals = by_energy[e]
            effs = np.array([v[0] for v in vals], dtype=float)
            errs = np.array([v[1] for v in vals], dtype=float)
            # weighted average if valid errors exist
            positive_err = errs > 0
            if positive_err.any():
                w = np.where(positive_err, 1.0 / (errs**2), 0.0)
                if w.sum() > 0:
                    eff_mean = (w * effs).sum() / w.sum()
                    err_comb = 1.0 / math.sqrt(w.sum())
                else:
                    eff_mean = effs.mean()
                    err_comb = effs.std(ddof=1) / math.sqrt(len(effs)) if len(effs) > 1 else 0.0
            else:
                eff_mean = effs.mean()
                err_comb = effs.std(ddof=1) / math.sqrt(len(effs)) if len(effs) > 1 else 0.0
            energies.append(e)
            effs_agg.append(eff_mean)
            errs_agg.append(err_comb)
        if energies:
            conc_data[key] = {
                "energies": np.array(energies),
                "effs": np.array(effs_agg),
                "errs": np.array(errs_agg)
            }

    if not conc_data:
        print(f"No aggregated concentrator data for condition {fixed_col}={fixed_val}.")
        return

    # Create two-row figure: top = concentrator comparison, bottom = percent reduction
    fig, (ax_main, ax_red) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_main.set_xlabel("")  # main x-label omitted, put on reduction axis
    # --- REVERT: Use original y-axis label ---
    ax_main.set_ylabel("Funnelling Efficiency [hits / N]")
    ax_red.set_xlabel("Incident energy [keV]")
    ax_red.set_ylabel("Reduction (%)")
    
    label_map = {"DCC": "DCC", "DPH": "DPH", "SP": "SP"}
    handles, labels = [], []

    # Plot main comparison
    for key in sorted(conc_data.keys()):
        data = conc_data[key]
        energies = data["energies"]
        effs = data["effs"]
        errs = data["errs"]
        order = np.argsort(energies)
        energies = energies[order]
        effs = effs[order]
        errs = errs[order]
        color = colors.get(key, "k")
        # --- FIX: Use 'errs' (statistical error) for the non-syst plot ---
        ax_main.fill_between(energies, np.maximum(0.0, effs - errs), effs + errs, color=color, alpha=0.2)
        line, = ax_main.plot(energies, effs, "-o", color=color, label=label_map.get(key, key))
        handles.append(line)
        labels.append(label_map.get(key, key))

    # --- plotting style / ticks (match funnelling_efficiency.py) ---
    ax_main.set_xlim(50, 1050)
    ax_main.set_xticks(np.arange(50, 1051, 50), minor=True)
    ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax_main.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax_main.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax_main.set_axisbelow(True)
    ax_main.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax_main.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    # --- Set y-axis limit and let formatter be default ---
    ax_main.set_ylim(1e-4, 1e-3)
    ax_main.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Ensure scientific notation is used for the offset
    ax_main.legend(handles, labels, title=legend_title, ncol=1, loc="lower right")

    # Reduction plot: percent reduction relative to reference concentrator
    # pick reference concentrator: prefer 'SP' else first key
    ref_key = 'SP' if 'SP' in conc_data else sorted(conc_data.keys())[0]
    ref = conc_data[ref_key]
    # build dictionary of percent reductions per other conc keyed by energy
    reductions = {}
    for key, data in conc_data.items():
        if key == ref_key:
            continue
        # find common energies between ref and this concentrator
        common_energies = np.intersect1d(ref["energies"], data["energies"])
        if common_energies.size == 0:
            continue
        pct = []
        energies_common = []
        for e in common_energies:
            ref_idx = np.where(ref["energies"] == e)[0][0]
            oth_idx = np.where(data["energies"] == e)[0][0]
            ref_eff = ref["effs"][ref_idx]
            oth_eff = data["effs"][oth_idx]
            if ref_eff and not np.isnan(ref_eff) and ref_eff != 0.0:
                reduction = 100.0 * (ref_eff - oth_eff) / ref_eff
            else:
                reduction = np.nan
            energies_common.append(e)
            pct.append(reduction)
        if energies_common:
            reductions[key] = (np.array(energies_common), np.array(pct))

    # Plot reductions
    red_handles, red_labels = [], []
    for key in sorted(reductions.keys()):
        energies, pct = reductions[key]
        color = colors.get(key, "k")
        h, = ax_red.plot(energies, pct, "-o", color=color, label=f"{label_map.get(key,key)} vs {label_map.get(ref_key,ref_key)}")
        red_handles.append(h)
        red_labels.append(f"{label_map.get(key,key)} vs {label_map.get(ref_key,ref_key)}")

    # zero line
    ax_red.axhline(0.0, color='gray', linestyle='--', linewidth=1.0)

    # style ticks and grid for reduction axis
    ax_red.set_xlim(50, 1050)
    ax_red.set_ylim(0, 100)

    ax_red.set_xticks(MAJOR_TICKS)
    ax_red.xaxis.set_minor_locator(ticker.MultipleLocator(50))  # 1 minor between majors (if majors evenly spaced)

    ax_red.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax_red.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax_red.set_axisbelow(True)
    ax_red.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax_red.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)

    ax_red.set_ylim(50, 100)

    if red_handles:
        ax_red.legend(red_handles, red_labels, title=f"Reduction vs {label_map.get(ref_key,ref_key)}", loc="upper right", ncol=1)

    plt.tight_layout()

    # Save plot
    fname = os.path.join(FIGURES_DIR, f"concentrator_comparison_{fixed_col}_{fixed_val}.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)

def plot_concentrator_comparison_with_syst(entries, fixed_col, fixed_val, colors, legend_title):
    """Plots concentrator comparison with layered statistical and systematic error bands."""
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val]
    if not filtered: return

    groups = {}
    for e in filtered:
        groups.setdefault(e.get("mirror") or "Unknown", []).append(e)

    conc_data = {}
    for key in sorted(groups.keys()):
        grp = [g for g in groups[key] if g.get("energy_keV") is not None and g.get("energy_keV") != 50]
        if not grp: continue
        by_energy = {}
        for g in grp:
            e = int(g["energy_keV"])
            # Use systematically corrected SmallDet values
            eff_val = g.get("eff_smalldet_syst")
            stat_err_val = g.get("err_smalldet_stat_syst")
            syst_err_val = g.get("err_smalldet_syst")
            if eff_val is None: continue
            by_energy.setdefault(e, []).append((eff_val, stat_err_val or 0.0, syst_err_val or 0.0))
        
        energies, effs_agg, stat_errs_agg, total_errs_agg = [], [], [], []
        for e in sorted(by_energy.keys()):
            vals = by_energy[e]
            effs = np.array([v[0] for v in vals])
            stat_errs = np.array([v[1] for v in vals])
            syst_errs = np.array([v[2] for v in vals])
            
            w = 1.0 / stat_errs**2 if np.all(stat_errs > 0) else np.ones_like(effs)
            if np.sum(w) == 0: w = np.ones_like(effs)
            eff_mean = np.sum(w * effs) / np.sum(w)
            stat_err_comb = 1.0 / np.sqrt(np.sum(w)) if np.all(stat_errs > 0) and np.sum(w) > 0 else (np.sqrt(np.sum(stat_errs**2))/len(stat_errs))
            syst_err_comb = np.sqrt(np.sum(syst_errs**2)) / len(syst_errs)

            energies.append(e)
            effs_agg.append(eff_mean)
            stat_errs_agg.append(stat_err_comb)
            total_errs_agg.append(np.sqrt(stat_err_comb**2 + syst_err_comb**2))

        if energies:
            conc_data[key] = {
                "energies": np.array(energies), "effs": np.array(effs_agg),
                "stat_errs": np.array(stat_errs_agg), "total_errs": np.array(total_errs_agg)
            }

    if not conc_data: return

    fig, (ax_main, ax_red) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_main.set_ylabel("Funnelling Efficiency [hits / N]")
    ax_red.set_xlabel("Incident energy [keV]")
    ax_red.set_ylabel("Reduction (%)")
    
    label_map = {"DCC": "DCC", "DPH": "DPH", "SP": "SP"}
    handles, labels = [], []

    for key in sorted(conc_data.keys()):
        data = conc_data[key]
        energies, effs, stat_errs, total_errs = data["energies"], data["effs"], data["stat_errs"], data["total_errs"]
        order = np.argsort(energies)
        energies, effs, stat_errs, total_errs = energies[order], effs[order], stat_errs[order], total_errs[order]
        
        color = colors.get(key, "k")
        # --- Plot only the total error band ---
        ax_main.fill_between(energies, np.maximum(0.0, effs - total_errs), effs + total_errs, color=color, alpha=0.2)
        line, = ax_main.plot(energies, effs, "-o", color=color, label=label_map.get(key, key))
        handles.append(line)
        labels.append(label_map.get(key, key))

    ax_main.set_xlim(50, 1050)
    ax_main.set_ylim(1e-4, 1e-3)
    ax_main.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax_main.legend(handles=handles, labels=labels, title=legend_title, ncol=1, loc="lower right")
    ax_main.grid(which='major', linestyle='--', alpha=0.6)

    # Reduction plot logic remains the same as it's a relative comparison
    ref_key = 'SP' if 'SP' in conc_data else (sorted(conc_data.keys())[0] if conc_data else None)
    if ref_key:
        ref = conc_data[ref_key]
        for key, data in conc_data.items():
            if key == ref_key: continue
            common_energies = np.intersect1d(ref["energies"], data["energies"])
            if common_energies.size > 0:
                ref_effs = np.interp(common_energies, ref["energies"], ref["effs"])
                oth_effs = np.interp(common_energies, data["energies"], data["effs"])
                reduction = 100.0 * (ref_effs - oth_effs) / ref_effs
                ax_red.plot(common_energies, reduction, "-o", color=colors.get(key, "k"), label=f"{label_map.get(key,key)} vs {label_map.get(ref_key,ref_key)}")
    
    ax_red.axhline(0.0, color='gray', linestyle='--', linewidth=1.0)
    ax_red.set_ylim(50, 100)
    ax_red.grid(which='major', linestyle='--', alpha=0.6)
    if len(conc_data) > 1:
        ax_red.legend(title=f"Reduction vs {label_map.get(ref_key,ref_key)}", loc="upper right")

    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"concentrator_comparison_{fixed_col}_{fixed_val}_syst.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)


def unique_eventid_in_circle(root_path, radius_cm=1.0, tree_names=("FocalDet","SmallDet","FocalDetSummary","SmallDetSummary")):
    """
    Return number of unique EventID that have at least one hit with sqrt(y^2+z^2) <= radius_cm
    in any of the provided trees. Assumes position branches are in cm. Tries a set of common
    branch name variants for (EventID, y, z).

    NOTE: treat SmallDetSummary (detector summary) as entirely inside the circle — count all
    EventID listed there without checking positions.
    """
    try:
        f = uproot.open(root_path)
    except Exception:
        return 0
    ids = set()
    # common candidate branch name triples (evt, y, z)
    candidates = [
        ("EventID","y","z"), ("EventID","Y","Z"),
        ("EventID","posY","posZ"), ("EventID","pos_y","pos_z"),
        ("EventID","HitPosY","HitPosZ"), ("EventID","yHit","zHit"),
    ]
    for t in tree_names:
        if t not in f:
            continue
        tree = f[t]

        # Shortcut: treat small-detector summary trees as entirely inside the circle
        if t.lower() == "smalldetsummary" or ("small" in t.lower() and "summary" in t.lower()):
            try:
                arrs = tree.arrays(["EventID"], library="np")
                ev = np.asarray(arrs["EventID"]).astype(int).ravel()
                if ev.size > 0:
                    ids.update(np.unique(ev))
            except Exception:
                pass
            continue

        # try to find a compatible branch triple for positional checks
        for evt_b, y_b, z_b in candidates:
            if evt_b in tree.keys() and y_b in tree.keys() and z_b in tree.keys():
                try:
                    arrs = tree.arrays([evt_b, y_b, z_b], library="np")
                except Exception:
                    continue
                evt = np.asarray(arrs[evt_b]).ravel().astype(int)
                yy = np.asarray(arrs[y_b]).ravel().astype(float)
                zz = np.asarray(arrs[z_b]).ravel().astype(float)
                # handle broadcasting / nested arrays if present
                if yy.ndim > 1:
                    yy = np.hstack([a.ravel() for a in yy])
                if zz.ndim > 1:
                    zz = np.hstack([a.ravel() for a in zz])
                # If EventID array is not the same shape as coords, try flattening per-hit event ids
                if evt.shape != yy.shape:
                    try:
                        evt = np.asarray(arrs[evt_b]).ravel().astype(int)
                    except Exception:
                        continue
                d = np.sqrt(yy**2 + zz**2)
                mask = d <= float(radius_cm)
                if mask.any():
                    ids.update(np.unique(evt[mask]))
                break
    return int(len(ids))

def plot_concentrator_circular(entries, fixed_col, fixed_val, colors, legend_title, radius_cm=1.0):
    """
    Compare concentrators using union of focal+detector hits that fall inside a circle
    of radius `radius_cm` (yz plane, origin at 0). Aggregates per-energy and plots efficiencies.
    """
    # filter entries by fixed condition
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val and e.get("energy_keV") is not None and e.get("energy_keV") != 50]
    if not filtered:
        print(f"No data for condition {fixed_col}={fixed_val}; skipping circular-area concentrator comparison.")
        return

    # group file paths by mirror -> energy
    grouped = {}
    for e in filtered:
        mirror = e.get("mirror") or "Unknown"
        energy = int(e["energy_keV"])
        grouped.setdefault(mirror, {}).setdefault(energy, []).append(e)

    conc_data = {}
    for mirror, energies_map in grouped.items():
        energies = []
        effs_agg = []
        errs_agg = []
        for energy in sorted(energies_map.keys()):
            vals = energies_map[energy]
            per_file_effs = []
            per_file_errs = []
            for v in vals:
                path = v["path"]
                N = v.get("N") or 0
                hole_status = v.get("hole")
                if N <= 0:
                    continue
                
                # --- Get correction factor for this file ---
                correction_factor = CORRECTION_FACTORS.get(hole_status, 1.0)

                # count unique EventID in circle (union of focal+detector because we check both trees)
                k = unique_eventid_in_circle(path, radius_cm=radius_cm)
                p, err = compute_binomial(k, N)
                if p is None:
                    continue
                
                # --- Apply correction factor ---
                p *= correction_factor
                if err is not None: err *= correction_factor

                eff_syst, stat_err_syst, syst_err = apply_syst(p, err, get_systematic_correction(energy))
                if eff_syst is None: continue
                
                per_file_effs.append(float(eff_syst))
                per_file_errs.append(float(stat_err_syst if stat_err_syst is not None else 0.0))
            if not per_file_effs:
                continue
            per_file_effs = np.array(per_file_effs)
            per_file_errs = np.array(per_file_errs)
            # aggregate across files for this energy (weighted by reported errors if available)
            positive_err = per_file_errs > 0
            if positive_err.any():
                w = np.where(positive_err, 1.0 / (per_file_errs**2), 0.0)
                if w.sum() > 0:
                    eff_mean = (w * per_file_effs).sum() / w.sum()
                    err_comb = 1.0 / math.sqrt(w.sum())
                else:
                    eff_mean = per_file_effs.mean()
                    err_comb = per_file_effs.std(ddof=1) / math.sqrt(len(per_file_effs)) if len(per_file_effs) > 1 else 0.0
            else:
                eff_mean = per_file_effs.mean()
                err_comb = per_file_effs.std(ddof=1) / math.sqrt(len(per_file_effs)) if len(per_file_effs) > 1 else 0.0

            energies.append(energy)
            effs_agg.append(eff_mean)
            errs_agg.append(err_comb)
        if energies:
            conc_data[mirror] = {
                "energies": np.array(energies),
                "effs": np.array(effs_agg),
                "errs": np.array(errs_agg)
            }

    if not conc_data:
        print(f"No aggregated circular-area data for condition {fixed_col}={fixed_val}.")
        return

    # plotting: one panel (could be two like other function), keep consistent style
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xlabel("Incident energy [keV]")
    # --- REVERT: Use original y-axis label ---
    ax.set_ylabel(f"Funnelling Efficiency [hits / N]")
    handles, labels = [], []
    for key in sorted(conc_data.keys()):
        data = conc_data[key]
        energies = data["energies"]
        effs = data["effs"]
        errs = data["errs"]
        order = np.argsort(energies)
        energies = energies[order]
        effs = effs[order]
        errs = errs[order]
        color = colors.get(key, "k")
        # --- FIX: Use 'errs' (statistical error) for the non-syst plot ---
        ax.fill_between(energies, np.maximum(0.0, effs - errs), effs + errs, color=color, alpha=0.2)
        line, = ax.plot(energies, effs, "-o", color=color, label=key)
        handles.append(line)
        labels.append(key)

    # ticks and grid like other plots
    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)
    ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.set_ylim(bottom=0)
    # --- Set y-axis limit and let formatter be default ---
    ax.set_ylim(1e-4, 1e-3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Ensure scientific notation is used for the offset
    ax.legend(handles, labels, title=legend_title, loc="lower right")
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"concentrator_circular_{fixed_col}_{fixed_val}_{int(radius_cm*10)}mm.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)

def plot_concentrator_circular_with_syst(entries, fixed_col, fixed_val, colors, legend_title, radius_cm=1.0):
    """Plots circular-area comparison with layered statistical and systematic error bands."""
    filtered = [e for e in entries if e.get(fixed_col) == fixed_val and e.get("energy_keV") is not None and e.get("energy_keV") != 50]
    if not filtered: return

    grouped = {}
    for e in filtered:
        grouped.setdefault(e.get("mirror") or "Unknown", {}).setdefault(int(e["energy_keV"]), []).append(e)

    conc_data = {}
    for mirror, energies_map in grouped.items():
        energies, effs_agg, stat_errs_agg, total_errs_agg = [], [], [], []
        for energy in sorted(energies_map.keys()):
            vals = energies_map[energy]
            per_file_effs, per_file_stat_errs, per_file_syst_errs = [], [], []
            for v in vals:
                path, N, hole_status = v["path"], v.get("N"), v.get("hole")
                if not N or N <= 0: continue
                
                correction_factor = CORRECTION_FACTORS.get(hole_status, 1.0)
                k = unique_eventid_in_circle(path, radius_cm=radius_cm)
                p, err = compute_binomial(k, N)
                if p is None: continue
                
                p *= correction_factor
                if err is not None: err *= correction_factor
                
                eff_syst, stat_err_syst, syst_err = apply_syst(p, err, get_systematic_correction(energy))
                if eff_syst is None: continue
                
                per_file_effs.append(eff_syst)
                per_file_stat_errs.append(stat_err_syst)
                per_file_syst_errs.append(syst_err)

            if not per_file_effs: continue
            
            effs = np.array(per_file_effs)
            stat_errs = np.array(per_file_stat_errs)
            syst_errs = np.array(per_file_syst_errs)
            
            w = 1.0 / stat_errs**2 if np.all(stat_errs > 0) else np.ones_like(effs)
            if np.sum(w) == 0: w = np.ones_like(effs)
            eff_mean = np.sum(w * effs) / np.sum(w)
            stat_err_comb = 1.0 / np.sqrt(np.sum(w)) if np.all(stat_errs > 0) and np.sum(w) > 0 else (np.sqrt(np.sum(stat_errs**2))/len(stat_errs))
            syst_err_comb = np.sqrt(np.sum(syst_errs**2)) / len(syst_errs)

            energies.append(energy)
            effs_agg.append(eff_mean)
            stat_errs_agg.append(stat_err_comb)
            total_errs_agg.append(np.sqrt(stat_err_comb**2 + syst_err_comb**2))

        if energies:
            conc_data[mirror] = {"energies": np.array(energies), "effs": np.array(effs_agg), "stat_errs": np.array(stat_errs_agg), "total_errs": np.array(total_errs_agg)}

    if not conc_data: return

    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xlabel("Incident energy [keV]")
    ax.set_ylabel("Funnelling Efficiency [hits / N]")
    
    handles, labels = [], []

    for key in sorted(conc_data.keys()):
        data = conc_data[key]
        energies, effs, stat_errs, total_errs = data["energies"], data["effs"], data["stat_errs"], data["total_errs"]
        order = np.argsort(energies)
        energies, effs, stat_errs, total_errs = energies[order], effs[order], stat_errs[order], total_errs[order]
        
        color = colors.get(key, "k")
        # --- Plot only the total error band ---
        ax.fill_between(energies, np.maximum(0.0, effs - total_errs), effs + total_errs, color=color, alpha=0.2)
        line, = ax.plot(energies, effs, "-o", color=color, label=key)
        handles.append(line)
        labels.append(key)

    ax.set_xlim(50, 1050)
    ax.set_xticks(MAJOR_TICKS)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.grid(which='major', linestyle='--', alpha=0.6)
    ax.set_ylim(1e-4, 1e-3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend(handles=handles, labels=labels, title=legend_title, loc="lower right")
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, f"concentrator_circular_{fixed_col}_{fixed_val}_{int(radius_cm*10)}mm_syst.pdf")
    plt.savefig(fname, dpi=200, format='pdf')
    plt.close(fig)
    print("Saved", fname)

# Helper for circular plot with syst
def apply_syst(eff, err, syst):
    if syst and eff is not None and err is not None:
        c_bar_eta = syst['C_bar_eta']
        sigma_c = syst['sigma_C']
        eff_syst = eff * c_bar_eta
        err_stat_syst = err * c_bar_eta
        rel_syst_err = (sigma_c / c_bar_eta) if c_bar_eta != 0 else 0
        err_syst = eff_syst * rel_syst_err
        return eff_syst, err_stat_syst, err_syst
    return None, None, None


if __name__ == "__main__":
    # directories to scan — adjust if your data is elsewhere
    default_dirs = [
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ]

    print("Scanning directories:", default_dirs)
    entries = collect_entries(default_dirs)
    print(f"Found {len(entries)} entries")

    if not entries:
        print("No entries found. Check directory paths and that ROOT files match pattern 'output_*.root'.")
    else:
        # compact CSV summary to stdout
        print("basename,mirror,filter,hole,energy_keV,N,hits_union,eff,err")
        for e in entries:
            print(",".join([
                e["basename"], e["mirror"] or "", e["filter"] or "", e["hole"] or "",
                str(e["energy_keV"] or ""), str(e["N"] or ""), str(e["hits_union"]),
                (f"{e['eff']:.6g}" if e.get('eff') is not None else ""), (f"{e['err']:.6g}" if e.get('err') is not None else "")
            ]))

        concentrators = sorted({e["mirror"] for e in entries if e.get("mirror")})
        
        print("\n--- Generating plots WITHOUT systematic uncertainties ---")
        for conc in concentrators:
            plot_comparison(entries, comparison_col="filter", fixed_col="hole", fixed_val="withHole",
                            concentrator=conc, colors=COLOR_FILTER, legend_title="Filter")
            plot_comparison(entries, comparison_col="hole", fixed_col="filter", fixed_val="filterOff",
                            concentrator=conc, colors=COLOR_HOLE, legend_title="Opening")

        plot_concentrator_comparison(entries, fixed_col="hole", fixed_val="withHole",
                                     colors=COLOR_MIRROR, legend_title="Concentrator")
        plot_concentrator_comparison(entries, fixed_col="filter", fixed_val="filterOff",
                                     colors=COLOR_MIRROR, legend_title="Concentrator")
        plot_concentrator_circular(entries, fixed_col="hole", fixed_val="withHole", colors=COLOR_MIRROR, legend_title="Concentrator", radius_cm=1.0)
        plot_concentrator_circular(entries, fixed_col="filter", fixed_val="filterOff", colors=COLOR_MIRROR, legend_title="Concentrator", radius_cm=1.0)

        print("\n--- Generating plots WITH systematic uncertainties ---")
        for conc in concentrators:
            plot_comparison_with_syst(entries, comparison_col="filter", fixed_col="hole", fixed_val="withHole",
                                      concentrator=conc, colors=COLOR_FILTER, legend_title="Filter")
            plot_comparison_with_syst(entries, comparison_col="hole", fixed_col="filter", fixed_val="filterOff",
                                      concentrator=conc, colors=COLOR_HOLE, legend_title="Opening")
        
        plot_concentrator_comparison_with_syst(entries, fixed_col="hole", fixed_val="withHole",
                                               colors=COLOR_MIRROR, legend_title="Concentrator")
        plot_concentrator_comparison_with_syst(entries, fixed_col="filter", fixed_val="filterOff",
                                               colors=COLOR_MIRROR, legend_title="Concentrator")
        
        plot_concentrator_circular_with_syst(entries, fixed_col="hole", fixed_val="withHole", colors=COLOR_MIRROR, legend_title="Concentrator", radius_cm=1.0)
        plot_concentrator_circular_with_syst(entries, fixed_col="filter", fixed_val="filterOff", colors=COLOR_MIRROR, legend_title="Concentrator", radius_cm=1.0)


    print("Done.")