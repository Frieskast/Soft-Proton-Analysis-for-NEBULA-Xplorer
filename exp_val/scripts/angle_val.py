import uproot
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import os
import glob
from collections import defaultdict

# --- MODIFICATION: Focus on a single directory ---
dir_ss   = "/home/frisoe/Desktop/exp/build/build/root/100kev/"
figures_dir = "/home/frisoe/Desktop/exp/figures/Angles"

# output plots directory
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)

# tolerant filename parsing regex (captures angle, energy, optional N token)
fname_pat = re.compile(r'output_([+-]?\d+(?:\.\d+)?)deg_([0-9]+)keV', re.IGNORECASE)

def parse_angle_energy_from_fname(fname):
    b = os.path.basename(fname)
    m = fname_pat.search(b)
    if not m:
        return None
    angle = float(m.group(1))
    energy = int(m.group(2))
    return energy, angle

def parse_energy_from_fname(fname):
    m = re.search(r'_(\d+(?:\.\d+)?)\s*(keV|MeV|eV)', fname, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == 'mev':
        return int(round(val * 1000.0))
    if unit == 'ev':
        return int(round(val * 1e-3))
    return int(round(val))

def parse_N_from_filename(fname, fallback=0):
    m = re.search(r'_N(\d+(?:\.\d+)?)([kKmM]?)', fname)
    if not m:
        return fallback
    val = float(m.group(1))
    suf = m.group(2).lower()
    if suf == 'k':
        return int(val * 1_000)
    if suf == 'm':
        # treat '17m' as 1.7e6 per user's earlier convention? keep straightforward -> 17e6
        return int(val * 1_000_000)
    return int(val)

def format_nEvents(n):
    if n >= 1_000_000:
        v = n / 1_000_000.0
        s = f"{v:.1f}m"
        if s.endswith(".0m"):
            s = s.replace(".0m", "m")
        return s
    if n >= 1_000:
        v = n // 1000
        return f"{v}k"
    return str(n)

# --- REMOVED: find_best_file function is no longer needed ---

# --- MODIFICATION: Group all files in the target directory by energy ---
all_files = glob.glob(os.path.join(dir_ss, "output_*.root"))
files_by_energy = defaultdict(list)
# --- ADDITION: Variable to store the incident angle for the title ---
incident_angle_for_title = None
for f in all_files:
    pe = parse_angle_energy_from_fname(f)
    if pe:
        energy_keV, angle_deg = pe
        files_by_energy[energy_keV].append(f)
        # --- ADDITION: Capture the incident angle from the first valid file ---
        if incident_angle_for_title is None:
            incident_angle_for_title = angle_deg

if not files_by_energy:
    raise SystemExit(f"No valid files found in source directory: {dir_ss}. Check paths and filenames.")

# --- REMOVED: CSV loading is no longer part of this plot's logic ---

# loader for histogram -> efficiency
def load_sim(root_path):
    if not os.path.exists(root_path):
        raise FileNotFoundError(root_path)
    with uproot.open(root_path) as f:
        h0_name = "Scattering angle 0-4 deg"
        counts = None
        bin_centers = None

        if h0_name in f:
            h0 = f[h0_name]
            c0, e0 = h0.to_numpy()
            counts = c0
            bin_centers = 0.5 * (e0[:-1] + e0[1:])
        elif "Scattering angle" in f:
            h = f["Scattering angle"]
            counts, edges = h.to_numpy()
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
        else:
            raise KeyError(f"No recognized scattering-angle histograms found in {root_path}")

    counts = np.asarray(counts, dtype=float)
    bin_centers = np.asarray(bin_centers, dtype=float)

    n = bin_centers.size
    if n == 0:
        bin_widths = np.array([], dtype=float)
    elif n == 1:
        bin_widths = np.array([0.0], dtype=float)
    elif n == 2:
        w = float(bin_centers[1] - bin_centers[0])
        bin_widths = np.array([w, w], dtype=float)
    else:
        bin_widths = np.empty(n, dtype=float)
        bin_widths[0] = float(bin_centers[1] - bin_centers[0])
        bin_widths[-1] = float(bin_centers[-1] - bin_centers[-2])
        bin_widths[1:-1] = 0.5 * (bin_centers[2:] - bin_centers[:-2])

    N_sim = parse_N_from_filename(os.path.basename(root_path), fallback=1000)
    aperture_solid_angle = 1.3e-6
    efficiency = counts / (N_sim * aperture_solid_angle)
    epsilon = 1e-12
    sigma_sim = np.sqrt(np.maximum(counts, 0.0) + epsilon) / (N_sim * aperture_solid_angle)
    return {
        "path": root_path,
        "counts": counts,
        "bin_centers": bin_centers,
        "bin_widths": bin_widths,
        "efficiency": efficiency,
        "sigma_sim": sigma_sim,
        "N_sim": N_sim
    }

# --- REMOVED: sim_avg_over_angle and interp_sim are no longer needed ---

# --- MODIFICATION: Define colors for each energy ---
energy_to_color = {
    100: "#DC143C",    # crimson red
    1000: "#4169E1"   # royal blue
}
energy_to_marker = {
    100: "^",
    1000: "o"
}

plt.rcParams.update({
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})



# --- MODIFICATION: Create a single plot for all data ---
print(f"Processing directory: {dir_ss}")
fig = plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.minorticks_on()
ax.grid(which='major', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
ax.grid(which='minor', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
ax.tick_params(axis="both", which="major", direction="in",length=8, width=2.0)
ax.tick_params(axis="both", which="minor", direction="in",length=4, width=1.5)
ax.set_axisbelow(True)

# Iterate over the energies found (e.g., 50, 1000)
for energy_keV in sorted(files_by_energy.keys()):
    print(f"  Plotting data for {energy_keV} keV...")
    
    # Get settings for this energy
    color = energy_to_color.get(energy_keV, 'k') # Default to black if energy not in map
    marker = energy_to_marker.get(energy_keV, 'x')
    legend_label = f"{energy_keV} keV"
    
    # Plot each file for this energy
    for i, fpath in enumerate(files_by_energy[energy_keV]):
        try:
            sim = load_sim(fpath)
            
            # Only apply the legend label to the first plot of this energy group
            label = legend_label if i == 0 else "_nolegend_"
            
            mask = (sim["bin_centers"] >= 0.0) & (sim["bin_centers"] <= 5.0) & (sim["efficiency"] > 0.0)
            x = sim["bin_centers"][mask]
            xerr = (sim.get("bin_widths", np.zeros_like(sim["bin_centers"]))[mask] / 2.0)
            y = sim["efficiency"][mask]
            yerr = sim["sigma_sim"][mask]

            order = np.argsort(x)
            x, y, xerr, yerr = x[order], y[order], xerr[order], yerr[order]

            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker, linestyle='none',
                         color=color, alpha=0.9, label=label, markersize=6)

            if x.size > 0:
                ax.plot(x, y, color=color, linestyle='-', alpha=0.6)

        except Exception as e:
            print(f"    Failed to load or plot {fpath}: {e}")

# --- REMOVED: Chi2 calculation and text box ---

ax.set_xlabel("Scattering angle [deg]")
ax.set_ylabel("Scattering efficiency [sr$^{-1}$]")
ax.set_yscale("log")
# --- MODIFICATION: Add the incident angle to the title ---
# if incident_angle_for_title is not None:
#     ax.set_title(f"Scattering Efficiency (Single Scattering) $\\theta_i$={incident_angle_for_title:.2f}$^\\circ$")
# else:
#     ax.set_title("Scattering Efficiency (Single Scattering)")
ax.set_xlim(0.0, 4.5)
ax.set_ylim(1e0, 1e4) # Adjusted ylim to potentially fit 1000 keV data
ax.legend()
plt.tight_layout()

out_pdf = os.path.join(figures_dir, "eff_100_and_1000_keV_comparison.pdf")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"Saved combined plot: {out_pdf}")
plt.close(fig)
