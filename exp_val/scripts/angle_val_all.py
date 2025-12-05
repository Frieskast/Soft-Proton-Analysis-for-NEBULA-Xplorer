import uproot
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import os
import glob
from collections import defaultdict
import sys
from matplotlib.patches import Patch

# --- Directories and Configuration ---
dir_ss   = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/"
dir_opt3 = "/home/frisoe/Desktop/exp/build/build/root/Physicslist3_final/"
dir_opt4 = "/home/frisoe/Desktop/exp/build/build/root/Physicslist4_final/"
figures_dir = "/home/frisoe/Desktop/exp/figures/Angles"

dirs = [
    (dir_opt3, "Multiple Scattering option 3"),
    (dir_opt4, "Multiple Scattering option 4"),
    (dir_ss,   "Single Scattering")
]

os.makedirs(figures_dir, exist_ok=True)

# CSV file containing experimental efficiency points
csv_file = os.path.join(os.path.dirname(__file__), "all_efficiencies_combined.csv")

# --- Plotting Style ---
plt.rcParams.update({
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "legend.title_fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

label_to_color = {
    "Multiple Scattering option 3": "#4169E1",
    "Multiple Scattering option 4": "#FFA500",
    "Single Scattering": "#DC143C"
}
label_to_marker = {
    "Multiple Scattering option 3": "o",
    "Multiple Scattering option 4": "s",
    "Single Scattering": "^"
}

# --- Helper Functions (Parsing, File Finding) ---
fname_pat = re.compile(r'output_([+-]?\d+(?:\.\d+)?)deg_([0-9]+)keV', re.IGNORECASE)
N_pat = re.compile(r'_N([0-9]*\.?[0-9]+[kKmM]?)')

def parse_angle_energy_from_fname(fname):
    b = os.path.basename(fname)
    m = fname_pat.search(b)
    if not m: return None, None
    return int(m.group(2)), float(m.group(1))

def parse_N_from_fname(fname):
    b = os.path.basename(fname)
    m = N_pat.search(b)
    if not m: return 0
    s = m.group(1)
    suf = s[-1].lower() if s and s[-1].isalpha() else ""
    mant = s[:-1] if suf in ("k", "m") else s
    try: val = float(mant)
    except Exception: return 0
    if suf == "k": val *= 1e3
    elif suf == "m": val *= 1e6
    return int(round(val))

def find_best_file_for(directory, energy_keV, theta_deg, target_N=None, tol_deg=1e-3):
    pattern = os.path.join(directory, "output_*.root")
    candidates = []
    for f in glob.glob(pattern):
        e, a = parse_angle_energy_from_fname(f)
        if e is None or e != energy_keV or abs(a - theta_deg) > tol_deg:
            continue
        N = parse_N_from_fname(f)
        candidates.append((N, f))
    if not candidates: return None
    if target_N:
        ge = sorted([c for c in candidates if c[0] >= target_N], key=lambda x: x[0])
        if ge: return ge[0][1]
        candidates.sort(key=lambda x: abs(x[0] - target_N))
        return candidates[0][1]
    candidates.sort(reverse=True)
    return candidates[0][1]

# --- Data Loading Functions ---
def load_csv_efficiency(csv_file):
    rows = []
    if not os.path.exists(csv_file): return rows
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((
                    int(round(float(row["energy_keV"]))),
                    float(row["inc_ang_deg"]),
                    float(row.get("inc_ang_err_deg", 0)),
                    float(row["scat_ang_deg"]),
                    float(row.get("scat_ang_err_deg", 0)),
                    float(row["efficiency_sr_inv"]),
                    float(row.get("efficiency_err_sr_inv", 0))
                ))
            except (ValueError, KeyError):
                continue
    return rows

def load_sim_efficiency(root_path):
    with uproot.open(root_path) as f:
        h_name = next((k for k in f.keys() if "Scattering angle" in k), None)
        if not h_name: raise KeyError("No scattering angle histogram found.")
        h = f[h_name]
        counts, edges = h.to_numpy()
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
    
    # --- ADDITION: Calculate bin widths for x-error bars ---
    n = bin_centers.size
    if n == 0:
        bin_widths = np.array([], dtype=float)
    elif n == 1:
        bin_widths = np.array([0.0], dtype=float)
    else:
        # Calculate widths based on distance to neighboring bin centers
        bin_widths = np.empty_like(bin_centers)
        bin_widths[0] = bin_centers[1] - bin_centers[0]
        bin_widths[-1] = bin_centers[-1] - bin_centers[-2]
        bin_widths[1:-1] = 0.5 * (bin_centers[2:] - bin_centers[:-2])

    N_sim = parse_N_from_fname(root_path)
    if N_sim == 0: N_sim = int(np.sum(counts))
    if N_sim == 0: raise ValueError("Cannot determine N_sim for normalization.")

    aperture_solid_angle = 1.3e-6
    efficiency = counts / (N_sim * aperture_solid_angle)
    sigma_sim = np.sqrt(np.maximum(counts, 1)) / (N_sim * aperture_solid_angle)
    
    return {
        "bin_centers": bin_centers,
        "bin_widths": bin_widths, # Add bin widths to the returned dictionary
        "efficiency": efficiency,
        "sigma_sim": sigma_sim,
    }

# --- Chi-squared Calculation Helper ---
def sim_avg_over_angle(sim_data, theta0_deg, sigma_theta_deg, n_sigma=4.0, npoints=101):
    if sigma_theta_deg is None or sigma_theta_deg <= 0:
        eff = np.interp(theta0_deg, sim_data["bin_centers"], sim_data["efficiency"])
        sig = np.interp(theta0_deg, sim_data["bin_centers"], sim_data["sigma_sim"])
        return eff, sig

    low, high = theta0_deg - n_sigma * sigma_theta_deg, theta0_deg + n_sigma * sigma_theta_deg
    thetas = np.linspace(low, high, npoints)
    weights = np.exp(-0.5 * ((thetas - theta0_deg) / sigma_theta_deg)**2)
    weights /= np.sum(weights)

    sim_eff_pts = np.interp(thetas, sim_data["bin_centers"], sim_data["efficiency"])
    sim_sig_pts = np.interp(thetas, sim_data["bin_centers"], sim_data["sigma_sim"])

    mean_avg = np.sum(weights * sim_eff_pts)
    sig_avg = np.sqrt(np.sum(weights**2 * sim_sig_pts**2))
    return mean_avg, sig_avg

# --- Main Script Logic ---
csv_rows = load_csv_efficiency(csv_file)
if not csv_rows:
    print(f"Warning: Could not load or parse experimental data from {csv_file}")

all_keys = set()
for d, label in dirs:
    for f in glob.glob(os.path.join(d, "output_*.root")):
        e, a = parse_angle_energy_from_fname(f)
        if e is not None:
            all_keys.add((e, round(a, 3)))

if not all_keys:
    print("No matching simulation files found. Exiting.")
    sys.exit(1)

# --- NEW: Data structures for systematic calculations ---
# Store ratios and weights: {sim_label: [(ratio, weight, exp_rel_err), ...]}
efficiency_systematics_data = defaultdict(list)

target_N_for = {
    "Multiple Scattering option 3": 100_000,
    "Multiple Scattering option 4": 100_000,
    "Single Scattering": 10_000
}

for energy_keV, theta_deg in sorted(all_keys):
    print(f"\nProcessing E={energy_keV} keV, Incident Angle={theta_deg:.3f} deg")

    chosen_files = []
    for dpath, label in dirs:
        target_N = target_N_for.get(label)
        f = find_best_file_for(dpath, energy_keV, theta_deg, target_N=target_N)
        if f:
            chosen_files.append({"label": label, "path": f})

    if not chosen_files:
        print("  No simulation files found for this combination. Skipping.")
        continue

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)

    sim_data_loaded = []
    for sim_info in chosen_files:
        try:
            sim_data = load_sim_efficiency(sim_info["path"])
            sim_data["label"] = sim_info["label"]
            sim_data_loaded.append(sim_data)
            
            lab, col, mk = sim_info["label"], label_to_color[sim_info["label"]], label_to_marker[sim_info["label"]]
            mask = sim_data["efficiency"] > 0
            
            # --- MODIFICATION: Use errorbar for simulation points ---
            x = sim_data["bin_centers"][mask]
            y = sim_data["efficiency"][mask]
            xerr = sim_data["bin_widths"][mask] / 2.0
            yerr = sim_data["sigma_sim"][mask]

            # Plot error bars without a connecting line first
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=mk, linestyle='none', color=col, label=lab, markersize=6)
            # Then plot a line connecting the points
            ax.plot(x, y, color=col, linestyle='-')

        except Exception as e:
            print(f"  ERROR loading {sim_info['path']}: {e}")

    matching_csv = [row for row in csv_rows if row[0] == energy_keV and abs(row[1] - theta_deg) < 1e-2]
    
    chi_lines = []
    if matching_csv:
        scat_ang = np.array([m[3] for m in matching_csv])
        scat_err = np.array([m[4] for m in matching_csv])
        eff_exp = np.array([m[5] for m in matching_csv])
        eff_err_exp = np.array([m[6] for m in matching_csv])

        # --- MODIFICATION: Sort points and add a filled error band for CSV data ---
        # Sort by scattering angle to ensure the line and fill are drawn correctly.
        sort_idx = np.argsort(scat_ang)
        scat_ang_sorted = scat_ang[sort_idx]
        eff_exp_sorted = eff_exp[sort_idx]
        eff_err_exp_sorted = eff_err_exp[sort_idx]

        # 1. Plot the original error bars and points.
        ax.errorbar(scat_ang, eff_exp, xerr=scat_err, yerr=eff_err_exp, fmt='D', color='black',
                    markerfacecolor="#808080", label="Diebold et al., (2015)", zorder=200)

        # 2. Add a grey line connecting the points.
        ax.plot(scat_ang_sorted, eff_exp_sorted, color='grey', linestyle='-', zorder=198)

        # 3. Add a filled grey area for the error band.
        lower_bound = eff_exp_sorted - eff_err_exp_sorted
        upper_bound = eff_exp_sorted + eff_err_exp_sorted
        ax.fill_between(scat_ang_sorted, lower_bound, upper_bound, color='grey', alpha=0.3, zorder=197)


        for sim in sim_data_loaded:
            sim_eff_conv = np.empty_like(scat_ang)
            sim_sig_conv = np.empty_like(scat_ang)
            for i, (sa, se) in enumerate(zip(scat_ang, scat_err)):
                sim_eff_conv[i], sim_sig_conv[i] = sim_avg_over_angle(sim, sa, se)

            combined_sigma = np.sqrt(eff_err_exp**2 + sim_sig_conv**2)
            valid = combined_sigma > 0
            n_valid = np.sum(valid)

            if n_valid > 0:
                diff = sim_eff_conv[valid] - eff_exp[valid]
                chi2 = np.sum((diff / combined_sigma[valid])**2)
                red_chi2 = chi2 / n_valid
                print(f"  {sim['label']}: chi2={chi2:.2f}, ndf={n_valid}, reduced={red_chi2:.2f}")
                chi_lines.append((sim["label"], red_chi2))

                # --- NEW: Calculate and store per-point ratios for systematics ---
                # Use only valid points where sim efficiency is positive
                calc_mask = (sim_eff_conv[valid] > 1e-9) & (eff_exp[valid] > 1e-9)
                if np.any(calc_mask):
                    ratios = eff_exp[valid][calc_mask] / sim_eff_conv[valid][calc_mask]
                    # For uniform weights, each point's weight is 1
                    weights = np.ones_like(ratios)
                    # Relative experimental error
                    rel_errs = eff_err_exp[valid][calc_mask] / eff_exp[valid][calc_mask]
                    
                    for r, w, e in zip(ratios, weights, rel_errs):
                        efficiency_systematics_data[sim["label"]].append((r, w, e))

            else:
                chi_lines.append((sim["label"], None))
    else:
        print("  No matching experimental data found.")
        for sim in sim_data_loaded:
            chi_lines.append((sim["label"], None))

    # --- Main Legend and Chi-squared Box ---
    # 1. Manually place the main legend in the top-right corner.
    main_legend = ax.legend(loc='upper right', edgecolor='black', bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(main_legend)

    # 2. Create the chi-squared box and place it directly under the main legend.
    if chi_lines:
        if energy_keV == 250:
            short_label = {
            "Multiple Scattering option 3": "MS opt3",
            "Multiple Scattering option 4": "MS opt4",
            "Single Scattering": "SS"
        }
        else:
            short_label = {
            "Multiple Scattering option 3": "MS opt3",
            "Multiple Scattering option 4": "MS opt4",
            "Single Scattering": "SS         "
        }
        chi_text_parts = []
        for label, red_chi2 in chi_lines:
            s_label = short_label.get(label, label)
            val_str = f"{red_chi2:.1f}" if red_chi2 is not None else "no data"
            chi_text_parts.append(f"{s_label}: red χ² = {val_str}")
        chi_text = "\n".join(chi_text_parts)
        
        # Use an invisible patch as a handle for our text-only legend
        proxy_artist = Patch(visible=False)
        
        chi_legend = ax.legend(
            [proxy_artist], [chi_text],
            loc='upper right',
            bbox_to_anchor=(1.0, 0.75), # Adjust the Y-value (e.g., 0.8) to position it below the main legend
            handlelength=0,
            handletextpad=0,
            frameon=True,
            facecolor='white',
            edgecolor='black'
        )
        # Adjust text properties within the new legend
        chi_legend.get_texts()[0].set_fontweight('normal')
        chi_legend.get_texts()[0].set_ha('left')


    ax.set_xlabel("Scattering angle [deg]")
    ax.set_ylabel("Scattering efficiency [sr$^{-1}$]")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 4.5)
    ax.set_ylim(1e1,1e4)
    plt.tight_layout()

    out_pdf = os.path.join(figures_dir, f"eff_{energy_keV}keV_{theta_deg:.2f}deg.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved plot: {out_pdf}")
    plt.close(fig)

# --- NEW: Calculate and print final systematic correction factors ---
print("\n" + "="*60)
print("Systematic Correction Factor for Efficiency (C_eta)")
print("="*60)

for label, data in sorted(efficiency_systematics_data.items()):
    if not data:
        print(f"\n--- {label} ---\n  No data points to calculate systematics.")
        continue

    ratios, weights, rel_errs = zip(*data)
    ratios = np.array(ratios)
    weights = np.array(weights)
    rel_errs = np.array(rel_errs)

    # Normalize weights so they sum to 1
    total_weight = np.sum(weights)
    if total_weight == 0:
        continue
    weights /= total_weight

    # Weighted average (C_bar_eta)
    c_bar_eta = np.sum(weights * ratios)

    # Weighted standard deviation (spread)
    spread = np.sqrt(np.sum(weights * (ratios - c_bar_eta)**2))
    
    # Max measurement error
    max_meas_err = np.max(rel_errs) if len(rel_errs) > 0 else 0.0
    # The systematic error on the ratio C is dominated by the experimental error
    # So we use the max relative experimental error on efficiency.
    sigma_c = max(spread, max_meas_err * c_bar_eta)


    print(f"\n--- {label} ---")
    print(f"  Correction Factor (C_bar_eta): {c_bar_eta:.4f}")
    print(f"  Systematic Uncertainty (sigma_C): {sigma_c:.4f}")
    print(f"    - Spread in ratios: {spread:.4f}")
    print(f"    - Max relative measurement error contribution: {max_meas_err * c_bar_eta:.4f}")
    print(f"  Relative Uncertainty (sigma_C / C_bar_eta): {sigma_c / c_bar_eta:.1%}")

print("\nAll plots generated.")
