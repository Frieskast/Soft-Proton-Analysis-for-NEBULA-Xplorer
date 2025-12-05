import os
import re
import glob
import sys
import csv
import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# --- directories / options (from your request) ---
dir_opt3 = "/home/frisoe/Desktop/exp/build/build/root/Physicslist3_final"
dir_opt4 = "/home/frisoe/Desktop/exp/build/build/root/Physicslist4_final"
dir_ss   = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final"
figures_dir = "/home/frisoe/Desktop/exp/figures/Energies"

dirs = [
    (dir_opt3, "Multiple Scattering option 3"),
    (dir_opt4, "Multiple Scattering option 4"),
    (dir_ss,   "Single Scattering")
]

os.makedirs(figures_dir, exist_ok=True)

# move style config out of the loop (set once)

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

# Toggle: when True plot count-based efficiency = counts / N_total with sqrt(N) errors
USE_SQRTN_ERRORS = False

label_to_color = {
    "Multiple Scattering option 3": "#4169E1",   # royal blue
    "Multiple Scattering option 4": "#FFA500",   # bright orange
    "Single Scattering": "#DC143C"               # crimson
}
label_to_marker = {
    "Multiple Scattering option 3": "o",
    "Multiple Scattering option 4": "s",
    "Single Scattering": "^"
}

# CSV file containing experimental energy-loss points (angle, value, errors)
csv_file = os.path.join(os.path.dirname(__file__), "all_energyLoss_combined.csv")

# --- filename parsing helpers ---
fname_pat = re.compile(r'output_([+-]?\d+(?:\.\d+)?)deg_([0-9]+)keV', re.IGNORECASE)
# allow decimal mantissa and optional k/K or m/M suffix, e.g. _N1.1m or _N500k
N_pat = re.compile(r'_N([0-9]*\.?[0-9]+[kKmM]?)')

def parse_angle_energy_from_fname(fname):
    b = os.path.basename(fname)
    m = fname_pat.search(b)
    if not m:
        return None, None
    angle = float(m.group(1))
    energy = int(m.group(2))
    return energy, angle

def parse_N_from_fname(fname):
    b = os.path.basename(fname)
    m = N_pat.search(b)
    if not m:
        return 0
    s = m.group(1)
    # detect suffix and mantissa
    suf = s[-1].lower() if s and s[-1].isalpha() else ""
    mant = s[:-1] if suf in ("k", "m") else s
    try:
        val = float(mant)
    except Exception:
        return 0
    if suf == "k":
        val *= 1e3
    elif suf == "m":
        val *= 1e6
    return int(round(val))

# find best file in a directory for given energy/angle with a target N preference
def find_best_file_for(directory, energy_keV, theta_deg, target_N=None, tol_deg=1e-3):
    pattern = os.path.join(directory, "output_*.root")
    candidates = []
    # debug print when inspecting 500 keV searches
    if energy_keV == 500:
        print(f"DEBUG: searching directory='{directory}' pattern='{pattern}' for E={energy_keV} keV, theta={theta_deg}, target_N={target_N}, tol={tol_deg}")
    for f in glob.glob(pattern):
        e, a = parse_angle_energy_from_fname(f)
        if e is None:
            continue
        if e != energy_keV:
            continue
        if abs(a - theta_deg) > tol_deg:
            continue
        N = parse_N_from_fname(f)
        candidates.append((N, f))
        if energy_keV == 500:
            print(f"DEBUG: candidate found: file='{os.path.basename(f)}' parsed_angle={a}, parsed_energy={e}, parsed_N={N}")
    if not candidates:
        if energy_keV == 500:
            print(f"DEBUG: no candidates in '{directory}' for E={energy_keV}, theta={theta_deg}")
        return None
    # prefer N >= target (smallest above target), otherwise pick closest to target (or largest if no target)
    if target_N:
        ge = [c for c in candidates if c[0] >= target_N]
        if ge:
            ge.sort(key=lambda x: x[0])
            if energy_keV == 500:
                print(f"DEBUG: selecting (>=target) {os.path.basename(ge[0][1])} with N={ge[0][0]}")
            return ge[0][1]
        # otherwise pick candidate minimizing abs(N-target)
        candidates.sort(key=lambda x: abs(x[0] - target_N))
        if energy_keV == 500:
            print(f"DEBUG: selecting (closest to target) {os.path.basename(candidates[0][1])} with N={candidates[0][0]}")
        return candidates[0][1]
    # no target: pick largest N
    candidates.sort(reverse=True)
    if energy_keV == 500:
        print(f"DEBUG: selecting (largest N) {os.path.basename(candidates[0][1])} with N={candidates[0][0]}")
    return candidates[0][1]

# --- CSV loader ---
def load_csv_energy(csv_file):
    rows = []
    if not os.path.exists(csv_file):
        return rows
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                energy = int(round(float(row.get("energy_keV", "nan"))))
                inc_ang = float(row.get("inc_ang_deg", "nan"))
                scat_ang = float(row.get("scat_ang_deg", "nan"))
                scat_err = float(row.get("scat_ang_err_deg", "0"))
                eloss = float(row.get("energyloss_keV", "nan"))
                eloss_err = float(row.get("energyloss_err_keV", "0"))
            except Exception:
                continue
            rows.append((energy, inc_ang, scat_ang, scat_err, eloss, eloss_err))
    return rows

csv_rows = load_csv_energy(csv_file)

# --- binned stats and sim convolution helpers (copied/adapted) ---
def binned_stats(x, y, bins):
    inds = np.digitize(x, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.full(len(centers), np.nan)
    stderrs = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        sel = inds == i
        cnt = sel.sum()
        counts[i] = cnt
        if cnt > 0:
            vals = y[sel]
            means[i] = vals.mean()
            # standard error of the mean: use ddof=1 when possible, else mark nan
            if cnt > 1:
                stdev = vals.std(ddof=1)
                stderrs[i] = stdev / np.sqrt(cnt)
            else:
                # single entry: cannot compute sample stdev -> leave as NaN to indicate missing uncertainty
                stderrs[i] = np.nan
    return centers, means, stderrs, counts

def interp_sim_mean(sim, theta_deg):
    eff = np.interp(theta_deg, sim["bin_centers"], sim["means"],
                    left=sim["means"][0], right=sim["means"][-1])
    sig = np.interp(theta_deg, sim["bin_centers"], sim["stderrs"],
                    left=sim["stderrs"][0], right=sim["stderrs"][-1])
    return eff, sig

def sim_avg_over_angle(sim, theta0_deg, sigma_theta_deg, npoints=101, n_sigma=4.0):
    if sigma_theta_deg is None or sigma_theta_deg <= 0.0 or not np.isfinite(sigma_theta_deg):
        return interp_sim_mean(sim, theta0_deg)
    low = theta0_deg - n_sigma * sigma_theta_deg
    high = theta0_deg + n_sigma * sigma_theta_deg
    thetas = np.linspace(low, high, npoints)
    weights = np.exp(-0.5 * ((thetas - theta0_deg) / sigma_theta_deg) ** 2)
    s = weights.sum()
    if s <= 0:
        return interp_sim_mean(sim, theta0_deg)
    weights /= s
    sim_means_pts = np.interp(thetas, sim["bin_centers"], sim["means"],
                              left=sim["means"][0], right=sim["means"][-1])
    sim_sig_pts = np.interp(thetas, sim["bin_centers"], sim["stderrs"],
                            left=sim["stderrs"][0], right=sim["stderrs"][-1])
    mean_avg = float(np.sum(weights * sim_means_pts))
    sig_avg = float(np.sqrt(np.sum((weights ** 2) * (sim_sig_pts ** 2))))
    return mean_avg, sig_avg

# --- scan directories and build grouping of available (energy, angle) combos ---
files_by_dir = {}
for d, label in dirs:
    files_by_dir[label] = sorted(glob.glob(os.path.join(d, "output_*.root")))

# build set of all (energy, angle) found across all dirs
all_keys = set()
for label, flist in files_by_dir.items():
    for f in flist:
        e, a = parse_angle_energy_from_fname(f)
        if e is None:
            continue
        all_keys.add((e, round(a, 3)))  # round key to 1e-3 deg to avoid tiny float diffs

if not all_keys:
    print("No matching files found in directories.")
    sys.exit(1)

# --- MODIFICATION: Add dictionary to store correction factors ---
# Structure: {sim_label: [(ratio, weight, exp_rel_err), ...]}
energyloss_systematics_data = defaultdict(list)


# targets for preferred N
target_N_for = {
    "Multiple Scattering option 3": 100_000,
    "Multiple Scattering option 4": 100_000,
    "Single Scattering": 10_000
}

# iterate keys sorted
for energy_keV, theta_deg in sorted(all_keys):
    # --- DEBUG PRINT: Announce processing for 500 keV ---
    if energy_keV == 500:
        print("\n" + "-"*80)
        print(f"DEBUG [500 keV]: Processing E={energy_keV} keV, Incident Angle={theta_deg:.3f} deg")
        print("-"*80)
    else:
        print(f"Processing E={energy_keV} keV  inc={theta_deg} deg")


    # find best file per label
    chosen = []
    for dpath, label in dirs:
        # allow substitutions for certain 500 keV angles to use nearby available files
        theta_lookup = theta_deg
        substituted = False
        if energy_keV == 500:
            # map: requested_theta -> substitute_theta
            sub_map = {
                0.89: 0.89,   # already had
                1.02: 1.06,   # use 1.02 for 1.06
                1.19: 1.23    # use 1.19 for 1.23
            }
            tol = 0.01
            for req_theta, sub_theta in sub_map.items():
                if abs(theta_deg - req_theta) < tol:
                    theta_lookup = sub_theta
                    substituted = True
                    break

        target = target_N_for.get(label, None)
        f = find_best_file_for(dpath, energy_keV, theta_lookup, target_N=target, tol_deg=1e-3)
        if f:
            chosen.append((label, f))
            msg = f"  Selected for {label}: {os.path.basename(f)} (N={parse_N_from_fname(f)})"
            if substituted:
                 msg += f"  [used files for {theta_lookup:.2f}deg instead of {theta_deg:.2f}deg]"
            # print(msg)
    if not chosen:
        print("  No root files for any option — skipping (no CSV-only).")
        continue

    # load each chosen file and compute binned energy-loss stats
    sims = []
    for label, fpath in chosen:
        try:
            with uproot.open(fpath) as rf:
                tree_key = None
                for k, obj in rf.items():
                    try:
                        if hasattr(obj, "keys") and "energy_keV" in obj.keys():
                            tree_key = k
                            break
                    except Exception:
                        continue
                if tree_key is None:
                    print("    no valid tree in", fpath); continue
                tree = rf[tree_key]
                energies = np.asarray(tree["energy_keV"].array(library="np"), dtype=float)
                theta = np.asarray(tree["theta_deg"].array(library="np"), dtype=float) if "theta_deg" in tree.keys() else None
        except Exception as ex:
            print("    error reading", fpath, ex)
            continue
        if theta is None:
            print("    no theta in", fpath); continue

        energy_loss = energy_keV - energies

        bins = np.linspace(0.0, 6.0, 30)
        centers, means, stderrs, counts = binned_stats(theta, energy_loss, bins)
        # compute half-widths for each bin center to use as horizontal error (bins may be uniform)
        bin_half_widths = 0.5 * np.diff(bins)
        # ensure shape matches centers
        if bin_half_widths.size == centers.size - 1:
            # for uniform bins np.diff yields len(centers), but handle general case by padding last
            bin_half_widths = np.concatenate([bin_half_widths, bin_half_widths[-1:]])
        elif bin_half_widths.size < centers.size:
            # pad with last value
            pad = centers.size - bin_half_widths.size
            bin_half_widths = np.concatenate([bin_half_widths, np.full(pad, bin_half_widths[-1] if bin_half_widths.size>0 else 0.0)])

        # total events read / parse from filename (fallback to actual event count)
        N_total = parse_N_from_fname(fpath)
        if not N_total:
            N_total = energies.size if 'energies' in locals() else int(np.nansum(counts))
        sims.append({
             "label": label,
             "path": fpath,
             "bin_centers": centers,
             "means": means,
             "stderrs": stderrs,
             "counts": counts
            ,"N_total": N_total
           ,"bin_half_widths": bin_half_widths
         })

    if not sims:
        continue

    # prepare plot
    fig, ax = plt.subplots(figsize=(9,6))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.4, color='gray', alpha=0.4)
    ax.tick_params(axis="both", which="major", direction="in",length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", direction="in",length=3, width=1.0)
    
    ax.set_axisbelow(True)

    # plot sims
    for sim in sims:
        lab = sim["label"]
        col = label_to_color.get(lab, "black")
        mk = label_to_marker.get(lab, "o")
        x = sim["bin_centers"]
        # choose what to plot: energy-loss means with sample standard error,
        # or count-based efficiency with Poisson sqrt(N) errors.
        if USE_SQRTN_ERRORS:
            Ntot = sim.get("N_total", 0) or 0
            counts = sim["counts"].astype(float)
            if Ntot > 0:
                y = counts / float(Ntot)
                yerr = np.sqrt(counts) / float(Ntot)
            else:
                y = counts * 0.0
                yerr = counts * 0.0
        else:
            y = sim["means"]
            yerr = sim["stderrs"]

        # filter out bins below the incident angle (theta_i)
        tol_angle = 1e-6
        valid_bins = x >= (theta_deg - tol_angle)
        # --- MODIFICATION: Always plot simulation data, even if valid_bins is empty ---
        if np.any(valid_bins):
            x_plot = x[valid_bins]
            y_plot = y[valid_bins]
            yerr_plot = yerr[valid_bins]
        else:
            x_plot = x
            y_plot = y
            yerr_plot = yerr

        # if uncertainties are NaN (e.g. single-event bins), suppress errorbars for those points
        yerr_plot = np.copy(yerr_plot)
        nan_mask = np.isnan(yerr_plot)
        if np.any(nan_mask):
            yerr_plot[nan_mask] = 0.0

        # sort x just in case and plot
        order = np.argsort(x_plot)
        x_plot = x_plot[order]; y_plot = y_plot[order]; yerr_plot = yerr_plot[order]
        bin_half = sim.get("bin_half_widths", np.zeros_like(x_plot))
        xerr_plot = bin_half if bin_half.size == x_plot.size else np.zeros_like(x_plot)
        ax.errorbar(x_plot, y_plot, xerr=xerr_plot, yerr=yerr_plot, fmt=mk, linestyle='-', color=col,
                    markerfacecolor=col, markeredgecolor='white', markeredgewidth=1.0,
                    capsize=0, label=lab)

    # --- CSV overlay and chi2 calculation ---
    # --- MODIFICATION: Add a substitution rule for CSV data matching ---
    csv_theta_lookup = theta_deg
    if energy_keV == 500 and abs(theta_deg - 1.23) < 1e-2:
        csv_theta_lookup = 1.19
        print(f"  INFO: For plot angle {theta_deg:.2f}°, using experimental data from {csv_theta_lookup:.2f}°.")
    elif energy_keV == 500 and abs(theta_deg - 1.06) < 1e-2:
        csv_theta_lookup = 1.02
        print(f"  INFO: For plot angle {theta_deg:.2f}°, using experimental data from {csv_theta_lookup:.2f}°.")
    elif energy_keV == 500 and abs(theta_deg - 0.89) < 1e-2:
        csv_theta_lookup = 0.85
        print(f"  INFO: For plot angle {theta_deg:.2f}°, using experimental data from {csv_theta_lookup:.2f}°.")

    # Match CSV data using the (potentially substituted) lookup angle
    matching = [(scat, serr, eloss, eloss_err) for (e, inc_ang, scat, serr, eloss, eloss_err) in csv_rows
                if e == energy_keV and abs(inc_ang - csv_theta_lookup) < 1e-2]

    # --- DEBUG PRINT: Show how many CSV points were matched ---
    if energy_keV == 500:
        print(f"DEBUG [500 keV]: Found {len(matching)} matching CSV points for incident angle ~{theta_deg:.3f} deg.")

    chi_lines = []
    if matching:
        scat_ang = np.array([m[0] for m in matching])
        scat_err = np.array([m[1] for m in matching])
        eloss = np.array([m[2] for m in matching])
        eloss_err = np.array([m[3] for m in matching])

        # filter CSV points below incident angle
        tol_angle = 1e-6
        valid_csv = scat_ang >= (theta_deg - tol_angle)

        # --- DEBUG PRINT: Show how many points remain after filtering ---
        if energy_keV == 500:
            print(f"DEBUG [500 keV]: {np.sum(valid_csv)} of {len(scat_ang)} CSV points have scattering angle >= incident angle.")

        if not np.any(valid_csv):
            print("  CSV points present but all are below incident angle -> skipping CSV/chi2 for this combo")
            # still draw legend, but no CSV plotting/chi2
            for sim in sims:
                chi_lines.append((sim["label"], None, None, 0, label_to_color.get(sim["label"], 'black')))
        else:
            scat_ang = scat_ang[valid_csv]
            scat_err = scat_err[valid_csv]
            eloss = eloss[valid_csv]
            eloss_err = eloss_err[valid_csv]

            # filled exp band behind
            ax.fill_between(scat_ang, eloss - eloss_err, eloss + eloss_err, color='grey', alpha=0.25, zorder=1)
            # CSV: line connecting points + markers, include error bars (no caps)
            csv_face = "#808080"
            ax.errorbar(
                scat_ang, eloss,
                xerr=scat_err, yerr=eloss_err,
                fmt='-D',
                color='black',
                markerfacecolor=csv_face,
                markeredgecolor='black',
                markersize=8,
                markeredgewidth=1.0,
                elinewidth=1.2,
                ecolor='black',
                capsize=0,
                label="Diebold et al., (2015)",
                zorder=200
            )

            # compute reduced chi2 per sim (fold sim over angular uncertainty) using the filtered CSV points
            for sim in sims:
                sim_eff = np.empty_like(scat_ang, dtype=float)
                sim_sig = np.empty_like(scat_ang, dtype=float)
                for j, theta_c in enumerate(scat_ang):
                    mean_avg, sig_avg = sim_avg_over_angle(sim, theta_c, scat_err[j], npoints=121, n_sigma=4.0)
                    sim_eff[j] = mean_avg
                    # if sim uncertainty unknown (nan), treat as zero in combined sigma
                    sim_sig[j] = 0.0 if np.isnan(sig_avg) else sig_avg
                combined_sigma = np.sqrt(np.maximum(eloss_err**2, 0.0) + np.maximum(sim_sig**2, 0.0))
                valid = combined_sigma > 0
                n_valid = int(np.count_nonzero(valid))

                # --- DEBUG PRINT: Show values for chi-squared calculation ---
                if energy_keV == 500:
                    print(f"  DEBUG [500 keV, {sim['label']}]:")
                    print(f"    - Num valid points for chi2 (n_valid): {n_valid}")
                    if n_valid > 0:
                        print(f"    - Exp E_loss [keV]: {np.round(eloss[valid], 2)}")
                        print(f"    - Sim E_loss [keV]: {np.round(sim_eff[valid], 2)}")
                        print(f"    - Combined σ [keV]: {np.round(combined_sigma[valid], 2)}")

                # --- MODIFICATION: Calculate and store correction factors ---
                if n_valid > 0:
                    # Ensure sim_eff is not zero to avoid division by zero
                    sim_eff_valid = sim_eff[valid]
                    eloss_valid = eloss[valid]
                    
                    # Calculate point-wise correction factors where sim_eff is positive
                    cf_mask = (sim_eff_valid > 1e-9) & (eloss_valid > 1e-9)
                    if np.any(cf_mask):
                        ratios = eloss_valid[cf_mask] / sim_eff_valid[cf_mask]
                        # For uniform weights, each point's weight is 1
                        weights = np.ones_like(ratios)
                        # Relative experimental error
                        rel_errs = eloss_err[valid][cf_mask] / eloss_valid[cf_mask]

                        for r, w, e in zip(ratios, weights, rel_errs):
                            energyloss_systematics_data[sim["label"]].append((r, w, e))

                if n_valid > 0:
                    diff = sim_eff[valid] - eloss[valid]
                    chi2 = float(np.sum((diff / combined_sigma[valid])**2))
                    red_chi2 = chi2 / float(n_valid)
                else:
                    chi2 = float('nan'); red_chi2 = float('nan')
                chi_lines.append((sim["label"], chi2, red_chi2, n_valid, label_to_color.get(sim["label"], 'black')))
                print(f"  {sim['label']}: chi2={chi2:.3f}, ndf={n_valid}, reduced={red_chi2:.3f}")
    else:
        # --- MODIFICATION: Still add legend and chi2 info for simulation, even if no CSV ---
        for sim in sims:
            chi_lines.append((sim["label"], None, None, 0, label_to_color.get(sim["label"], 'black')))
        print("  No CSV points for this (E,inc)")

    # --- MODIFIED: Manually place legends to avoid overlap ---
    # 1. Place the main legend in the bottom-right corner.
    main_legend = ax.legend(
        loc='lower right',
        edgecolor='black',
        bbox_to_anchor=(1.0, 0.00),  # Anchor near the bottom-right
        framealpha=0.9,
    )
    ax.add_artist(main_legend)  # Fix the main legend in place

    # 2. Prepare the text for the chi-squared box.
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
    chi_lines_text = []
    for slabel, chi2v, redv, npts, col in chi_lines:
        s = short_label.get(slabel, slabel)
        if chi2v is None or not np.isfinite(redv):
            chi_lines_text.append(f"{s}: no data")
        else:
            chi_lines_text.append(f"{s}: red χ²={redv:.0f}")
    chi_text = "\n".join(chi_lines_text)

    # 3. Create and place the chi-squared box above the main legend.
    from matplotlib.patches import Patch
    proxy_artist = Patch(visible=False)

    chi_legend = ax.legend(
        [proxy_artist], [chi_text],
        loc='lower right',
        bbox_to_anchor=(1.00, 0.35),  # Adjust Y to position it above the main legend
        handlelength=0,
        handletextpad=0,
        fontsize=18,
        frameon=True,
        edgecolor='black',
        facecolor='white',
        framealpha=0.9
    )
    chi_legend.get_texts()[0].set_fontweight('normal')
    chi_legend.get_texts()[0].set_ha('left')


    ax.set_xlabel("Scattering angle [deg]")
    ax.set_ylabel("Energy loss [keV]")
    # ax.set_title(rf"Energy loss ($E_{{inc}}$={energy_keV} keV, $\theta_i$={theta_deg:.2f}$^\circ$)")
    ax.set_yscale("log", base=10)
    # For incident angles 1 and above, use xlim 1 to 7; otherwise, use 0 to 6
    if theta_deg >= 1.0:
        ax.set_xlim(0.0, 6.0)
    else:
        ax.set_xlim(0.0, 6.0)
    ax.set_ylim(1e-1, 5e2)
    ax.grid(True, which='both', alpha=0.6)

    out_pdf = os.path.join(figures_dir, f"energy_loss_{energy_keV}keV_{theta_deg:.2f}deg.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

# --- NEW: Calculate and print final systematic correction factors ---
print("\n" + "="*60)
print("Systematic Correction Factor for Energy Loss (R_delta_E)")
print("="*60)

for label, data in sorted(energyloss_systematics_data.items()):
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

    # Weighted average (R_bar_delta_E)
    r_bar_delta_e = np.sum(weights * ratios)

    # Weighted standard deviation (spread)
    spread = np.sqrt(np.sum(weights * (ratios - r_bar_delta_e)**2))
    
    # Max measurement error
    max_meas_err = np.max(rel_errs) if len(rel_errs) > 0 else 0.0
    sigma_r = max(spread, max_meas_err * r_bar_delta_e)

    print(f"\n--- {label} ---")
    print(f"  Correction Factor (R_bar_delta_E): {r_bar_delta_e:.4f}")
    print(f"  Systematic Uncertainty (sigma_R): {sigma_r:.4f}")
    print(f"    - Spread in ratios: {spread:.4f}")
    print(f"    - Max relative measurement error contribution: {max_meas_err * r_bar_delta_e:.4f}")
    print(f"  Relative Uncertainty (sigma_R / R_bar_delta_E): {sigma_r / r_bar_delta_e:.1%}")

print("\n" + "="*50)