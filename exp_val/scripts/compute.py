import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

# Configure base Speedtest dirs (try a couple of likely locations)
candidates = [
    "build/build/root/Speedtest",
    "build/root/Speedtest",
    "/home/frisoe/Desktop/exp/build/build/root/Speedtest",
    "/home/frisoe/geant4/geant4-v11.3.1/examples/projects/exp_val/build/build/root/Speedtest",
]
base_dir = next((p for p in candidates if os.path.isdir(p)), candidates[0])
print("Using Speedtest base dir:", base_dir)

# Map subfolders -> legend labels
option_map = OrderedDict([
    ("O3", "Option 3"),
    ("O4", "Option 4"),
    ("SS", "Single Scattering"),
])

# Read all metadata for each option
records = []  # list of dicts: {option, energy_keV, angle_deg, nEvents, elapsed_s, file}
for sub, label in option_map.items():
    meta_path = os.path.join(base_dir, sub, "all_run_metadata.txt")
    if not os.path.exists(meta_path):
        # try without subfolder (legacy)
        alt = os.path.join(base_dir, f"all_run_metadata_{sub}.txt")
        if os.path.exists(alt):
            meta_path = alt
    if not os.path.exists(meta_path):
        print(f"Warning: metadata not found for {label} (tried {meta_path}), skipping")
        continue

    with open(meta_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            entry = {"option": label}
            for part in ln.split():
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                entry[k] = v
            # convert
            try:
                entry["energy_keV"] = float(entry.get("energy_keV", 0.0))
                entry["angle_deg"] = float(entry.get("angle_deg", 0.0))
                entry["elapsed_s"] = float(entry.get("elapsed_s", 0.0))
                entry["nEvents"] = int(float(entry.get("nEvents", 0)))
                # Only accept runs with exactly 10k events
                if entry["nEvents"] != 10000:
                    continue
                entry["file"] = entry.get("file", "")
                # normalized CPU time per 1e6 events
                entry["time_per_1e6"] = entry["elapsed_s"] / max(1.0, entry["nEvents"]) * 1e6
                records.append(entry)
            except Exception:
                print("Skipping malformed line in", meta_path, ":", ln)
                continue

if not records:
    raise SystemExit("No metadata records found. Check Speedtest directories and filenames.")

# Group records by energy_keV (sorted)
by_energy = defaultdict(list)
energies = sorted({int(r["energy_keV"]) for r in records})
for r in records:
    by_energy[int(r["energy_keV"])].append(r)

# Print a summary table
print("\nSummary (per-run) -- time normalized to 1e6 events:")
print("option           energy_keV  angle_deg  nEvents    elapsed_s   time_per_1e6_s   file")
for r in sorted(records, key=lambda x: (x["option"], x["energy_keV"], x["angle_deg"])):
    print(f"{r['option']:<15s} {int(r['energy_keV']):10d}  {r['angle_deg']:8.3f}  {int(r['nEvents']):8d}  {r['elapsed_s']:10.3f}   {r['time_per_1e6']:12.3f}   {os.path.basename(r.get('file',''))}")

# Compute median time_per_1e6 per option and normalize to Option 3 = 1.0
opt_times = defaultdict(list)
for r in records:
    opt_times[r["option"]].append(r["time_per_1e6"])

median_per_opt = {}
for opt, tlist in opt_times.items():
    median_per_opt[opt] = float(np.median(tlist)) if tlist else float('nan')

# baseline = Option 3 median (if missing, pick first available)
baseline_label = "Option 3"
if baseline_label not in median_per_opt or not np.isfinite(median_per_opt[baseline_label]):
    if median_per_opt:
        baseline_label = next(iter(median_per_opt.keys()))
        print(f"Warning: Option 3 not found, using {baseline_label} as baseline.")
    else:
        raise SystemExit("No option timings found to normalize against.")
baseline = median_per_opt[baseline_label]
print("\nMedian time_per_1e6 (s) per option and normalized to Option 3 = 1.0:")
print("option               median_s     normalized_to_Option3")
for opt, med in median_per_opt.items():
    norm = med / baseline if baseline > 0 else float('nan')
    print(f"{opt:<20s} {med:12.3f}     {norm:12.3f}")

# Plot: one subplot per energy, comparing options over angle, normalized to Option 3
n_eps = len(energies)
fig, axes = plt.subplots(n_eps, 1, figsize=(8, 3*n_eps), sharex=False, squeeze=False)
axes = axes.flatten()

colors = {"Option 3": "tab:blue", "Option 4": "tab:green", "Single Scattering": "tab:orange"}

for i, energy in enumerate(energies):
    ax = axes[i]
    runs = by_energy[energy]
    # group by option
    opt_groups = defaultdict(list)
    for r in runs:
        opt_groups[r["option"]].append(r)
    for opt_label in option_map.values():
        grp = sorted(opt_groups.get(opt_label, []), key=lambda x: x["angle_deg"])
        if not grp:
            continue
        angles = np.array([g["angle_deg"] for g in grp])
        times = np.array([g["time_per_1e6"] for g in grp])
        # normalize to Option 3 baseline
        times_norm = times / baseline
        nEvents = np.array([g["nEvents"] for g in grp])
        sizes = 40.0 * np.sqrt(nEvents / np.maximum(1, nEvents.max()))
        ax.plot(angles, times_norm, marker='o', linestyle='-', label=opt_label, color=colors.get(opt_label))
        ax.scatter(angles, times_norm, s=sizes, color=colors.get(opt_label), edgecolor='k', zorder=3)
    # ax.set_yscale("log")
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("CPU time (Option3=1.0)")
    ax.set_title(f"Energy = {energy} keV")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.show()