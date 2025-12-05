import os
import re
import numpy as np
import matplotlib.pyplot as plt
import uproot
import glob
from collections import defaultdict

# locate metadata
meta_paths = [
    "/home/frisoe/geant4/geant4-v11.3.1/examples/projects/exp_val/build/build/root/all_run_metadata.txt",
    "all_run_metadata.txt",
]
meta_path = next((p for p in meta_paths if os.path.exists(p)), meta_paths[0])
print("Using metadata:", meta_path)

# read metadata
entries = []
with open(meta_path) as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        kv = {}
        for part in ln.split():
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        try:
            kv["energy_keV"] = float(kv.get("energy_keV", 0.0))
            kv["angle_deg"]   = float(kv.get("angle_deg", 0.0))
            kv["elapsed_s"]   = float(kv.get("elapsed_s", 0.0))
            kv["nEvents"]     = int(float(kv.get("nEvents", 0)))
            kv["file"]        = kv.get("file", "")
            entries.append(kv)
        except Exception:
            continue

if not entries:
    raise SystemExit("No metadata entries found")

# scan entries, open root files and extract counts at chosen bin
aperture_solid_angle = 1.3e-6  # sr (same as used in your analysis)
records = []
for e in entries:
    path = e["file"]
    # Robust path resolution: try multiple candidate locations
    candidates = [
        path,
        os.path.join(os.path.dirname(meta_path), path),
        os.path.join(os.getcwd(), path),
        os.path.join(os.getcwd(), os.path.basename(path)),
        path.replace("build/root/", "build/build/root/"),
        os.path.join(os.path.dirname(meta_path), "..", path),
        os.path.abspath(path),
    ]
    resolved = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            resolved = cand
            break
    # If not found, search recursively in the metadata directory
    if resolved is None:
        search_dir = os.path.dirname(meta_path)
        fname = os.path.basename(path)
        matches = list(glob.glob(os.path.join(search_dir, '**', fname), recursive=True))
        if matches:
            resolved = matches[0]
    if resolved is None:
        print(f"Warning: file not found: {e['file']} (tried {candidates}), skipping")
        continue
    path = resolved
    # debug info
    # print(f"Using histogram file: {path}")

    try:
        with uproot.open(path) as f:
            hname = "Scattering angle 0-4 deg"
            if hname in f:
                h = f[hname]
            elif "Scattering angle" in f:
                h = f["Scattering angle"]
            else:
                print(f"No scattering hist in {path}, skipping")
                continue
            counts, edges = h.to_numpy()
            centers = 0.5 * (edges[:-1] + edges[1:])
    except Exception as ex:
        print("Failed to open/read", path, ex)
        continue

    # choose bin of interest: use the peak bin (most hits) for best statistical precision
    # option A: nearest to 2*incident angle (default)
    # target = 2.0 * e["angle_deg"]
    # idx = int(np.argmin(np.abs(centers - target)))

    # (alternative: choose peak bin)
    # idx = int(np.nanargmax(counts))
    # pick the bin with the maximum raw counts (highest S/N)
    idx = int(np.nanargmax(counts))
    target = float(centers[idx])

    cnt = float(counts[idx])
    Nsim = float(e["nEvents"])
    eff = cnt / (Nsim * aperture_solid_angle) if Nsim > 0 else 0.0
    sigma_abs = np.sqrt(max(cnt, 0.0)) / (Nsim * aperture_solid_angle) if Nsim > 0 else np.nan
    rel_prec = sigma_abs / eff if eff > 0 else np.nan

    records.append({
        "path": path,
        "energy_keV": e["energy_keV"],
        "angle_deg": e["angle_deg"],
        "nEvents": Nsim,
        "target_deg": target,
        "bin_center": centers[idx],
        "counts": cnt,
        "eff": eff,
        "sigma_abs": sigma_abs,
        "rel_prec": rel_prec,
        "elapsed_s": e.get("elapsed_s", 0.0),  # store elapsed time if present
    })

if not records:
    raise SystemExit("No valid histogram records found")

# --- New: total time summary ---
total_elapsed_s = sum(r.get("elapsed_s", 0.0) for r in records)
if total_elapsed_s > 0:
    total_hours = total_elapsed_s / 3600.0
    print(f"\nTotal simulated wall time (sum of elapsed_s): {total_elapsed_s:.1f} s  = {total_hours:.2f} hours")
else:
    print("\nTotal simulated wall time: no elapsed_s available in metadata")

# group by energy for plotting
by_energy = defaultdict(list)
for r in records:
    by_energy[r["energy_keV"]].append(r)

# prepare arrays for global fit of scaling rel ~ k / sqrt(N)
Ns = np.array([r["nEvents"] for r in records])
rels = np.array([r["rel_prec"] for r in records])
mask = np.isfinite(rels) & (rels > 0)
if mask.sum() >= 2:
    k_vals = rels[mask] * np.sqrt(Ns[mask])
    k_med = np.median(k_vals)
else:
    k_med = np.nan

# --- Print percent errors and threshold checks ---
thresholds = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
print("\nPer-run statistical precision (percent):")
# added "bin_count" column
print("energy_keV  angle_inc_deg  bin_deg   bin_count   nEvents     rel_err(%)   abs_sigma    pass<=1%  pass<=5%  pass<=10%")
for r in records:
    rel = r["rel_prec"]
    pct = 100.0 * rel if np.isfinite(rel) else np.nan
    abs_sig = r["sigma_abs"]
    bin_cnt = int(r.get("counts", 0))
    flags = [(pct <= 100.0*t) if np.isfinite(pct) else False for t in thresholds]
    print(f"{int(r['energy_keV']):7d}   {r['angle_deg']:12.3f}   {r['bin_center']:7.3f}   {bin_cnt:9d}   {int(r['nEvents']):9d}   {pct:10.3f}%   {abs_sig:12.3e}   "
          f"{'Y' if flags[0] else 'N':>5}     {'Y' if flags[1] else 'N':>5}     {'Y' if flags[2] else 'N':>5}")

# If we have k_med, compute required N for each threshold
if np.isfinite(k_med):
    print("\nEstimated N required to reach given relative precision (using median k):")
    for t in thresholds:
        Nreq = (k_med / t)**2
        print(f"  target {100.0*t:.1f}% -> N_required ~ {int(np.ceil(Nreq)):,d}")
else:
    print("\nCould not estimate N_required (insufficient data to compute k_med).")

# After records and k_med computed:

# choose target relative precision
rel_target = 0.05  # 5%

print("\nPer-run required N to reach target relative precision:")
print("energy  angle_inc  bin_deg  bin_count  N_current   rel_err(%)  N_required    extra_time_s   extra_time_h")

total_extra_s = 0.0
for r in records:
    rel = r["rel_prec"]
    if not np.isfinite(rel) or rel <= 0 or r["counts"] <= 0:
        print(f"{int(r['energy_keV']):6d}  {r['angle_deg']:8.3f}  {r['bin_center']:7.3f}   {int(r.get('counts',0)):8d}  {int(r['nEvents']):10d}   {'N/A':>9}   {'N/A':>12}   {'N/A':>12}   {'N/A':>12}")
        continue
    pct = 100.0 * rel
    Ncur = float(r["nEvents"])
    # N scales like (rel_current / rel_target)^2
    Nreq = int(np.ceil(Ncur * (rel / rel_target)**2))
    extra_time = float('nan')
    # try to locate elapsed_s from entries (match by file)
    elapsed = None
    for en in entries:
        if os.path.abspath(en.get("file","")) == os.path.abspath(r["path"]) or os.path.basename(en.get("file","")) == os.path.basename(r["path"]):
            elapsed = en.get("elapsed_s", None)
            break
    if elapsed:
        extra_time = elapsed * (Nreq / Ncur - 1.0)
    extra_hours = extra_time / 3600.0 if np.isfinite(extra_time) else float('nan')
    if np.isfinite(extra_time):
        total_extra_s += extra_time
    print(f"{int(r['energy_keV']):6d}  {r['angle_deg']:8.3f}  {r['bin_center']:7.3f}   {int(r['counts']):8d}  {int(Ncur):10d}   {pct:9.3f}%   {Nreq:12d}   {extra_time:12.1f}   {extra_hours:12.3f}")

# Print total extra time summary
if total_extra_s > 0:
    total_hours = total_extra_s / 3600.0
    print(f"\nEstimated total extra wall time to reach target precision (sum over runs): {total_extra_s:.1f} s = {total_hours:.2f} hours")
else:
    print("\nEstimated total extra wall time: no elapsed_s available to estimate extras")

# --- Plotting (relative precision in percent) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# left: relative precision (%) vs nEvents (log-log)
for energy, runs in sorted(by_energy.items()):
    runs = sorted(runs, key=lambda r: r["nEvents"])
    N = np.array([r["nEvents"] for r in runs])
    rel = np.array([r["rel_prec"] for r in runs])
    rel_pct = 100.0 * rel
    sc = ax1.scatter(N, rel_pct, s=50, label=f"{int(energy)} keV", alpha=0.9)
    for r in runs:
        if np.isfinite(r["rel_prec"]):
            # annotate with incident angle -> chosen bin center and bin count
            ax1.annotate(f"{r['angle_deg']:.2f}°→{r['bin_center']:.2f}° ({int(r['counts'])})",
                         (r["nEvents"], 100.0*r["rel_prec"]), fontsize=7, xytext=(3,3), textcoords="offset points")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("nEvents (incident protons)")
ax1.set_ylabel("Relative precision (%)")
ax1.set_title("Relative statistical precision vs sample size (percent)")

# overlay expected 1/sqrt(N) lines for thresholds using k_med (converted to percent)
if np.isfinite(k_med):
    Ngrid = np.logspace(np.log10(Ns.min()*0.9), np.log10(Ns.max()*1.1), 200)
    ax1.plot(Ngrid, 100.0 * k_med / np.sqrt(Ngrid), 'k--', label=f"~ {k_med:.3g}/sqrt(N) (scaled %)")
    # show horizontal lines at thresholds (%)
    for t in thresholds:
        ax1.axhline(100.0 * t, linestyle=':', linewidth=1.0, color='gray')
        ax1.text(Ngrid[0]*1.1, 100.0*t, f" {100.0*t:.0f}%", va='bottom', color='gray', fontsize=8)

ax1.legend()
ax1.grid(True, which="both", ls=":", alpha=0.6)

# right: absolute sigma vs nEvents (log-log), same as before
for energy, runs in sorted(by_energy.items()):
    runs = sorted(runs, key=lambda r: r["nEvents"])
    N = np.array([r["nEvents"] for r in runs])
    sig = np.array([r["sigma_abs"] for r in runs])
    ax2.scatter(N, sig, s=50, label=f"{int(energy)} keV", alpha=0.8)
    for r in runs:
        ax2.annotate(f"{r['angle_deg']:.2f}°→{r['bin_center']:.2f}° ({int(r['counts'])})",
                     (r["nEvents"], r["sigma_abs"]), fontsize=7, xytext=(3,3), textcoords="offset points")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("nEvents (incident protons)")
ax2.set_ylabel("Absolute sigma (sr^-1)")
ax2.set_title("Absolute statistical error vs sample size")
ax2.legend()
ax2.grid(True, which="both", ls=":", alpha=0.6)

plt.tight_layout()
plt.show()