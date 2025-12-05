import os
import numpy as np
import uproot
import csv
from collections import defaultdict, OrderedDict
import glob

# --- Configuration: update these to match your setup ---
speedtest_base = "build/build/root/Speedtest"   # as before
exp_csv = "all_efficiencies_combined.csv"       # CSV with experimental points
aperture_solid_angle = 1.3e-6  # sr, same convention as your analysis

# CSV format expected (modify parsing if different):
# Columns (header): energy_keV,scat_ang_deg,scat_ang_err,eff,eff_err
# If no header, adjust below accordingly.
def load_experimental(csv_path, plot_energy=None, theta_0=None, ang_tol=1e-2, energy_tol=1e-6):
    """
    Parse CSV using DictReader. If plot_energy is provided, return a list of
    rows matching that energy and (optionally) incident angle theta_0.
    Otherwise return a dict keyed by int(energy) -> list of (scat_ang, scat_ang_err, eff, eff_err).
    Accepts column names like:
      energy_keV or energy
      inc_ang_deg or inc_ang or angle_deg
      scat_ang_deg or scat_ang
      scat_ang_err_deg or scat_ang_err
      efficiency_sr_inv or efficiency
      efficiency_err_sr_inv or eff_err or efficiency_err
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    def get_first(keys, row, default=None):
        for k in keys:
            if k in row and row[k] != "":
                return row[k]
        return default

    grouped = defaultdict(list)
    filtered = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # if no header, reader.fieldnames may be None; handle separately
        for row in reader:
            try:
                energy = float(get_first(['energy_keV', 'energy'], row, row.get(reader.fieldnames[0], None)))
                inc_ang = float(get_first(['inc_ang_deg', 'inc_ang', 'angle_deg', 'inc_ang_deg'], row, row.get(reader.fieldnames[1], 0.0)))
                scat_ang = float(get_first(['scat_ang_deg', 'scat_ang', 'scattering_angle'], row, row.get(reader.fieldnames[3], None)))
                scat_ang_err = float(get_first(['scat_ang_err_deg', 'scat_ang_err', 'scat_ang_err'], row, '0.0'))
                eff = float(get_first(['efficiency_sr_inv', 'efficiency', 'eff'], row, row.get(reader.fieldnames[5], 0.0)))
                eff_err = float(get_first(['efficiency_err_sr_inv', 'eff_err', 'efficiency_err'], row, '0.0'))
            except Exception:
                # skip rows we cannot parse
                continue

            if plot_energy is not None:
                if not np.isclose(energy, plot_energy, atol=energy_tol):
                    continue
                if theta_0 is not None and not np.isclose(inc_ang, theta_0, atol=ang_tol):
                    continue
                filtered.append((scat_ang, scat_ang_err, eff, eff_err, inc_ang))
            else:
                grouped[int(round(energy))].append((scat_ang, scat_ang_err, eff, eff_err))

    if plot_energy is not None:
        return filtered
    # sort grouped lists by scattering angle
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda x: x[0])
    return grouped

# helper functions must be defined before they are used
def resolve_file(path, base_dir):
    """Try several candidate paths and a recursive search under base_dir."""
    if not path:
        return None
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(base_dir, path),
        os.path.abspath(path),
        os.path.join(base_dir, os.path.basename(path)),
        os.path.join(os.getcwd(), os.path.basename(path)),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    # recursive search for the basename under base_dir
    matches = glob.glob(os.path.join(base_dir, '**', os.path.basename(path)), recursive=True)
    return matches[0] if matches else None

def load_hist_counts(root_path):
    """Open ROOT file, read 'Scattering angle 0-4 deg' (fallback) and return (counts, centers)."""
    try:
        with uproot.open(root_path) as f:
            hname = "Scattering angle 0-4 deg"
            if hname in f:
                h = f[hname]
            elif "Scattering angle" in f:
                h = f["Scattering angle"]
            else:
                return None
            counts, edges = h.to_numpy()
            centers = 0.5 * (edges[:-1] + edges[1:])
            return np.asarray(counts, dtype=float), np.asarray(centers, dtype=float)
    except Exception:
        return None

# find metadata files (O3,O4,SS) and parse like compute.py
option_map = OrderedDict([("O3","Option 3"), ("O4","Option 4"), ("SS","Single Scattering")])
records = defaultdict(list)  # records[(option_label, energy)] -> list of metadata entries

for sub, lab in option_map.items():
    meta_path = os.path.join(speedtest_base, sub, "all_run_metadata.txt")
    if not os.path.exists(meta_path):
        print(f"Warning: metadata not found for {lab} at {meta_path}, skipping")
        continue
    with open(meta_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            kv = {}
            for part in ln.split():
                if "=" in part:
                    k,v = part.split("=",1); kv[k]=v
            try:
                energy = int(round(float(kv.get("energy_keV", 0.0))))
                angle = float(kv.get("angle_deg", 0.0))
                nEvents = int(float(kv.get("nEvents", 0)))
                elapsed_s = float(kv.get("elapsed_s", 0.0))
                filep = kv.get("file","")
                records[(lab,energy)].append({"angle":angle,"nEvents":nEvents,"elapsed_s":elapsed_s,"file":filep})
            except Exception:
                continue

if not records:
    raise SystemExit("No metadata found in Speedtest directories.")

exp_data = load_experimental(exp_csv)

# --- Diagnostic prints ---
print("Experimental energies found:", sorted(exp_data.keys()))
sim_keys = sorted({energy for (_, energy) in records.keys()})
print("Simulation (option,energy) keys parsed:", sim_keys)
# count per option/energy
from collections import Counter
cnt = Counter((opt, en) for (opt, en) in records.keys())
print("Parsed simulation entries per (option,energy):")
for k in sorted(records.keys()):
    print("  ", k, " ->", len(records[k]), "runs")

# track unresolved files/hist missing counts while processing
missing_file_warn = 0
missing_hist_warn = 0

# compute chi2 per (option,energy)
results = []
for (opt_label, energy), runs in records.items():
    if energy not in exp_data:
        # diagnostic: no experimental data at this energy
        # print once for this energy
        print(f"Note: no experimental points for energy {energy} keV -> skipping {opt_label}")
        continue

    # get experimental points for this energy
    exp_pts = exp_data.get(energy, [])
    if not exp_pts:
        continue
    # for chi2 we want a single simulation per option+energy: if multiple runs exist, sum counts and sum nEvents
    # sum histogram counts from all files listed
    total_counts = None; total_centers = None; total_nEvents = 0
    for r in runs:
        filep = r["file"]
        resolved = resolve_file(filep, speedtest_base)
        if not resolved:
            missing_file_warn += 1
            # print(f"Warning: sim file not found for {opt_label} {energy}: {filep}")
            continue
        loaded = load_hist_counts(resolved)
        if loaded is None:
            missing_hist_warn += 1
            # print(f"Warning: missing hist in {resolved}")
            continue
        counts, centers = loaded
        if total_counts is None:
            total_counts = counts.copy()
            total_centers = centers.copy()
        else:
            if total_centers.shape != centers.shape or not np.allclose(total_centers, centers, atol=1e-8):
                print(f"Warning: histogram binning mismatch for {resolved}; skipping addition")
                continue
            total_counts += counts
        total_nEvents += r["nEvents"]
    if total_counts is None or total_nEvents == 0:
        continue
    # compute sim efficiency and sim sigma per bin
    eff_sim_bins = total_counts / (total_nEvents * aperture_solid_angle)
    sigma_sim_bins = np.sqrt(np.maximum(total_counts,0.0)) / (total_nEvents * aperture_solid_angle)
    # interpolate sim eff and sigma to experimental angles
    sim_eff_interp = lambda ang: np.interp(ang, total_centers, eff_sim_bins, left=np.nan, right=np.nan)
    sim_sig_interp = lambda ang: np.interp(ang, total_centers, sigma_sim_bins, left=np.nan, right=np.nan)
    # compute chi2 over experimental points
    chi2 = 0.0; n_points = 0
    for (a, aerr, eff_exp, eff_err) in exp_pts:
        sim_eff = sim_eff_interp(a)
        sim_sig = sim_sig_interp(a)
        if np.isnan(sim_eff) or np.isnan(sim_sig):
            # skip if outside sim angle range
            continue
        sigma_tot = np.sqrt(eff_err**2 + sim_sig**2)
        if sigma_tot <= 0:
            continue
        chi2 += (sim_eff - eff_exp)**2 / (sigma_tot**2)
        n_points += 1
    if n_points == 0:
        continue
    red_chi2 = chi2 / max(1, n_points - 1)  # ndf = n_points - number of fit params (0->1)
    results.append({"option":opt_label,"energy":energy,"chi2":chi2,"ndf":n_points-1,"red_chi2":red_chi2,"n_points":n_points})

# print and rank
if not results:
    print("No chi2 comparisons possible (no overlapping data).")
else:
    print("\nChi2 summary (lower better).")
    print("option               energy  n_points   chi2    ndf   red_chi2")
    for r in sorted(results, key=lambda x: (x["energy"], x["red_chi2"])):
        print(f"{r['option']:<20s} {r['energy']:6d}  {r['n_points']:8d}  {r['chi2']:8.3f}  {r['ndf']:4d}  {r['red_chi2']:10.3f}")

    # rank per energy
    from collections import defaultdict
    by_e = defaultdict(list)
    for r in results:
        by_e[r["energy"]].append(r)
    print("\nBest option per energy (by lowest reduced chi2):")
    for e, lst in sorted(by_e.items()):
        best = min(lst, key=lambda x: x["red_chi2"])
        print(f"Energy {e} keV -> {best['option']} (red_chi2={best['red_chi2']:.3f})")

# after loop, print summary:
print(f"\nDiagnostics: missing files: {missing_file_warn}, missing histograms: {missing_hist_warn}")