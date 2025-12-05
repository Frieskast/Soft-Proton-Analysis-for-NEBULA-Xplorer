import os
import re
import glob
import argparse
import math
import numpy as np
import uproot
import pandas as pd

# --- Constants for Fluence Calculation (from calculate_flux.py) ---
# Using CL95, 45-deg orbit as a default scenario for this calculation
CL95_45_deg = {100: 1.12E+08, 250: 1.16E+08, 500: 1.21E+08, 1000: 1.25E+08}
DEFAULT_FLUENCE_MAP = CL95_45_deg

CORRECTION_FACTORS = {
    "withOpening": 0.92131,
    "withoutOpening": 0.92022
}

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
    """Parses mirror type, filter status, and energy from a filename."""
    base = os.path.basename(fname)
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else "N/A"
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else "N/A"
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else 0
    N = parse_N_from_name(base)
    # Detect hole status from the full path
    hole = "withOpening" if "withhole" in fname.lower() else ("withoutOpening" if "withouthole" in fname.lower() else "N/A")
    return mirror, filter_flag, hole, energy, N

def calculate_mean_energies(dirs):
    """
    Scans ROOT files, calculates the mean energy of detected protons, and returns a list of results.
    """
    # --- Constants for fluence calculation ---
    r_out_mm = 47.0
    aperture_area_cm2 = math.pi * (r_out_mm / 10.0)**2
    x_aperture_map = {"DPH": 901.0, "DCC": 901.0, "SP": 851.0}

    results = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Warning: Directory not found: {d}")
            continue
        pattern = os.path.join(d, "**", "output_*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        
        for fpath in files:
            try:
                mirror, filter_flag, hole, incident_energy, N = parse_metadata_from_name(fpath)
                
                # Exclude 50 keV incident energy as requested
                if incident_energy == 50:
                    continue
                
                with uproot.open(fpath) as f:
                    target_tree_name = "SmallDetSummary"
                    if target_tree_name not in f:
                        continue

                    tree = f[target_tree_name]
                    # Geant4 default energy is MeV. Using 'entryE' as requested.
                    energies_MeV = tree.arrays(["entryE"], library="np")["entryE"]
                    
                    num_hits = len(energies_MeV)
                    if num_hits > 0:
                        mean_energy = np.mean(energies_MeV) 
                        std_dev_energy = np.std(energies_MeV)
                    else:
                        mean_energy = 0.0
                        std_dev_energy = 0.0

                    # --- Fluence Calculation Logic ---
                    total_fluence = 0.0
                    if N is not None and N > 0:
                        funnelling_eff = num_hits / float(N)
                        x_aperture_mm = x_aperture_map.get(mirror)
                        mission_fluence = DEFAULT_FLUENCE_MAP.get(incident_energy, 0.0)
                        correction_factor = CORRECTION_FACTORS.get(hole, 1.0)

                        if x_aperture_mm is not None:
                            theta_max = math.atan(r_out_mm / x_aperture_mm)
                            a_eff_cm2 = aperture_area_cm2 * 0.25 * (math.sin(theta_max)**2)
                            total_fluence = mission_fluence * a_eff_cm2 * funnelling_eff * correction_factor

                    results.append({
                        "Incident Energy [keV]": incident_energy,
                        "Mirror": mirror,
                        "Filter": filter_flag,
                        "Hole": hole,
                        "Hits": num_hits,
                        "Mean Energy [keV]": mean_energy,
                        "Std Dev [keV]": std_dev_energy,
                        "Total Fluence": total_fluence,
                    })
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the mean energy of protons reaching the detector.")
    parser.add_argument("--dirs", "-d", nargs="+", default=[
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ], help="Directories to scan for ROOT files.")
    args = parser.parse_args()

    print("Calculating mean detected energy for all configurations...")
    
    all_results = calculate_mean_energies(args.dirs)

    if not all_results:
        print("No valid data found; cannot generate report.")
    else:
        df = pd.DataFrame(all_results)
        df.sort_values(by=["Incident Energy [keV]", "Mirror", "Filter", "Hole"], inplace=True)
        
        # Set pandas display options for a clean printout
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', '{:.2f}'.format)

        # --- Update float format for the new fluence column ---
        formatters = {
            'Mean Energy [keV]': '{:.2f}'.format,
            'Std Dev [keV]': '{:.2f}'.format,
            'Total Fluence': '{:.2e}'.format,
        }

        print("\n" + "="*120)
        print("Mean Energy and Total Fluence of Protons at Detector (Scenario: CL95, 45-deg Orbit)")
        print("="*120)
        print(df.to_string(index=False, formatters=formatters))
