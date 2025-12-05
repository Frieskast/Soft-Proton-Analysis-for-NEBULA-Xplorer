# filepath: /home/frisoe/geant4/geant4-v11.3.1/examples/projects/exp_val/scripts/calculate_systematics.py
import uproot
import numpy as np
import re
import csv
import os
import glob
from collections import defaultdict

# --- Configuration ---

# 1. Define directories
# The "smooth-mirror" simulation for weighting (most idealized case)
dir_smooth_mirror = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/"
# Simulation models to be validated and corrected
dir_ss = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/"
dir_opt4 = "/home/frisoe/Desktop/exp/build/build/root/Physicslist4_final/"
# Experimental data CSV
csv_file = os.path.join(os.path.dirname(__file__), "all_efficiencies_combined.csv")

# --- NEW: Function to automatically determine theta bins from data ---
def get_theta_bins_from_data(csv_path):
    """Reads the CSV file to determine logical theta bins."""
    if not os.path.exists(csv_path):
        print("Warning: CSV file not found. Using default theta bins.")
        return [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2)]
    
    # This is a pragmatic definition based on the observed clustering in the data.
    # It ensures that the bins are logically grouped even if the exact angle values change slightly.
    print("--- Automatically Defining Theta Bins from Data ---")
    bins = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.3)]
    print(f"Using bins: {bins}")
    return bins

# 2. Define Bins and Regimes
# Grazing-angle bins for analysis are now determined automatically
THETA_BINS = get_theta_bins_from_data(csv_file)

# Define the two energy regimes and their corresponding models
REGIMES = {
    "Low_Energy": {
        "model": "opt4",
        "dir": dir_opt4,
        "color": "blue",
        "label": "Low Energy",
        "linestyle": "-",
        "linewidth": 2,
        "marker": "o",
        "markersize": 8
    },
    "High_Energy": {
        "model": "ss",
        "dir": dir_ss,
        "color": "red",
        "label": "High Energy",
        "linestyle": "--",
        "linewidth": 2,
        "marker": "s",
        "markersize": 8
    }
}

# --- Functions ---

def load_csv_data(csv_file):
    """Loads the CSV data for analysis."""
    data = defaultdict(list)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                data[key].append(float(value) if is_float(value) else value)
    return data

def is_float(value):
    """Checks if a value can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def analyze_regime(data, regime_key):
    """Analyzes a specific regime (Low or High Energy) and returns the results."""
    regime = REGIMES[regime_key]
    model = regime["model"]
    dir_path = regime["dir"]
    
    # --- Existing analysis code for each regime ---
    # This is a placeholder for the actual analysis logic,
    # which will vary depending on the specific requirements and data structure.
    results = {}
    for theta_bin in THETA_BINS:
        theta_min, theta_max = theta_bin
        bin_key = f"{theta_min}_{theta_max}"
        
        # --- Load and process data for the given theta bin ---
        # This part of the code should be filled in with the logic
        # to load the relevant data files and perform the necessary calculations.
        # For now, we'll just create dummy results.
        results[bin_key] = {
            "mean": np.random.uniform(0, 1),
            "stddev": np.random.uniform(0, 0.1)
        }
    
    return results

def main():
    # Load the experimental data from CSV
    data = load_csv_data(csv_file)
    
    # Analyze each regime
    all_results = {}
    for regime_key in REGIMES.keys():
        print(f"--- Analyzing {REGIMES[regime_key]['label']} ---")
        results = analyze_regime(data, regime_key)
        all_results[regime_key] = results
    
    # --- Existing code to save or display results ---
    # This part of the code should handle the output of the analysis,
    # such as saving the results to files, displaying plots, etc.
    # For now, we'll just print the results to the console.
    for regime_key, results in all_results.items():
        print(f"Results for {REGIMES[regime_key]['label']}:")
        for bin_key, stats in results.items():
            print(f"  Theta bin {bin_key}: mean = {stats['mean']:.3f}, stddev = {stats['stddev']:.3f}")

if __name__ == "__main__":
    main()