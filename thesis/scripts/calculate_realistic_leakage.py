import os
import re
import glob
import argparse
import math
import numpy as np
import uproot
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- Apply consistent plotting style from calculate_flux.py ---
plt.rcParams.update({
    "axes.labelsize": 22,
    "axes.titlesize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 18,
    "legend.title_fontsize": 22,
    "figure.labelsize": 18,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

COLOR_MIRROR = {
    "SP":  "#4169E1",   # royal blue
    "DCC": "#2ca02c",   # green
    "DPH": "#DC143C",   # crimson
}

ORBIT_LABEL = {"45-deg": "45\u00b0", "98-deg": "98\u00b0"}
def orbit_label(orbit):
    return ORBIT_LABEL.get(orbit, orbit.replace("-deg", "\u00b0"))


# --- Physical and Detector Constants ---
ALPHA_DAMAGE_RATE = 1e-17  # Current-related damage rate (A/cm)
# DETECTOR_THICKNESS_CM is no longer the primary 'd', it will be calculated per energy.
INITIAL_LEAKAGE_CURRENT_A = 2.3e-12
INITIAL_FWHM_RESOLUTION_EV = 127.4
DETECTOR_RADIUS_CM = 0.4
DETECTOR_AREA_CM2 = np.pi * DETECTOR_RADIUS_CM**2

# --- Define noise components for resolution calculation ---
FANO_NOISE_EV = 118.0
SERIES_NOISE_EV = 30.1
ELEC_NOISE_240K = 48.0

# --- Fluence and Geometry Constants ---
FLUENCE_MAPS_ALL = {
    'CL95': {
        "45-deg": {100: 2.69746044e11, 250: 2.05480469e11, 500: 1.41679924e11, 1000: 6.26134501e10},
        "98-deg-AP9": {100: 6.92936322e11, 250: 4.52721202e11, 500: 2.29025363e11, 1000: 5.24018357e10},
        "98-deg-SP": {100: 2.321100e12, 250: 5.304100e11, 500: 1.702800e11, 1000: 5.484100e10}
    },
    'CL75': {
        "45-deg": {100: 1.575964e11, 250: 1.220416e11, 500: 7.381142e10, 1000: 3.236780e10},
        "98-deg-AP9": {100: 3.113434e11, 250: 2.047505e11, 500: 1.081152e11, 1000: 2.757775e10},
        "98-deg-SP": {100: 3.747300e12, 250: 6.560000e11, 500: 1.171120e11, 1000: 4.482700e10}
    },
    'CL50': {
        "45-deg": {100: 9.373315e10, 250: 7.207972e10, 500: 5.209968e10, 1000: 2.478172e10},
        "98-deg-AP9": {100: 2.163474e11, 250: 1.382338e11, 500: 6.787237e10, 1000: 1.903805e10},
        "98-deg-SP": {100: 4.887900e12, 250: 7.113000e11, 500: 1.606100e11, 1000: 3.644600e10}
    }
}
# Energy bin widths in MeV for fluence correction (from calculate_flux.py)
ENERGY_BIN_WIDTHS_MEV = {
    100: 0.108,
    250: 0.196,
    500: 0.353,
    1000: 0.707
}
CORRECTION_FACTORS = {"withOpening": 0.92131, "withoutOpening": 0.92022}


def load_and_prepare_interpolator(csv_path, key_x, key_y, x_multiplier=1.0):
    """Generic function to load a CSV and create a linear interpolation function."""
    try:
        df = pd.read_csv(csv_path)
        df[key_x] = df[key_x] * x_multiplier
        df.sort_values(key_x, inplace=True)
        
        interpolator = interp1d(
            df[key_x], df[key_y], kind='linear', bounds_error=False, 
            fill_value=(df[key_y].iloc[0], df[key_y].iloc[-1])
        )
        return interpolator
    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_path}")
        return None

def parse_metadata_from_name(fname):
    """Parses metadata from a filename, including the number of primary particles N."""
    base = os.path.basename(fname)
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else "N/A"
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else "N/A"
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else 0
    
    m_N = re.search(r"_N(\d+)([kmKM])?", base)
    N = None
    if m_N:
        n_val = int(m_N.group(1))
        suf = m_N.group(2).lower() if m_N.group(2) else ''
        if suf == "k": N = n_val * 1000
        elif suf == "m": N = n_val * 1000000
        else: N = n_val
        
    hole = "withOpening" if "withhole" in fname.lower() else "withoutOpening"
    return mirror, filter_flag, hole, energy, N

def calculate_base_metrics(dirs, kappa_interpolator, range_interpolator):
    """Calculates environment-independent metrics for each simulation configuration."""
    r_out_mm = 47.0
    aperture_area_cm2 = math.pi * (r_out_mm / 10.0)**2
    x_aperture_map = {"DPH": 901.0, "DCC": 901.0, "SP": 851.0}
    
    results = []
    for d in dirs:
        pattern = os.path.join(d, "**", "output_*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        
        for fpath in files:
            try:
                mirror, filt, hole, inc_E, N = parse_metadata_from_name(fpath)
                if inc_E == 50 or N is None: continue

                with uproot.open(fpath) as f:
                    if "SmallDetSummary" not in f: continue
                    energies_keV = f["SmallDetSummary/entryE"].array(library="np")
                    
                    num_hits = len(energies_keV)
                    if num_hits == 0: continue

                    mean_detected_energy = np.mean(energies_keV)
                    funnelling_eff = num_hits / float(N)
                    x_aperture_mm = x_aperture_map.get(mirror)
                    
                    if x_aperture_mm is None: continue
                    theta_max = math.atan(r_out_mm / x_aperture_mm)
                    a_eff_cm2 = aperture_area_cm2 * 0.25 * (math.sin(theta_max)**2)
                    interpolated_kappa = kappa_interpolator(mean_detected_energy)
                    # Calculate projected range 'd' in cm
                    projected_range_um = range_interpolator(mean_detected_energy)
                    projected_range_cm = projected_range_um * 1e-4 # convert um to cm

                    results.append({
                        "Incident_Energy": inc_E, "Mirror": mirror, "Filter": filt, "Hole": hole,
                        "Mean_Detected_Energy": mean_detected_energy,
                        "Funnelling_Efficiency": funnelling_eff,
                        "A_eff_cm2": a_eff_cm2,
                        "Correction_Factor": CORRECTION_FACTORS.get(hole, 1.0),
                        "Interpolated_Kappa": interpolated_kappa,
                        "Projected_Range_cm": projected_range_cm
                    })
            except Exception as e:
                print(f"Failed to process {fpath}: {e}")
    return pd.DataFrame(results)

def plot_degradation_bar_charts(df, cl_level, figures_dir="/home/frisoe/Desktop/Thesis/figures/degradation_analysis"):
    """
    Generates grouped bar charts to compare resolution degradation for each orbit.
    """
    # Style is now set globally, so local update is removed.
    
    # Create a combined configuration category for the x-axis
    df['config'] = df['Filter'] + ' / ' + df['Hole']
    
    # Define the order for the x-axis categories
    config_order = [
        'filterOn / withOpening', 'filterOn / withoutOpening',
        'filterOff / withOpening', 'filterOff / withoutOpening'
    ]
    
    mirror_order = ['DCC', 'DPH', 'SP']

    for orbit in df['orbit'].unique():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        orbit_df = df[df['orbit'] == orbit]
        
        # --- Manual Matplotlib bar chart implementation ---
        x = np.arange(len(config_order))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        for mirror in mirror_order:
            offset = width * multiplier
            
            # Filter data for the current mirror and get values in the correct order
            subset = orbit_df[orbit_df['Mirror'] == mirror]
            data_map = pd.Series(subset.Degradation_eV.values, index=subset.config).to_dict()
            values = [data_map.get(cat, 0) for cat in config_order]

            rects = ax.bar(x + offset, values, width, label=mirror, color=COLOR_MIRROR.get(mirror), edgecolor='black', linewidth=1.2)
            ax.bar_label(rects, fmt='%.1f', fontsize=14, padding=3)
            
            multiplier += 1
        
        # --- Adjust x-ticks and labels for manual plot ---
        ax.set_xticks(x + width, [
            'Filter On /\nWith Opening', 
            'Filter On /\nWithout Opening',
            'Filter Off /\nWith Opening', 
            'Filter Off /\nWithout Opening'
        ])

        # ax.set_xlabel("Configuration (Filter / Opening)")
        ax.set_ylabel("Total Degradation [eV]")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # Ensure grid is behind bars
        ax.legend(title='Concentrator Type', loc="lower right", framealpha=1.0)
        ax.yaxis.set_minor_locator(MultipleLocator(5))

        fig.tight_layout()

        # --- Save the figure ---
        try:
            os.makedirs(figures_dir, exist_ok=True)
            fname_base = os.path.join(figures_dir, f"degradation_barchart_{orbit.replace('-', '')}_{cl_level}")
            fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
            print(f"Saved bar chart to {fname_base}.pdf/.png")
        except Exception as e:
            print(f"Failed to save bar chart for {orbit}: {e}")
        
        # --- Show the plot ---
        plt.show()
        plt.close()

def plot_final_resolution_charts(df, cl_level, figures_dir="/home/frisoe/Desktop/Thesis/figures/degradation_analysis"):
    """
    Generates grouped bar charts showing the final resolution against performance thresholds.
    """
    config_order = [
        'filterOn / withOpening', 'filterOn / withoutOpening',
        'filterOff / withOpening', 'filterOff / withoutOpening'
    ]
    mirror_order = ['DCC', 'DPH', 'SP']

    for orbit in df['orbit'].unique():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        orbit_df = df[df['orbit'] == orbit]
        
        x = np.arange(len(config_order))
        width = 0.25
        multiplier = 0

        for mirror in mirror_order:
            offset = width * multiplier
            
            subset = orbit_df[orbit_df['Mirror'] == mirror]
            data_map = pd.Series(subset.New_FWHM_eV.values, index=subset.config).to_dict()
            values = [data_map.get(cat, 0) for cat in config_order]

            rects = ax.bar(x + offset, values, width, label=mirror, color=COLOR_MIRROR.get(mirror), edgecolor='black', linewidth=1.2)
            ax.bar_label(rects, fmt='%.1f', fontsize=14, padding=3)
            
            multiplier += 1
        
        # Add threshold and initial resolution lines
        ax.axhline(150, color='red', linestyle='--', linewidth=2.5, label='<150 eV Requirement')
        ax.axhline(INITIAL_FWHM_RESOLUTION_EV, color='blue', linestyle=':', linewidth=2.5, label='Initial Resolution (127.4 eV)')

        # --- Add a shaded red region above the 150 eV threshold ---
        ax.axhspan(150, 165, facecolor='red', alpha=0.15)

        ax.set_ylabel("Final FWHM Resolution [eV]")
        ax.set_ylim(125, 155)
        ax.set_xticks(x + width, [
            'Filter On /\nWith Opening', 
            'Filter On /\nWithout Opening',
            'Filter Off /\nWith Opening', 
            'Filter Off /\nWithout Opening'
        ])
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(title='Legend', loc="upper right", framealpha=1.0)
        ax.yaxis.set_minor_locator(MultipleLocator(2.5))

        fig.tight_layout()

        try:
            os.makedirs(figures_dir, exist_ok=True)
            fname_base = os.path.join(figures_dir, f"final_resolution_barchart_{orbit.replace('-', '')}_{cl_level}")
            fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
            print(f"Saved final resolution chart to {fname_base}.pdf/.png")
        except Exception as e:
            print(f"Failed to save final resolution chart for {orbit}: {e}")
        
        plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate realistic leakage current for multiple scenarios using interpolated hardness factors and projected range.")
    parser.add_argument("--dirs", "-d", nargs="+", default=["/home/frisoe/Desktop/Root/withhole/", "/home/frisoe/Desktop/Root/withouthole/"], help="Directories to scan for ROOT files.")
    parser.add_argument("--kappa_csv", default="scripts/kappa.csv", help="Path to the kappa.csv file.")
    parser.add_argument("--range_csv", default="scripts/proton_stopping_range.csv", help="Path to the proton_stopping_range.csv file.")
    args = parser.parse_args()

    # Calculate the initial leakage noise component by subtracting constant noise sources in quadrature
    try:
        initial_leakage_noise_sq = INITIAL_FWHM_RESOLUTION_EV**2 - FANO_NOISE_EV**2 - SERIES_NOISE_EV**2
        if initial_leakage_noise_sq < 0:
            raise ValueError("Initial FWHM is smaller than the sum of Fano and Series noise.")
        initial_leakage_noise_ev = np.sqrt(initial_leakage_noise_sq)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


    # 1. Create interpolation functions for kappa and projected range
    kappa_func = load_and_prepare_interpolator(args.kappa_csv, 'kin[MeV]', 'D/(95MeVmb)', x_multiplier=1000)
    range_func = load_and_prepare_interpolator(args.range_csv, 'IonEnergy(keV)', 'ProjectedRange(um)')
    if kappa_func is None or range_func is None: exit(1)

    # 2. Calculate base metrics from all simulation files once
    print("Calculating base metrics from all simulation files...")
    base_metrics_df = calculate_base_metrics(args.dirs, kappa_func, range_func)

    if base_metrics_df.empty:
        print("No valid simulation data found.")
        exit(1)

    # 3. Loop through each CL and Orbit scenario to calculate final degradation
    all_scenarios_results = [] # Collect results from all scenarios for final plotting
    for cl_level, orbit_maps in FLUENCE_MAPS_ALL.items():
        # --- Only process and print results for CL95 ---
        if cl_level != 'CL95':
            continue

        # Combine 98-deg orbits
        orbit_maps['98-deg'] = {
            energy: orbit_maps["98-deg-AP9"].get(energy, 0) + orbit_maps["98-deg-SP"].get(energy, 0)
            for energy in set(orbit_maps["98-deg-AP9"]) | set(orbit_maps["98-deg-SP"])
        }
        
        for orbit_name, fluence_map in orbit_maps.items():
            if 'AP9' in orbit_name or 'SP' in orbit_name: continue # Skip individual 98-deg components

            scenario_df = base_metrics_df.copy()
            
            # Calculate scenario-specific values
            scenario_df['Mission_Fluence_Env_Diff'] = scenario_df['Incident_Energy'].map(fluence_map)
            scenario_df['Energy_Bin_Width_MeV'] = scenario_df['Incident_Energy'].map(ENERGY_BIN_WIDTHS_MEV)
            
            # Total protons hitting the detector
            scenario_df['Total_Protons'] = (
                scenario_df['Mission_Fluence_Env_Diff'] *
                scenario_df['Energy_Bin_Width_MeV'] *
                scenario_df['A_eff_cm2'] * 
                scenario_df['Funnelling_Efficiency'] * 
                scenario_df['Correction_Factor']
            )
            
            # NIEL equivalent damage
            scenario_df['Equivalent_Damage'] = scenario_df['Total_Protons'] * scenario_df['Interpolated_Kappa']
            
            # Leakage current increase for this specific energy/configuration
            scenario_df['Delta_I_Contribution'] = ALPHA_DAMAGE_RATE * scenario_df['Projected_Range_cm'] * scenario_df['Equivalent_Damage']

            # --- Calculate degradation for all configs and create a summary table ---
            
            all_config_results = []
            # Group by each unique configuration to calculate its total degradation
            for config_keys, group_df in scenario_df.groupby(['Mirror', 'Filter', 'Hole']):
                mirror, filt, hole = config_keys
                
                # Sum the contributions from all incident energies for this config
                total_delta_I = group_df['Delta_I_Contribution'].sum()
                # print(total_delta_I)
                
                # version 1
                # Calculate new total leakage current and new noise components

                # ENC_extra = math.sqrt((4/3)*((total_delta_I*1e-6)/1.602e-19))
                # E_extra = ENC_extra*2.355*3.6
                
                # Etot_new= math.sqrt(127.3891675**2+E_extra**2)
                # new_fwhm_resolution_ev = Etot_new
                
                # degradation_ev = Etot_new - INITIAL_FWHM_RESOLUTION_EV
                # degradation_per_year_ev = degradation_ev / 5.0

                # version 2

                ENC_extra = math.sqrt((4/3)*((total_delta_I*1e-6)/1.602e-19))
                E_extra = ENC_extra*2.355*3.6

                new_elec_noise = math.sqrt(E_extra**2+ELEC_NOISE_240K**2)

                new_res = math.sqrt(new_elec_noise**2 + FANO_NOISE_EV**2)
                new_fwhm_resolution_ev = new_res
                degradation_ev = new_res - INITIAL_FWHM_RESOLUTION_EV
                degradation_per_year_ev = degradation_ev / 5.0

                # # version 3
                new_total_current_I1 = INITIAL_LEAKAGE_CURRENT_A + total_delta_I

                # print(total_delta_I)

                # new_leakage_noise_ev = initial_leakage_noise_ev * np.sqrt(1 + (total_delta_I/ INITIAL_LEAKAGE_CURRENT_A))
                # # print(new_leakage_noise_ev)
                # new_fwhm_resolution_ev = np.sqrt(FANO_NOISE_EV**2 + SERIES_NOISE_EV**2 + new_leakage_noise_ev**2)
                
                # degradation_ev = new_fwhm_resolution_ev - INITIAL_FWHM_RESOLUTION_EV
                # degradation_per_year_ev = degradation_ev / 5.0

    

                all_config_results.append({
                    'Mirror': mirror, 'Filter': filt, 'Hole': hole,
                    'New_FWHM_eV': new_fwhm_resolution_ev,
                    'Degradation_eV': degradation_ev,
                    'Degradation_eV_per_Year': degradation_per_year_ev
                })
            
            if not all_config_results:
                print(f"\n--- No data to generate summary table for scenario: {cl_level}, {orbit_name} ---")
                continue

            results_df = pd.DataFrame(all_config_results)
            
            # --- FIX: Add the orbit info and append the results for final plotting ---
            results_df['orbit'] = orbit_name
            all_scenarios_results.append(results_df)


            # --- Generate the Delta Degradation Comparison Table ---
            print("\n" + "="*140)
            print(f"Energy Resolution Degradation Comparison Table at {cl_level}, {orbit_name}")
            print(f"Baseline: DCC, filterOn, withOpening")
            print("="*140)

            # Find the baseline degradation
            baseline_row = results_df[
                (results_df['Mirror'] == 'DCC') &
                (results_df['Filter'] == 'filterOn') &
                (results_df['Hole'] == 'withOpening')
            ]

            if baseline_row.empty:
                print("Baseline configuration not found. Cannot calculate percentage change.")
            else:
                baseline_degradation = baseline_row['Degradation_eV'].iloc[0]
                if baseline_degradation > 0:
                    results_df['Degradation_Change_pct'] = ((results_df['Degradation_eV'] - baseline_degradation) / baseline_degradation) * 100
                else:
                    results_df['Degradation_Change_pct'] = 0.0 # Avoid division by zero if baseline has no degradation
                
                # Sort for presentation
                results_df.sort_values(by=['Mirror', 'Filter', 'Hole'], inplace=True)

                # Print the formatted table
                header = f"{'Mirror':<6} | {'Filter':<9} | {'Hole':<14} | {'New FWHM [eV]':>15} | {'Degradation [eV]':>18} | {'Degradation [eV/yr]':>20} | {'Degradation Change (%)':>24}"
                print(header)
                print("-" * len(header))
                for _, row in results_df.iterrows():
                    print(f"{row['Mirror']:<6} | {row['Filter']:<9} | {row['Hole']:<14} | {row['New_FWHM_eV']:>15.2f} | {row['Degradation_eV']:>18.2f} | {row['Degradation_eV_per_Year']:>20.2f} | {row['Degradation_Change_pct']:>24.2f}")
            print("="*140)

    # --- After all scenarios, generate plots ---
    if all_scenarios_results:
        master_results_df = pd.concat(all_scenarios_results, ignore_index=True)
        
        print("\n" + "="*140)
        print("Generating final bar charts for CL95...")
        print("="*140)
        plot_degradation_bar_charts(master_results_df, 'CL95')
        
        print("\n" + "="*140)
        print("Generating final resolution summary charts for CL95...")
        print("="*140)
        plot_final_resolution_charts(master_results_df, 'CL95')