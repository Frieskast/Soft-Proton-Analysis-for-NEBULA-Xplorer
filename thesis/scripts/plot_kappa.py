import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def plot_kappa_data(csv_path, figures_dir):
    """
    Reads kappa data from a CSV and plots it with styling similar to the flux script.
    """
    # Apply the same plotting style as the calculate_flux.py script
    plt.rcParams.update({
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18,
        "figure.labelsize": 18,
        "lines.linewidth": 4.0,
        "lines.markersize": 10,
    })

    # --- Load the data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return

    # Convert energy from MeV to keV
    df['kin[keV]'] = df['kin[MeV]'] * 1000

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the data using keV
    ax.plot(df['kin[keV]'], df['D/(95MeVmb)'], marker='o', linestyle='-', color='#4169E1')

    # --- Configure axes and labels ---
    ax.set_xlabel("Kinetic Energy [keV]")
    ax.set_ylabel("$D_p (E)$ / $D_n$ (1 MeV)")
    # ax.set_title("Hardness Factor vs. Proton Kinetic Energy")
    
    # Set x-axis limits from 1 to 1000 keV
    ax.set_xlim(1e0, 1e3)
    ax.set_ylim(1e1,1e4)

    # Set custom ticks as requested
    major_ticks = [100, 500, 1000]
    minor_ticks = [20, 30, 50, 70, 200, 300, 700]
    ax.set_xticks(major_ticks)
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))

    # Use logarithmic scales for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add grid for better readability
    ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # --- Save the figure ---
    try:
        os.makedirs(figures_dir, exist_ok=True)
        fname_base = os.path.join(figures_dir, "kappa_vs_energy")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        # print(f"Saved hardness factor plot to {fname_base}.pdf")
    except Exception as e:
        print(f"Failed to save the plot: {e}")
    
    # Show the plot instead of saving
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Define the path to the CSV file and the output directory
    # Assumes the script is in the same directory as the CSV file
    csv_file_path = os.path.join(os.path.dirname(__file__), 'kappa.csv')
    output_directory = "/home/frisoe/Desktop/Thesis/figures/hardness_factor/"
    
    plot_kappa_data(csv_file_path, output_directory)