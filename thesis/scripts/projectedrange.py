import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def plot_range_data(csv_path, figures_dir):
    """
    Reads proton stopping range data from a CSV and plots it with styling
    similar to the kappa script.
    """
    # Apply the same plotting style as the other scripts
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

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the data
    ax.plot(df['IonEnergy(keV)'], df['ProjectedRange(um)'], marker='o', linestyle='-', color='#3CB371')

    # --- Configure axes and labels ---
    ax.set_xlabel("Proton Energy [keV]")
    ax.set_ylabel("Projected Range [µm]")
    
    # Set axis limits
    ax.set_xlim(10, 1000)
    ax.set_ylim(0.1, 20)

    # Set custom ticks
    ax.set_xticks([10, 100, 1000])
    ax.xaxis.set_minor_locator(FixedLocator([20, 30, 50, 70, 200, 300, 500, 700]))
    ax.set_yticks([0.1, 1, 10])
    ax.yaxis.set_minor_locator(FixedLocator([0.2, 0.3, 0.5, 0.7, 2, 3, 5, 7]))

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
        fname_base = os.path.join(figures_dir, "proton_range_vs_energy")
        fig.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{fname_base}.pdf", bbox_inches='tight')
        print(f"Saved proton range plot to {fname_base}.pdf/.png")
    except Exception as e:
        print(f"Failed to save the plot: {e}")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Define the path to the CSV file and the output directory
    csv_file_path = os.path.join(os.path.dirname(__file__), 'proton_stopping_range.csv')
    output_directory = "/home/frisoe/Desktop/Thesis/figures/stopping_range/"
    
    plot_range_data(csv_file_path, output_directory)