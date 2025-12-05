# filepath: /home/frisoe/geant4/geant4-v11.3.1/examples/projects/thesis/scripts/mean_energy.py
import os
import re
import glob
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MultipleLocator
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gaussian_kde

# style (match other scripts)
plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "legend.title_fontsize": 18,
    "figure.labelsize": 18,
    "lines.linewidth": 4.0,
    "lines.markersize": 10,
})

def parse_N_from_name(fname):
    m = re.search(r"_N(\d+)([kmKM])?", fname)
    if not m:
        return None
    n = int(m.group(1))
    suf = m.group(2)
    if not suf:
        return n
    suf = suf.lower()
    if suf == "k":
        return n * 1000
    if suf == "m":
        return n * 1000000
    return n

def parse_metadata_from_name(fname):
    base = os.path.basename(fname)
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    N = parse_N_from_name(base)
    return energy, N

def _ensure_figures_dir(dirname="figures/reflection_analysis"):
    os.makedirs(dirname, exist_ok=True)
    return dirname

def collect_entry_energies(dirs, pattern="**/output_*.root"):
    by_energy = {}  # energy_keV -> list of arrays (per-file)
    files_processed = 0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, pattern), recursive=True))
        for fpath in files:
            try:
                with uproot.open(fpath) as f:
                    energy, N = parse_metadata_from_name(fpath)
                    if energy is None or energy == 50:
                        continue
                    # Prefer SmallDetSummary.entryE (keV). Fallback to SmallDet.Ekin if needed.
                    entry_vals = None
                    if "SmallDetSummary" in f:
                        t = f["SmallDetSummary"]
                        if "entryE" in t.keys():
                            entry_vals = t["entryE"].array(library="np")
                    elif "SmallDet" in f:
                        t = f["SmallDet"]
                        if "Ekin" in t.keys():
                            entry_vals = t["Ekin"].array(library="np")
                    if entry_vals is None or entry_vals.size == 0:
                        continue
                    # ensure numpy 1D array of floats (keV)
                    entry_vals = np.asarray(entry_vals, dtype=float)
                    by_energy.setdefault(energy, []).append(entry_vals)
                    files_processed += 1
            except Exception:
                # skip malformed files silently
                continue
    return by_energy, files_processed

def combine_groups(by_energy):
    energies = sorted(by_energy.keys())
    combined = {}
    for e in energies:
        arrays = by_energy[e]
        # concatenate all arrays for this incident energy
        try:
            vals = np.concatenate(arrays) if len(arrays) > 1 else arrays[0].ravel()
        except Exception:
            # fallback: flatten pieces manually
            vals = np.hstack([np.asarray(a).ravel() for a in arrays])
        combined[e] = vals
    return combined

def plot_violin_and_means(combined, outdir):
    energies = sorted(combined.keys())
    data_lists = [combined[e] for e in energies]

    if not energies:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8,6))
    # violin plot at x positions = incident energies
    positions = np.array(energies, dtype=float)
    # width: relative spacing
    if len(positions) > 1:
        dx = np.min(np.diff(np.sort(positions)))
        width = dx * 1.8
    else:
        width = 40.0
    parts = ax.violinplot(data_lists, positions=positions, widths=width, showmeans=False, showmedians=False, showextrema=False)
    
    # --- Gradient Fill Logic ---
    # Get plot limits
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    cmap = 'plasma' # A perceptually uniform colormap

    # Make original violin bodies transparent but keep edges
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor('none')
        pc.set_edgecolor('k')
        pc.set_alpha(1)
        pc.set_linewidth(1.5)

        # --- Create a custom gradient based on data density (KDE) ---
        # Get the raw data for this specific violin
        data = data_lists[i]
        if len(data) < 2: continue # KDE needs at least 2 points

        # Calculate the Kernel Density Estimate
        kde = gaussian_kde(data)
        
        # Create a series of y-points spanning the violin's range
        y_vals = np.linspace(data.min(), data.max(), 256)
        
        # Evaluate the density at these points
        density = kde(y_vals)
        
        # Normalize the density to create the gradient
        density_gradient = (density - density.min()) / (density.max() - density.min())
        
        # Create the gradient image from the density values
        gradient_img = plt.get_cmap(cmap)(density_gradient)[:, :3] # Get RGB, drop alpha
        gradient_img = gradient_img.reshape(256, 1, 3) # Reshape for imshow

        # Create a clipping path from the violin body
        path = Path(pc.get_paths()[0].vertices)
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        
        # Add the gradient image, clipped to the path of the violin
        img = ax.imshow(gradient_img, origin="lower", extent=[xmin, xmax, data.min(), data.max()], aspect="auto",
                        clip_path=patch)

    # --- Add a colorbar for the gradient ---
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    # The colorbar now represents normalized density (0 to 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm, orientation='vertical')
    cb.set_label('Normalized Proton Density')


    # compute stats and overplot mean +/- std
    means = []
    stds = []
    medians = []
    counts = []
    for vals in data_lists:
        vals = np.asarray(vals)
        counts.append(vals.size)
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if vals.size>1 else 0.0)
        medians.append(np.median(vals))

    means = np.array(means)
    stds = np.array(stds)
    medians = np.array(medians)
    counts = np.array(counts)

    # Replace the error bars with a single scatter plot for the mean value.
    # This plots a white dot with a black outline for visibility.
    # Increase the 's' parameter (e.g., from 50 to 100) to make the marker larger.
    ax.scatter(positions, means, marker='o', c='white', edgecolor='black', s=100, zorder=5, label='Mean')
    
    # ax.scatter(positions, medians, marker='D', color='white', edgecolor='black', label='Median')

    ax.set_xlabel('Incident proton energy [keV]')
    ax.set_ylabel('Entry energy [keV]')
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # Set custom x-axis ticks
    ax.set_xticks([100, 250, 500, 1000])

    # Highlight the 0-15 keV region
    ax.axhspan(0, 15, color='#DC143C', alpha=0.2, zorder=0, label='0-15 keV Region')

    # Set y-axis ticks
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))

    ax.set_ylim(0, 1001)
    # ax.set_xlim(0, None)
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    outdir = _ensure_figures_dir(outdir)
    fname_pdf = os.path.join(outdir, "hit_entry_energy_vs_incident.pdf")
    fname_png = os.path.join(outdir, "hit_entry_energy_vs_incident.png")
    fig.savefig(fname_pdf, dpi=400)
    fig.savefig(fname_png, dpi=400)
    plt.show()
    plt.close(fig)
    print(f"Saved {fname_pdf}")

def print_summary_table(combined):
    print(f"{'E_inc [keV]':>12} | {'n_hits':>8} | {'mean [keV]':>12} | {'std [keV]':>10} | {'median [keV]':>12}")
    print("-"*66)
    for e in sorted(combined.keys()):
        vals = combined[e]
        n = vals.size
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if n>1 else 0.0
        med = np.median(vals)
        print(f"{int(e):12d} | {n:8d} | {mean:12.2f} | {std:10.2f} | {med:12.2f}")

def main():
    parser = argparse.ArgumentParser(description="Plot energy of protons that hit the detector per incident energy")
    parser.add_argument("--dirs", "-d", nargs="+", default=[
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ], help="Directories to scan for ROOT files")
    parser.add_argument("--outdir", "-o", default="figures/reflection_analysis", help="Output directory for figures")
    args = parser.parse_args()

    by_energy, files = collect_entry_energies(args.dirs)
    if files == 0 or not by_energy:
        print("No input files found or no SmallDetSummary.entryE data in files.")
        return

    combined = combine_groups(by_energy)
    print_summary_table(combined)
    plot_violin_and_means(combined, args.outdir)

if __name__ == "__main__":
    main()