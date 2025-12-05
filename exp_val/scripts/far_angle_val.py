import ROOT
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as mticker

root1 = "/home/frisoe/Desktop/exp/build/build/root/output_0.36deg_250keV_Nm_r0.root"
root2 = "/home/frisoe/Desktop/exp/build/build/root/output_0.36deg_250keV_N300k_r0.root"
figures_dir = "/home/frisoe/Desktop/exp/figures"

def get_scattering_distribution(rootfile, histname="Scattering angle 0-90 deg;1"):
    """Return bin centers, widths, normalized counts, and errors from ROOT histogram."""
    f = ROOT.TFile.Open(rootfile)
    if not f or f.IsZombie():
        raise IOError(f"Could not open {rootfile}")

    h = f.Get(histname)
    if not h:
        raise KeyError(f"Histogram {histname} not found in {rootfile}")

    nbins = h.GetNbinsX()
    bin_edges = np.array([h.GetBinLowEdge(i+1) for i in range(nbins)] + [h.GetXaxis().GetXmax()])
    bin_centers = np.array([h.GetBinCenter(i+1) for i in range(nbins)])
    counts = np.array([h.GetBinContent(i+1) for i in range(nbins)])

    # restrict to 0–90 deg
    mask = (bin_centers >= 0) & (bin_centers <= 90)
    bin_edges = bin_edges[np.where(mask)[0][0] : np.where(mask)[0][-1] + 2]
    bin_centers = bin_centers[mask]
    counts = counts[mask]

    total = np.sum(counts)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    norm_counts = counts / (total * bin_widths)
    errors = np.sqrt(counts) / (total * bin_widths)
    n_primaries = int(h.GetEntries())

    f.Close()
    return bin_centers, bin_widths, norm_counts, errors, int(total), n_primaries


plt.rcParams.update({
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

# --- Create a figure with a single subplot ---
fig, ax = plt.subplots(figsize=(8, 6))

# --- Define a helper function to avoid duplicating plotting code ---
def plot_histograms_on_ax(ax):
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)

    # --- Plot first histogram ---
    bin_centers, bin_widths, norm_counts, errors, total_events, n_primaries = get_scattering_distribution(root1)
    ax.bar(bin_centers, norm_counts, width=bin_widths, align='center',
           alpha=0.5, label="Multiple Scattering option 4", color="#2ca02c", edgecolor="black")
    ax.errorbar(bin_centers, norm_counts, yerr=errors, fmt='o', markersize=3, capsize=0.0, color="#2ca02c")

    # --- Plot second histogram ---
    bin_centers, bin_widths, norm_counts, errors, total_events, n_primaries = get_scattering_distribution(root2)
    ax.bar(bin_centers, norm_counts, width=bin_widths, align='center',
           alpha=0.5, label="Single Scattering", color="#DC143C", edgecolor="black")
    ax.errorbar(bin_centers, norm_counts, yerr=errors, fmt='o', markersize=3, capsize=0.0, color="#DC143C")

    ax.set_xlabel("Scattering angle [deg]")
    ax.set_ylabel("Normalized counts [deg$^{-1}$]")
    ax.set_xlim(0, 90)
    ax.legend()

# --- Plot on the axis ---
plot_histograms_on_ax(ax)
# ax.set_title("Scattering angle distributions for 250 keV and $\\theta_i = 0.36^\\circ$")
# --- Format y-axis labels to have 2 decimal points ---
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.tight_layout()

out_pdf = os.path.join(figures_dir, "eff_250keV_0.36deg.pdf")
    # save as PDF (vector) and then show/close
plt.savefig(out_pdf, bbox_inches="tight")

plt.show()
