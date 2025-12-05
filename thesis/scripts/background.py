"""
Very small scatter-only Y-Z plotter for SmallDet and FocalDet hits.

Run:
    python background.py

Collects y,z from all found output_*.root files and plots two scatter panels:
 left: SmallDet (y vs z), right: FocalDet (y vs z).
"""
import glob
import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.labelsize": 26, 
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16, 
    "legend.fontsize": 20, 
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
})

def find_tree_names(root):
    keys = list(root.keys())
    sd = next((k for k in keys if k.startswith("SmallDet")), None)
    sd_summary = next((k for k in keys if k.startswith("SmallDetSummary")), None)
    fd = next((k for k in keys if k.startswith("FocalDet")), None)
    return sd, sd_summary, fd

def parse_traj_string(traj_s):
    # traj_s like "x,y,z; x,y,z; ... [| other trajectories]"
    ys = []
    zs = []
    if traj_s is None:
        return ys, zs
    # ensure string type
    s = traj_s.decode() if isinstance(traj_s, (bytes, bytearray)) else str(traj_s)
    # multiple proton trajectories separated by '|'
    parts = [p for p in s.split("|") if p]
    for p in parts:
        # points separated by ';'
        for pt in p.split(";"):
            pt = pt.strip()
            if not pt:
                continue
            coords = pt.split(",")
            if len(coords) < 3:
                continue
            try:
                y = float(coords[1])
                z = float(coords[2])
            except Exception:
                continue
            ys.append(y)
            zs.append(z)
    return ys, zs

def collect_yz_from_file(path):
    try:
        f = uproot.open(path)
    except Exception:
        return None, None, None, None
    sd, sd_summary, fd = find_tree_names(f)
    small_y = small_z = None
    focal_y = focal_z = None
    try:
        if sd:
            arr = f[sd].arrays(["y","z"], library="np")
            y = np.asarray(arr["y"]).astype(float)
            z = np.asarray(arr["z"]).astype(float)
            mask = np.isfinite(y) & np.isfinite(z)
            small_y, small_z = y[mask], z[mask]
        elif sd_summary:
            # parse trajectory strings from SmallDetSummary -> extract y,z points
            arr = f[sd_summary].arrays(["trajectory"], library="np")
            trajs = arr["trajectory"]
            ys_all = []
            zs_all = []
            for t in trajs:
                ys, zs = parse_traj_string(t)
                ys_all.extend(ys)
                zs_all.extend(zs)
            if ys_all:
                small_y = np.asarray(ys_all, dtype=float)
                small_z = np.asarray(zs_all, dtype=float)
    except Exception:
        small_y = small_z = None
    try:
        if fd:
            arr = f[fd].arrays(["y","z"], library="np")
            y = np.asarray(arr["y"]).astype(float)
            z = np.asarray(arr["z"]).astype(float)
            mask = np.isfinite(y) & np.isfinite(z)
            focal_y, focal_z = y[mask], z[mask]
    except Exception:
        focal_y = focal_z = None
    return small_y, small_z, focal_y, focal_z

def get_summary_entries_from_file(path):
    try:
        f = uproot.open(path)
    except Exception:
        return 0
    _, sd_summary, _ = find_tree_names(f)
    if not sd_summary:
        return 0
    try:
        return int(f[sd_summary].num_entries)
    except Exception:
        try:
            # fallback: length of arrays
            arr = f[sd_summary].arrays(library="np")
            first_key = next(iter(arr))
            return len(arr[first_key])
        except Exception:
            return 0

# --- Make gather_files accept a concentrator type ---
def gather_files(concentrator):
    # adjust path pattern as needed; now filters by concentrator type
    pattern = f"/home/frisoe/Desktop/Thesis/build/build/root/withhole/output_{concentrator}_filterOn_1000keV_option4_N1m.root"
    return sorted(glob.glob(pattern, recursive=True))

# --- Function to apply consistent tick styling ---
def apply_tick_styling(ax):
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
    ax.set_axisbelow(True)
    # ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
    # ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

def main():
    # --- Define the output directory ---
    output_dir = "/home/frisoe/geant4/geant4-v11.3.1/examples/projects/thesis/figures/background"
    os.makedirs(output_dir, exist_ok=True)

    # --- Loop through concentrator types ---
    for concentrator in ['DCC', 'DPH', 'SP']:
        print(f"\nGenerating plot for concentrator: {concentrator}")
        files = gather_files(concentrator)

        if not files:
            print(f"No ROOT files found for concentrator: {concentrator}")
            continue

        small_ys = []
        small_zs = []
        focal_ys = []
        focal_zs = []

        # total entries in SmallDetSummary across files (used in title)
        small_summary_entries_total = 0

        print(f"Found {len(files)} ROOT files to process...")
        for p in files:
            # --- Print the name of the file being processed ---
            print(f"  -> Processing {os.path.basename(p)}")
            sy, sz, fy, fz = collect_yz_from_file(p)
            if sy is not None and sz is not None and sy.size>0:
                small_ys.append(sy); small_zs.append(sz)
            if fy is not None and fz is not None and fy.size>0:
                focal_ys.append(fy); focal_zs.append(fz)
            small_summary_entries_total += get_summary_entries_from_file(p)

        if not small_ys and not focal_ys:
            print("No hit data found in files.")
            continue

        small_y_all = np.concatenate(small_ys) if small_ys else np.array([])
        small_z_all = np.concatenate(small_zs) if small_zs else np.array([])
        focal_y_all = np.concatenate(focal_ys) if focal_ys else np.array([])
        focal_z_all = np.concatenate(focal_zs) if focal_zs else np.array([])

        # --- Combine Focal and Small detector hits for a single heatmap ---
        combined_y = np.concatenate([focal_y_all, small_y_all])
        combined_z = np.concatenate([focal_z_all, small_z_all])

        # --- Create a single plot instead of two subplots ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # --- Plot all hits as a single heatmap ---
        if combined_y.size > 0:
            # --- Calculate histogram data first to handle zeros explicitly ---
            counts, xedges, yedges = np.histogram2d(combined_y, combined_z, bins=(500, 500))
            
            # --- Apply a Gaussian filter to smooth the data ---
            # The `sigma` value controls the amount of blurring. Higher values = more smearing.
            # We filter the raw counts (with zeros).
            smoothed_counts = gaussian_filter(counts, sigma=1)

            # --- Remove the mask to allow colors to bleed into empty areas ---
            # By not setting original zero-bins to NaN, the smoothing will fill them.

            # Get the colormap
            cmap = plt.get_cmap('nipy_spectral')
            # Set the color for underflows (values near zero) to black.
            cmap.set_under('black')

            # --- Use imshow for a smoother plot ---
            # The LogNorm's vmin is set to a small number just above zero.
            # This ensures that smoothed areas with very low (but non-zero) counts are colored,
            # while true zero areas (if any remain) will be black.
            mesh = ax.imshow(smoothed_counts.T, cmap=cmap, norm=LogNorm(vmin=1e-1, vmax=1e4), 
                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                             origin='lower', interpolation='bilinear')
            
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Number of Protons")
        else:
            ax.text(0.5, 0.5, "No detector hits found", ha="center", va="center", color='white')

        # --- Add a circle in the middle of the plot ---
        circle = Circle((0, 0), 4, edgecolor='white', facecolor='none', linewidth=1.5, linestyle='--')
        ax.add_patch(circle)

        # --- Set titles and labels for the combined plot ---
        small_hits_display = small_summary_entries_total if small_summary_entries_total > 0 else small_y_all.size
        # ax.set_title(
        #     f"Total Proton Hits on Focal Plane & Detector - {concentrator}"
        # )
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("z [mm]")
        # ax.set_ylim(-100,100)
        # ax.set_xlim(-100,100)
        # ax.set_aspect("equal", adjustable="box")
        # --- Set a black background that will show through for zero-hit bins ---
        ax.set_facecolor("black")
        # ax.legend() # Display the label for the rectangle

        # --- Apply consistent tick styling ---
        apply_tick_styling(ax)

        plt.tight_layout()
        # --- Save the plot as a PDF ---
        output_path = os.path.join(output_dir, f"focal_plane_hits_{concentrator}.pdf")
        plt.savefig(output_path, format='pdf')
        print(f"Saved plot to: {output_path}")
        plt.close()
        plt.show()

if __name__ == "__main__":
    main()