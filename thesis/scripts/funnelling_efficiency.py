import os
import re
import glob
import math
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# apply style once
plt.rcParams.update({
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

# color palette (use provided colors / related)
COLOR_ORBIT = {
    "45-deg": "#4169E1",    # royal blue
    "98-deg": "#DC143C",    # crimson red (use red/blue combination)
}

# color: one color per option
COLOR_MIRROR = {
    "SP":  "#4169E1",   # royal blue
    "DCC": "#2ca02c",   # green
    "DPH": "#DC143C",   # crimson
}

# --- Color map for filter/hole configurations ---
COLOR_FILTER = {
    "filterOn": "#d62728", # Brick Red
    "filterOff": "#1f77b4", # Muted Blue
}
COLOR_HOLE = {
    "withHole": "#d62728",    # Brick Red
    "withoutHole": "#1f77b4", # Medium Gray
}

# --- Correction factors for dumbbell opening ---
CORRECTION_FACTORS = {
    "withHole": 0.92131,
    "withoutHole": 0.92022
}


def parse_N_from_name(fname):
    """Extracts N (number of events) from filename."""
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
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else ""
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else ""
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    N = parse_N_from_name(base)
    return mirror, filter_flag, energy, N

def find_tree_keys(f):
    keys = list(f.keys())
    for k in keys:
        if k.startswith("FocalDet"):
            return k, None
    for k in keys:
        if k.startswith("SmallDet"):
            return None, k
    return None, None

def sum_tid_tree(rootfile):
    # locate a TID tree (name starting with "TID") and sum its columns
    for k in rootfile.keys():
        if k.startswith("TID"):
            try:
                tree = rootfile[k]
                cols = []
                if "TID_krad_det" in tree.keys():
                    cols.append("TID_krad_det")
                if "TID_krad_focal" in tree.keys():
                    cols.append("TID_krad_focal")
                if not cols:
                    return None, None, k
                arr = tree.arrays(cols, library="np")
                det_sum = float(np.nansum(arr["TID_krad_det"])) if "TID_krad_det" in arr else 0.0
                focal_sum = float(np.nansum(arr["TID_krad_focal"])) if "TID_krad_focal" in arr else 0.0
                return det_sum, focal_sum, k
            except Exception:
                return None, None, k
    return None, None, None

def extract_first_values(rootfile):
    """
    Return a dict: key -> ("tree", {col: first_value, ...}) or ("hist", first_bin_value).
    """
    firsts = {}
    for k in rootfile.keys():
        try:
            obj = rootfile[k]
        except Exception:
            continue
        # TTree / ntuple: try num_entries and read first entry
        try:
            if hasattr(obj, "num_entries") and obj.num_entries > 0:
                try:
                    arr = obj.arrays(library="np", entry_stop=1)
                    colvals = {}
                    for col, a in arr.items():
                        if len(a) > 0:
                            v = a[0]
                            # convert numpy scalar to Python scalar where possible
                            try:
                                v = v.item()
                            except Exception:
                                pass
                        else:
                            v = None
                        colvals[col] = v
                    firsts[k] = ("tree", colvals)
                    continue
                except Exception:
                    pass
        except Exception:
            pass
        # Histogram: try to_numpy() -> (values, edges)
        try:
            vals_edges = obj.to_numpy()
            if isinstance(vals_edges, tuple) and len(vals_edges) >= 1:
                vals = vals_edges[0]
                first_bin = float(vals[0]) if len(vals) > 0 else None
                firsts[k] = ("hist", first_bin)
                continue
        except Exception:
            pass
    return firsts

def count_unique_events_in_tree(rootfile, treename, col="EventID"):
    try:
        tree = rootfile[treename]
        arr = tree.arrays([col], library="np")
        evts = np.asarray(arr[col]).astype(int)
        return int(np.unique(evts).size)
    except Exception:
        return 0

def compute_errors(k, N):
    """Computes efficiency and both binomial and Poisson errors."""
    if N is None or N <= 0 or k is None:
        return None, None, None
    
    k = max(0, k)
    p = k / float(N)
    
    # Binomial error: sqrt(p*(1-p)/N)
    # Use max(0,...) to avoid domain error if p > 1 due to float precision
    err_binom = math.sqrt(max(0, p * (1.0 - p) / N))
    
    # Poisson error for count k is sqrt(k). Error on efficiency p=k/N is sqrt(k)/N.
    err_poiss = math.sqrt(k) / N if k > 0 else 0.0
    
    return p, err_binom, err_poiss

def scan_and_plot(dirs, show=True, save=None, save_txt=None):
    entries = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        # recurse into subfolders so script finds files with the same subfolder structure
        pattern = os.path.join(d, "**", "output_*.root")
        files = sorted(glob.glob(pattern, recursive=True))
        for fpath in files:
            try:
                f = uproot.open(fpath)
            except Exception as e:
                print("Failed to open", fpath, e)
                continue
            # extract first values for all ntuples and histograms (for quick inspection)
            first_vals = extract_first_values(f)
            
            # --- Only use SmallDetSummary for hits calculation ---
            # The old find_tree_keys is no longer needed for hit calculation.
            # focal_key, small_key = find_tree_keys(f)
            
            tid_det_sum, tid_focal_sum, tid_tree = sum_tid_tree(f)
            mirror, filter_flag, energy, N = parse_metadata_from_name(fpath)
            # detect hole status from path (case-insensitive)
            hole = "withHole" if "withhole" in fpath.lower() else ("withoutHole" if "withouthole" in fpath.lower() else "")
            
            # --- FILTER OUT 50 KEV FILES ---
            if energy == 50:
                continue
            
            # --- Only process 250 keV files with 300k particles ---
            if energy == 250 and N != 300000:
                continue
            
            hits = 0
            source = "noTree"
            target_tree_name = "SmallDetSummary" # Target ntuple as requested
            
            # Check if the target ntuple exists in the file
            if target_tree_name in f:
                hits = count_unique_events_in_tree(f, target_tree_name, col="EventID")
                source = target_tree_name
            
            eff, err_binom, err_poiss = compute_errors(hits, N)

            # --- Apply correction factor based on hole status ---
            correction_factor = CORRECTION_FACTORS.get(hole, 1.0)
            if eff is not None:
                eff *= correction_factor
            if err_binom is not None:
                err_binom *= correction_factor
            if err_poiss is not None:
                err_poiss *= correction_factor

            # compose a short first-value summary (ensure fv_summary exists before appending)
            fv_parts = []
            for key, info in sorted(first_vals.items()):
                if info[0] == "tree":
                    # show up to first 2 columns
                    cols = list(info[1].items())
                    if cols:
                        k0, v0 = cols[0]
                        fv_parts.append(f"{key}:{k0}={v0}")
                        if len(cols) > 1:
                            k1, v1 = cols[1]
                            fv_parts.append(f"{key}:{k1}={v1}")
                elif info[0] == "hist":
                    fv_parts.append(f"{key}:bin0={info[1]}")
            fv_summary = " | ".join(fv_parts[:6])  # limit length

            entries.append({
                "path": fpath,
                "basename": os.path.basename(fpath),
                "mirror": mirror,
                "filter": filter_flag,
                "hole": hole,
                "energy_keV": energy,
                "N": N,
                "hits": hits,
                "eff": eff,
                "err": err_binom,
                "err_poisson": err_poiss,
                "tid_det_sum": tid_det_sum,
                "tid_focal_sum": tid_focal_sum,
                "tid_tree": tid_tree,
                "tree": source,
                "fv_summary": fv_summary
            })
            # per-file printing removed to suppress rootfile content output
    if not entries:
        print("No files found; nothing to plot.")
        return

    # Print all efficiencies and errors (CSV header + rows)
    print("basename,mirror,filter,hole,energy_keV,N,hits,eff,err_binomial")
    for e in entries:
        print(",".join([
            str(e.get('basename','')),
            str(e.get('mirror','')),
            str(e.get('filter','')),
            str(e.get('hole','')),
            str(e.get('energy_keV','')),
            str(e.get('N','')),
            str(e.get('hits','')),
            (f"{e['eff']:.6g}" if e.get('eff') is not None else ""),
            (f"{e['err']:.6g}" if e.get('err') is not None else ""),
        ]))

    # --- REVISED: Plotting function for single-concentrator plots ---
    def plot_single_comparison(entries, comparison_col, fixed_col, fixed_val, concentrator, colors, title_prefix, filename_prefix, save, show):
        """
        Plots a comparison for a single concentrator.
        """
        # Filter the initial dataset based on the fixed condition and concentrator
        filtered_entries = [
            e for e in entries if e.get(fixed_col) == fixed_val and e.get('mirror') == concentrator
        ]

        if not filtered_entries:
            print(f"No data for concentrator '{concentrator}' with condition '{fixed_col} = {fixed_val}'; skipping plot.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        # full_title = f"{title_prefix}\n(Concentrator: {concentrator})"
        # ax.set_title(full_title,)

        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Funnelling Efficiency [hits / N]")

        # Collect handles and labels for the legend
        handles, labels = [], []
        seen_labels = set()

        legend_label_mapping = {
            "filterOn": "Filter On", "filterOff": "Filter Off",
            "withHole": "With Opening", "withoutHole": "Without Opening"
        }

        # Group data by the column we want to compare (e.g., 'filterOn' vs 'filterOff')
        comparison_groups = {}
        for e in filtered_entries:
            key = e[comparison_col]
            if key not in comparison_groups:
                comparison_groups[key] = []
            comparison_groups[key].append(e)

        all_energies = set()
        # Plot each group in the comparison
        for key, group_entries in sorted(comparison_groups.items()):
            energies = [e["energy_keV"] for e in group_entries]
            efficiencies = [e["eff"] for e in group_entries]
            errors = [e["err"] for e in group_entries]

            sorted_indices = sorted(range(len(energies)), key=lambda k: energies[k])
            energies_sorted = [energies[i] for i in sorted_indices]
            efficiencies_sorted = [efficiencies[i] for i in sorted_indices]
            errors_sorted = [errors[i] for i in sorted_indices]
            
            all_energies.update(energies_sorted)

            color = colors.get(key, 'k')
            
            upper_bound = [e + err for e, err in zip(efficiencies_sorted, errors_sorted)]
            lower_bound = [e - err for e, err in zip(efficiencies_sorted, errors_sorted)]
            lines = ax.plot(energies_sorted, efficiencies_sorted, '-o', label=key, color=color)
            ax.fill_between(energies_sorted, lower_bound, upper_bound, color=color, alpha=0.2)
            
            legend_label = legend_label_mapping.get(key, key)
            
            if legend_label not in seen_labels:
                handles.append(lines[0])
                labels.append(legend_label)
                seen_labels.add(legend_label)

        ax.grid(True)
        
        # --- FINAL CORRECTED TICK SETUP ---
        ax.set_xlim(50, 1050)
        ax.set_ylim(1e-4,1e-3)
        major_ticks = [100, 250, 500, 1000]
        ax.set_xticks(major_ticks)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", direction="in", length=8, width=2.0)
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.5)
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', linestyle='--', linewidth=1.0, color='gray', alpha=0.7)
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.6, color='gray', alpha=0.4)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        # Add a descriptive legend title depending on the comparison type
        if comparison_col == 'filter':
            lg = ax.legend(handles, labels, title='Filter', loc='lower right')
        elif comparison_col == 'hole':
            lg = ax.legend(handles, labels, title='Opening', loc='lower right')
        else:
            lg = ax.legend(handles, labels, loc='lower right')

        plt.tight_layout()

        if save:
            figures_dir = "/home/frisoe/geant4/geant4-v11.3.1/examples/projects/thesis/figures/eff_analysis"
            os.makedirs(figures_dir, exist_ok=True)
            filename = f"{filename_prefix}_{concentrator}.pdf"
            fname = os.path.join(figures_dir, filename)
            plt.savefig(fname, dpi=200, format='pdf')
            print(f"Saved plot to {fname}")
        if show:
            plt.show()
        plt.close()

    # --- Generate plots for each concentrator individually ---
    concentrators_to_plot = ["DCC", "DPH", "SP"]

    for concentrator in concentrators_to_plot:
        # 1. Filter comparison (for withHole configuration)
        plot_single_comparison(
            entries=entries,
            comparison_col='filter',
            fixed_col='hole',
            fixed_val='withHole',
            concentrator=concentrator,
            colors=COLOR_FILTER,
            title_prefix="Funnelling Efficiency: Filter On vs. Off",
            filename_prefix="eff_filter_comparison",
            save=save,
            show=show
        )

        # 2. Hole comparison (for filterOff configuration)
        plot_single_comparison(
            entries=entries,
            comparison_col='hole',
            fixed_col='filter',
            fixed_val='filterOff',
            concentrator=concentrator,
            colors=COLOR_HOLE,
            title_prefix="Funnelling Efficiency: With vs. Without Hole",
            filename_prefix="eff_hole_comparison",
            save=save,
            show=show
        )

    # option: write summary to CSV file (unchanged)
    if save_txt and entries:
        try:
            with open(save_txt, "w", newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(["basename","mirror","filter","hole","energy_keV","N","hits","eff","err_binomial","TID_det_sum","TID_focal_sum","FIRST"])
                for e in entries:
                    writer.writerow([
                        e['basename'],
                        e['mirror'],
                        e['filter'],
                        e['hole'],
                        e['energy_keV'] or "",
                        e['N'] or "",
                        e['hits'],
                        e['eff'] if e['eff'] is not None else "",
                        e['err'] if e['err'] is not None else "",
                        e['tid_det_sum'] if e['tid_det_sum'] is not None else "",
                        e['tid_focal_sum'] if e['tid_focal_sum'] is not None else "",
                        e.get('fv_summary','')
                    ])
            print("Saved CSV summary to", save_txt)
        except Exception as ex:
            print("Failed to write summary CSV:", ex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot focusing efficiency from ROOT files (no CSV).")
    parser.add_argument("action", nargs="?", choices=["show","save","noshow"], default="show",
                        help="Action: show (default) = display plot; save = save to file; noshow = run without showing")
    parser.add_argument("--out", "-o", default="efficiency.png", help="Output filename when action=save")
    parser.add_argument("--txt", "-t", default=None, help="Write summary to text file (tab-separated)")
    parser.add_argument("--dirs", "-d", nargs="+", default=[
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ])
    args = parser.parse_args()

    if args.action == "show":
        scan_and_plot(args.dirs, show=True, save=None, save_txt=args.txt)
    elif args.action == "noshow":
        scan_and_plot(args.dirs, show=False, save=None, save_txt=args.txt)
    else:  # save
        scan_and_plot(args.dirs, show=False, save=args.out, save_txt=args.txt)