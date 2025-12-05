import uproot
import pandas as pd
import glob
import os
import argparse
import numpy as np
import re 
import matplotlib.pyplot as plt 

# --- Apply consistent Matplotlib style from calculate_flux.py ---
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "figure.labelsize": 18,
    "figure.titlesize": 22, 
    "lines.linewidth": 3.0,
    "lines.markersize": 8,
})

# --- Color mapping for concentrators (used in plots) ---
COLOR_MIRROR = {
    "SP":  "#4169E1",   # royal blue
    "DCC": "#2ca02c",   # green
    "DPH": "#DC143C",   # crimson
}

# --- Function to parse concentrator type from filename ---
def parse_mirror_type(fname):
    """Parses the mirror type (e.g., DCC, DPH, SP) from a filename."""
    m = re.search(r"output_([^_]+)_", os.path.basename(fname))
    return m.group(1) if m else "Unknown"

def analyze_reflection_materials(root_dirs):
    """
    Reads ROOT files from multiple directories, extracts mirror hit data,
    and prints a detailed percentile and heatmap-style analysis.
    """
    all_files = []
    for d in root_dirs:
        pattern = os.path.join(d, "**", "output_*.root")
        all_files.extend(glob.glob(pattern, recursive=True))

    if not all_files:
        print(f"Error: No files found in the specified directories.")
        return

    all_data = []
    ntuple_name = "SmallDetSummary"
    
    # --- Define both old and new naming conventions ---
    max_hits_to_read = 5

    new_names = {
        "count": "nReflections",
        "vol_prefix": "refl",
        "mat_prefix": "refl",
        "x_prefix": "refl",
        "y_prefix": "refl",
        "z_prefix": "refl",
    }
    # Old convention for backward compatibility
    old_names = {
        "count": "nMirrorHits",
        "vol_prefix": "hit",
        "mat_prefix": "hit",
    }

    for fpath in all_files:
        try:
            with uproot.open(fpath) as f:
                if ntuple_name not in f:
                    continue
                
                tree = f[ntuple_name]
                available_branches = tree.keys()

                # --- Detect which naming convention to use ---
                names = new_names if new_names["count"] in available_branches else old_names
                
                if names["count"] not in available_branches:
                    continue # Skip file if neither count branch is found

                # Build the list of columns to read
                cols_to_read = [names["count"]]
                for i in range(1, max_hits_to_read + 1):
                    cols_to_read.append(f'{names["vol_prefix"]}{i}_vol')
                    cols_to_read.append(f'{names["mat_prefix"]}{i}_mat')
                    if "x_prefix" in names: # Only add x,y,z for new convention
                        cols_to_read.append(f'{names["x_prefix"]}{i}_x')
                        cols_to_read.append(f'{names["y_prefix"]}{i}_y')
                        cols_to_read.append(f'{names["z_prefix"]}{i}_z')

                final_cols_to_read = [col for col in cols_to_read if col in available_branches]
                df = tree.arrays(final_cols_to_read, library="pd")
                
                # --- Rename columns to a consistent internal format ---
                rename_map = {names["count"]: "nMirrorHits"}
                for i in range(1, max_hits_to_read + 1):
                    if f'{names["vol_prefix"]}{i}_vol' in df.columns:
                        rename_map[f'{names["vol_prefix"]}{i}_vol'] = f'hit{i}_vol'
                    if f'{names["mat_prefix"]}{i}_mat' in df.columns:
                        rename_map[f'{names["mat_prefix"]}{i}_mat'] = f'hit{i}_mat'
                    if "x_prefix" in names and f'{names["x_prefix"]}{i}_x' in df.columns:
                        rename_map[f'{names["x_prefix"]}{i}_x'] = f'refl{i}_x'
                        rename_map[f'{names["y_prefix"]}{i}_y'] = f'refl{i}_y'
                        rename_map[f'{names["z_prefix"]}{i}_z'] = f'refl{i}_z'

                df.rename(columns=rename_map, inplace=True)

                # Add the concentrator type to the dataframe
                df['mirror'] = parse_mirror_type(fpath)
                # --- Keep source file path so we can determine filter/opening attributes later ---
                df['source_file'] = fpath
                
                all_data.append(df)

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")

    if not all_data:
        print("\nNo valid data could be extracted from any files.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    # --- Use filename metadata parsing (user-provided pattern) ---
    def parse_N_from_name(base):
        """Try to extract an integer N from the basename (various conventions)."""
        m = re.search(r"_N(\d+)", base, re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"_n(\d+)", base, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def parse_metadata_from_name(fname):
        """Parses mirror type, filter status, and energy from a filename."""
        base = os.path.basename(fname)
        m_mirror = re.search(r"output_([^_]+)_", base)
        mirror = m_mirror.group(1) if m_mirror else ""
        m_filter = re.search(r"_(filterOn|filterOff)_", base)
        filter_flag = m_filter.group(1) if m_filter else ""
        m_energy = re.search(r"_([0-9]+)keV_", base)
        energy = int(m_energy.group(1)) if m_energy else None
        N = parse_N_from_name(base)
        # Detect hole status from the full path, not just the basename
        hole = "withOpening" if "withhole" in fname.lower() else ("withoutOpening" if "withouthole" in fname.lower() else "")
        return mirror, filter_flag, hole, energy, N

    # apply parser and map to our canonical status names
    meta_cols = master_df['source_file'].apply(lambda s: pd.Series(parse_metadata_from_name(s)))
    meta_cols.columns = ['meta_mirror', 'filter_flag', 'hole', 'energy_keV', 'N']
    master_df = pd.concat([master_df.reset_index(drop=True), meta_cols.reset_index(drop=True)], axis=1)

    # map parsed tokens to existing status names used downstream
    def map_filter(flag):
        if flag == 'filterOn':
            return 'with_filter'
        if flag == 'filterOff':
            return 'without_filter'
        return 'unknown_filter'

    def map_hole(h):
        if h == 'withOpening':
            return 'with_opening'
        if h == 'withoutOpening':
            return 'without_opening'
        return 'unknown_opening'

    master_df['filter_status'] = master_df['filter_flag'].apply(map_filter)
    master_df['opening_status'] = master_df['hole'].apply(map_hole)
    # allow metadata mirror to override earlier parse if present
    master_df.loc[master_df['meta_mirror'].astype(bool), 'mirror'] = master_df.loc[master_df['meta_mirror'].astype(bool), 'meta_mirror']

    # --- Calculate the number of Gold reflections for each event ---
    def count_material_hits(row, material='G4_Au', max_hits=5):
        count = 0
        num_hits = int(row.get('nMirrorHits', 0))
        for i in range(1, min(num_hits, max_hits) + 1):
            mat_col = f'hit{i}_mat'
            if mat_col in row and row[mat_col] == material:
                count += 1
        return count

    master_df['nAuHits'] = master_df.apply(lambda row: count_material_hits(row, material='G4_Au'), axis=1)

    # Debug: show what the script actually detected
    print("\nDetected filter_status counts:")
    print(master_df['filter_status'].value_counts(dropna=False))
    print("\nDetected opening_status counts:")
    print(master_df['opening_status'].value_counts(dropna=False))

    # --- Convert Awkward-backed columns to standard pandas Series if necessary ---
    for col in master_df.columns:
        if ('mat' in col or 'vol' in col) and hasattr(master_df[col], 'to_list'):
            master_df[col] = pd.Series(master_df[col].to_list())

    print("\n" + "="*80)
    print("Analysis of Mirror Reflections for Detector Hits")
    print("="*80)
    print(f"Total events analyzed (hits on SmallDet): {len(master_df):,}")

    # --- Print average Gold reflections per detected proton ---
    print("\n--- Average Gold Reflections per Detected Proton ---")
    overall_avg_au_hits = master_df['nAuHits'].mean()
    print(f"Overall Average: {overall_avg_au_hits:.3f} Au reflections per detected proton.")

    avg_by_config = master_df.groupby(['mirror', 'filter_status', 'opening_status'])['nAuHits'].mean().reset_index()
    avg_by_config.sort_values(by=['mirror', 'filter_status', 'opening_status'], inplace=True)

    print("\nBy Configuration:")
    header = f"{'Mirror':<6} | {'Filter':<14} | {'Opening':<15} | {'Avg Au Reflections':>20}"
    print(header)
    print("-" * len(header))
    for _, row in avg_by_config.iterrows():
        if 'unknown' in row['filter_status'] or 'unknown' in row['opening_status']:
            continue
        print(f"{row['mirror']:<6} | {row['filter_status']:<14} | {row['opening_status']:<15} | {row['nAuHits']:>20.3f}")


    # --- Percentage tables for combinations of filter / opening ---
    print("\n--- Percentage of events over number of reflections for Filter / Opening combinations ---")
    combos = master_df.groupby(['filter_status', 'opening_status'])
    for (filt, opening), group in combos:
        if filt is None or opening is None:
            continue
        ct = group['nMirrorHits'].value_counts().sort_index()
        total = ct.sum()
        pct = (ct / total * 100).round(2)
        print(f"\nCombination: {filt} | {opening}  (n={total:,})")
        print("Reflections : Count   (% of this combination)")
        for refl_count in sorted(ct.index):
            print(f"{refl_count:>11} : {ct[refl_count]:>7,}   ({pct[refl_count]:>5.2f}%)")
        # ensure 0,1,2,3 presentation even when missing
        for needed in [0,1,2]:
            if needed not in ct.index:
                print(f"{needed:>11} : {0:>7,}   ({0.00:>5.2f}%)")
        # 3 combined
        ct_3plus = ct[ct.index >= 3].sum() if any(ct.index >= 3) else 0
        pct_3plus = (ct_3plus / total * 100).round(2) if total>0 else 0.0
        print(f"{'3+':>11} : {ct_3plus:>7,}   ({pct_3plus:>5.2f}%)")

    # --- build percent tables and plotting for filter / opening comparisons ---
    def build_status_percent_table(column):
        """
        Build a percent table indexed by status (e.g. with_filter/without_filter
        or with_opening/without_opening) with columns [0,1,2,'3+'] giving
        percentage of events with that number of reflections.
        """
        ct = pd.crosstab(master_df[column], master_df['nMirrorHits'])
        # Ensure columns 0,1,2 exist
        for i in [0,1,2]:
            if i not in ct.columns:
                ct[i] = 0
        # Combine 3+
        high_cols = [c for c in ct.columns if (isinstance(c, (int, np.integer)) and c >= 3)]
        if high_cols:
            ct['3+'] = ct[high_cols].sum(axis=1)
            ct = ct.drop(columns=high_cols)
        ct = ct.reindex(columns=[0,1,2,'3+'], fill_value=0)
        pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        return pct

    def plot_status_comparison(pct_table, statuses, title, outname, legend_title="Status", label_map=None):
        """
        Plot comparison for given statuses. legend_title is used as the legend title.
        label_map maps internal status keys (index values) to display labels.
        """
        output_dir = "figures/reflection_analysis"
        os.makedirs(output_dir, exist_ok=True)
        categories = [0,1,2,'3+']
        x = np.arange(len(categories))
        n = len(statuses)
        total_width = 0.8
        width = total_width / n if n>0 else total_width

        fig, ax = plt.subplots(figsize=(8,6))
        for i, st in enumerate(statuses):
            vals = pct_table.loc[st].values if st in pct_table.index else np.zeros(len(categories))
            offset = x + (i - (n-1)/2) * width
            # label resolution: use provided map or prettify
            if label_map and st in label_map:
                lbl = label_map[st]
            else:
                lbl = str(st).replace('_', ' ').title()
            bars = ax.bar(offset, vals, width=width, label=lbl, edgecolor='black')
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in categories])
        ax.set_xlabel('Number of Reflections')
        ax.set_ylabel('Percentage of Events (%)')
        # ax.set_title(title)
        ax.legend(title=legend_title, loc='upper right', ncol=1)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        fname = os.path.join(output_dir, outname)
        plt.savefig(fname, dpi=200)
        plt.show()
        plt.close(fig)
        print(f"Saved plot: {fname}")

    # Build percent tables
    pct_by_filter = build_status_percent_table('filter_status')
    pct_by_opening = build_status_percent_table('opening_status')

    # Determine preferred ordering for plotting (try to pick with/without)
    filter_statuses = []
    if 'with_filter' in pct_by_filter.index and 'without_filter' in pct_by_filter.index:
        filter_statuses = ['with_filter', 'without_filter']
    else:
        # fallback: use all non-'unknown' indices
        filter_statuses = [s for s in pct_by_filter.index if 'unknown' not in str(s).lower()][:2]

    opening_statuses = []
    if 'with_opening' in pct_by_opening.index and 'without_opening' in pct_by_opening.index:
        opening_statuses = ['with_opening', 'without_opening']
    else:
        opening_statuses = [s for s in pct_by_opening.index if 'unknown' not in str(s).lower()][:2]

    # Plot comparisons if we have at least two statuses to compare
    if len(filter_statuses) >= 2:
        plot_status_comparison(
            pct_by_filter,
            filter_statuses,
            "Comparison: With vs Without Filter",
            "filter_comparison.pdf",
            legend_title="Filter",
            label_map={"with_filter": "With Filter", "without_filter": "Without Filter"}
        )
    else:
        print("Skipping filter comparison plot: not enough filter-status categories found.")

    if len(opening_statuses) >= 2:
        plot_status_comparison(
            pct_by_opening,
            opening_statuses,
            "Comparison: With vs Without Opening",
            "opening_comparison.pdf",
            legend_title="Opening",
            label_map={"with_opening": "With Opening", "without_opening": "Without Opening"}
        )
    else:
        print("Skipping opening comparison plot: not enough opening-status categories found.")

    # --- 1. Percentile analysis for number of mirror hits ---
    print("\n--- Overall Distribution of Mirror Hits (All Concentrators) ---")
    print(master_df['nMirrorHits'].describe(percentiles=[.25, .5, .75, .9, .99]))

    # --- 2. Percentile analysis for Gold and Aluminium hits ---
    if 'hit1_mat' in master_df.columns:
        print("\n--- Distribution for First Hit on Gold (G4_Au) ---")
        df_gold = master_df[master_df['hit1_mat'] == 'G4_Au']
        if not df_gold.empty:
            print(df_gold['nMirrorHits'].describe(percentiles=[.25, .5, .75, .9, .99]))
        else:
            print("No events with a first hit on Gold found.")

        print("\n--- Distribution for First Hit on Aluminium (G4_Al) ---")
        df_alu = master_df[master_df['hit1_mat'] == 'G4_Al']
        if not df_alu.empty:
            print(df_alu['nMirrorHits'].describe(percentiles=[.25, .5, .75, .9, .99]))
        else:
            print("No events with a first hit on Aluminium found.")

        # --- Percentages of protons that hit Gold vs Aluminium when reaching the concentrator ---
        # Define sets
        total_events = len(master_df)
        reached_mask = master_df['nMirrorHits'] > 0
        total_reached = reached_mask.sum()

        overall_counts = master_df['hit1_mat'].value_counts(dropna=True)
        reached_counts = master_df.loc[reached_mask, 'hit1_mat'].value_counts(dropna=True)

        au_all = overall_counts.get('G4_Au', 0)
        al_all = overall_counts.get('G4_Al', 0)
        au_reached = reached_counts.get('G4_Au', 0)
        al_reached = reached_counts.get('G4_Al', 0)

        print("\n--- Percentages: First hit material (Gold vs Aluminium) ---")
        print(f"Total events: {total_events:,}; events that reached concentrator (nMirrorHits>0): {total_reached:,}")
        print(f"Overall first-hit counts: Gold={au_all:,}, Al={al_all:,} "
              f"({au_all+al_all:,} of {total_events:,} accounted for these two materials)")
        print(f"Overall percentages (of all events):  Gold = {au_all/total_events*100:.2f}%,  Al = {al_all/total_events*100:.2f}%")
        if total_reached:
            print(f"Reached concentrator (first-hit among reached): Gold = {au_reached:,} ({au_reached/total_reached*100:.2f}%), "
                  f"Al = {al_reached:,} ({al_reached/total_reached*100:.2f}%)")
        else:
            print("No events reached the concentrator; cannot compute reached percentages.")

        # --- EXPANDED: Percentages for 1st, 2nd and 3rd hits (Gold vs Aluminium) ---
        for i in (1, 2, 3):
            hit_col = f'hit{i}_mat'
            mask_i = master_df['nMirrorHits'] >= i
            n_with_i = mask_i.sum()
            if n_with_i == 0:
                print(f"\nHit #{i}: No events with >= {i} hits.")
                continue

            counts_i = master_df.loc[mask_i, hit_col].value_counts(dropna=True)
            au_i = counts_i.get('G4_Au', 0)
            al_i = counts_i.get('G4_Al', 0)
            other_i = n_with_i - (au_i + al_i)

            print(f"\nHit #{i} (events with >= {i} hits): n={n_with_i:,}")
            print(f"  Gold (G4_Au): {au_i:,} ({au_i/n_with_i*100:.2f}%)")
            print(f"  Aluminium (G4_Al): {al_i:,} ({al_i/n_with_i*100:.2f}%)")
            if other_i:
                print(f"  Other/unknown: {other_i:,} ({other_i/n_with_i*100:.2f}%)")

            # breakdown by filter and opening for this hit index (optional, concise)
            if 'filter_status' in master_df.columns:
                for filt, grp in master_df[mask_i].groupby('filter_status'):
                    if not filt or 'unknown' in str(filt).lower(): 
                        continue
                    n_grp = len(grp)
                    a = grp[hit_col].value_counts().get('G4_Au', 0)
                    b = grp[hit_col].value_counts().get('G4_Al', 0)
                    print(f"    {filt}: n={n_grp:,}  Gold={a:,} ({a/n_grp*100:.2f}%),  Al={b:,} ({b/n_grp*100:.2f}%)")

        # --- Plot composition per hit index (Gold / Aluminium / Other) for hits 1..3 ---
        def plot_hit_material_composition(df, max_hits=3):
            output_dir = "figures/reflection_analysis"
            os.makedirs(output_dir, exist_ok=True)
            hit_indices = list(range(1, max_hits + 1))
            comps = ['G4_Au', 'G4_Al', 'Other']
            # store percentages
            pct_matrix = np.zeros((len(hit_indices), len(comps)))
            counts_n = []
            for idx, i in enumerate(hit_indices):
                col = f'hit{i}_mat'
                if col not in df.columns:
                    counts_n.append(0)
                    continue
                mask = df['nMirrorHits'] >= i
                n = mask.sum()
                counts_n.append(n)
                if n == 0:
                    continue
                vc = df.loc[mask, col].value_counts(dropna=True)
                au = vc.get('G4_Au', 0)
                al = vc.get('G4_Al', 0)
                other = n - (au + al)
                pct_matrix[idx, 0] = au / n * 100
                pct_matrix[idx, 1] = al / n * 100
                pct_matrix[idx, 2] = max(0.0, other / n * 100)

            # Plot stacked bar chart
            fig, ax = plt.subplots(figsize=(8,6))
            x = np.arange(len(hit_indices))
            colors = ['goldenrod', 'silver', 'lightgray']
            bottoms = np.zeros(len(hit_indices))
            labels = ['Gold', 'Aluminium', 'Other/Unknown']
            for j in range(len(comps)):
                vals = pct_matrix[:, j]
                bars = ax.bar(x, vals, bottom=bottoms, color=colors[j], edgecolor='black', label=labels[j])
                # annotate segment percentages if visible (>2%)
                for xi, v, b in zip(x, vals, bottoms):
                    if v >= 2.0:
                        ax.text(xi, b + v/2, f"{v:.1f}%", ha='center', va='center', fontsize=10)
                bottoms += vals

            # xtick labels include n counts
            xticklabels = [f"Interaction #{i}\n(n={counts_n[idx]:,})" for idx, i in enumerate(hit_indices)]
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            ax.set_ylabel("Percentage of events (%)")
            ax.set_ylim(0, 100)
            # ax.set_xlabel("Hit index (events with >= N hits)")
            # ax.legend(title="Material", ncol=1, loc='upper right')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            plt.tight_layout()
            fname = os.path.join(output_dir, "hit_material_composition_1to3.pdf")
            plt.savefig(fname, dpi=200)
            plt.show()
            plt.close(fig)
            print(f"Saved plot: {fname}")

        # call the new plot function
        plot_hit_material_composition(master_df, max_hits=3)
    # --- 3. Heatmap-style table of reflections per concentrator ---
    print("\n" + "="*80)
    print("Reflection Count 'Heatmap' per Concentrator")
    print("="*80)
    
    heatmap_df = pd.crosstab(master_df['mirror'], master_df['nMirrorHits'])
    
    for i in range(4):
        if i not in heatmap_df.columns: heatmap_df[i] = 0
            
    if len(heatmap_df.columns) > 4:
        cols_to_sum = [c for c in heatmap_df.columns if c >= 3]
        heatmap_df['3+'] = heatmap_df[cols_to_sum].sum(axis=1)
        cols_to_drop = [c for c in cols_to_sum if c != '3+']
        heatmap_df = heatmap_df.drop(columns=cols_to_drop)

    heatmap_df = heatmap_df.reindex(columns=[0, 1, 2, '3+'], fill_value=0)
    
    total_counts = heatmap_df.sum(axis=1)
    percent_df = heatmap_df.div(total_counts, axis=0).fillna(0) * 100
    
    heatmap_df, percent_df = heatmap_df.align(percent_df, join='outer', axis=0)

    display_df = heatmap_df.astype(str) + " (" + percent_df.round(1).astype(str) + "%)"
    display_df['Total'] = total_counts

    print("Showing: Count (% of Total for that Concentrator)")
    print(display_df)

    # --- Average entry energy for events with mirror hits (if available) ---
    if 'entryE' in master_df.columns:
        print("\n--- Average Entry Energy at Small Detector (keV) ---")
        # Overall mean
        overall = master_df['entryE'].dropna()
        if not overall.empty:
            print(f"All protons: {overall.mean():.2f} keV (n={len(overall)})")

        # With / without mirror hits
        with_hits = master_df[master_df['nMirrorHits'] > 0]['entryE'].dropna()
        without_hits = master_df[master_df['nMirrorHits'] == 0]['entryE'].dropna()

        if not with_hits.empty:
            print(f"Protons with Mirror Hits:    {with_hits.mean():.2f} keV (n={len(with_hits)})")
        else:
            print("Protons with Mirror Hits:    No data")

        if not without_hits.empty:
            print(f"Protons without Mirror Hits: {without_hits.mean():.2f} keV (n={len(without_hits)})")
        else:
            print("Protons without Mirror Hits: No data")

        # Per-concentrator means for events that had mirror hits
        per_mirror = master_df[master_df['nMirrorHits'] > 0].groupby('mirror')['entryE']
        if not per_mirror.size().empty:
            print("\nPer concentrator (events with mirror hits):")
            for mirror, series in per_mirror:
                series = series.dropna()
                if series.empty:
                    print(f"{mirror}: No data")
                else:
                    print(f"{mirror}: {series.mean():.2f} keV (n={len(series)})")

    # --- Plot reflection-count distribution (normalized percentage) ---
    def plot_reflection_distribution(percent_df):
        """
        Save a grouped bar plot showing percentage of events (per concentrator)
        vs number of reflections (0,1,2,3+).
        """
        output_dir = "figures/reflection_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Ensure columns in desired order and types
        categories = [0, 1, 2, '3+']
        plot_df = percent_df.copy()
        plot_df = plot_df.reindex(columns=categories, fill_value=0)

        concentrators = plot_df.index.tolist()
        x = np.arange(len(categories))
        n_conc = len(concentrators)
        total_width = 0.8
        width = total_width / n_conc if n_conc > 0 else total_width

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use the COLOR_MIRROR mapping for bar colors
        for i, conc in enumerate(concentrators):
            offset = x + (i - (n_conc - 1) / 2) * width
            vals = plot_df.loc[conc].values
            color = COLOR_MIRROR.get(conc, None)
            bars = ax.bar(offset, vals, width=width, label=conc, color=color, edgecolor='black')
            # label bars with percent values
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in categories])
        ax.set_xlabel('Number of Reflections')
        ax.set_ylabel('Percentage of Events (%)')
        ax.legend(title='Concentrator')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()

        fname = os.path.join(output_dir, "reflection_count_distribution.pdf")
        plt.savefig(fname, dpi=200)
        plt.close(fig)
        plt.show()
        print(f"Saved plot: {fname}")

    # Call the reflection-distribution plot
    plot_reflection_distribution(percent_df)

    # --- 4. --- Show example event trajectories ---
    if 'hit1_vol' in master_df.columns:
        print("\n" + "="*80)
        print("Example Event Trajectories")
        print("="*80)
        # Filter for events with at least one hit
        hit_df = master_df[master_df['nMirrorHits'] > 0].dropna(subset=['hit1_vol'])
        for _, row in hit_df.head(10).iterrows():
            path = str(row['hit1_vol'])
            if row['nMirrorHits'] > 1 and pd.notna(row.get('hit2_vol')):
                path += f" -> {row['hit2_vol']}"
            if row['nMirrorHits'] > 2 and pd.notna(row.get('hit3_vol')):
                path += f" -> {row['hit3_vol']}"
            print(f"Path: {path}")

    # --- Show example reflection coordinates ---
    if 'refl1_x' in master_df.columns:
        print("\n" + "="*80)
        print("Example Reflection Coordinates")
        print("="*80)
        # Filter for rows where reflection data exists
        xyz_df = master_df.dropna(subset=['refl1_x'])
        # Use .head() to avoid printing thousands of lines
        for _, row in xyz_df.head(20).iterrows():
            coords = []
            # Ensure nMirrorHits is an integer for the range function
            num_hits = int(row.get('nMirrorHits', 0))
            for i in range(1, num_hits + 1):
                if f'refl{i}_x' in row and pd.notna(row[f'refl{i}_x']):
                    coords.append(f"({row[f'refl{i}_x']:.1f}, {row[f'refl{i}_y']:.1f}, {row[f'refl{i}_z']:.1f})")
            if coords: # Only print if there are coordinates to show
                print(f"Event with {num_hits} hits: " + " -> ".join(coords))

    # --- 5. --- Count hits on individual gold and aluminium mirrors ---
    if 'hit1_vol' in master_df.columns:
        
        # --- Helper function to extract LV number for sorting ---
        def get_lv_number(segment_name):
            """Extracts the LV number from a segment name string for sorting."""
            match = re.search(r'_(\d+)', str(segment_name))
            return int(match.group(1)) if match else -1

        # --- Helper function to print summary tables ---
        def print_material_summary(title, hit_counts_dict):
            print("\n" + "="*80)
            print(title)
            print("="*80)
            
            if not hit_counts_dict:
                print("No hits recorded for this material.")
                return

            for concentrator, counts in sorted(hit_counts_dict.items()):
                print(f"\n--- Concentrator: {concentrator} ---")
                if not counts:
                    print("No hits for this type.")
                    continue

                # --- Sort by LV number ---
                sorted_counts = sorted(counts.items(), key=lambda item: get_lv_number(item[0]))
                total_hits = sum(count for _, count in sorted_counts)

                print(f"{'Mirror shell':<45} | {'Hit Count':>12} | {'Percentage':>12}")
                print("-" * 75)
                for segment, count in sorted_counts:
                    percentage = (count / total_hits) * 100 if total_hits > 0 else 0
                    print(f"{segment:<45} | {count:>12,} | {percentage:>11.2f}%")

        # --- Plotting function for normalized gold mirror hits ---
        def plot_gold_hit_summary(hit_counts_dict):
            """Generates and saves bar plots for normalized gold mirror hits, split by primary/secondary."""
            output_dir = "figures/reflection_analysis"
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n--- Generating Gold Hit Distribution Plots (saved to {output_dir}) ---")

            # --- Helper function to plot on a subplot axis ---
            def plot_subplot(ax, data, total_hits, title):
                if not data:
                    ax.text(0.5, 0.5, 'No Hits Recorded', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title)
                    return

                sorted_data = sorted(data.items(), key=lambda item: get_lv_number(item[0]))
                segment_numbers = [get_lv_number(item[0]) for item in sorted_data]
                hit_values = [item[1] for item in sorted_data]
                
                # Normalize hits based on the total for the entire concentrator
                normalized_hits = [h / total_hits for h in hit_values]

                ax.bar(range(len(segment_numbers)), normalized_hits, color='goldenrod', edgecolor='black')
                # --- Remove hardcoded styles to use rcParams ---
                ax.set_title(title)
                ax.set_ylabel("Normalized Hit Count")
                ax.set_xticks(range(len(segment_numbers)))
                ax.set_xticklabels(segment_numbers)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)

            for concentrator, counts in sorted(hit_counts_dict.items()):
                if not counts:
                    print(f"Skipping plot for {concentrator}: No gold hits recorded.")
                    continue

                total_gold_hits = sum(counts.values())
                if total_gold_hits == 0:
                    continue

                primary_counts = {k: v for k, v in counts.items() if 'Primary' in str(k)}
                
                if concentrator == 'SP':
                    # --- SP: Plot only primary mirror on a single subplot ---
                    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
                    # fig.suptitle(f"Normalized Gold Mirror Hit Distribution for {concentrator}")
                    
                    plot_subplot(ax, primary_counts, total_gold_hits, "Primary Mirror")
                    ax.set_xlabel("Mirror Shell Number ")
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                else:
                    # --- Other concentrators: Plot both primary and secondary ---
                    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
                    # fig.suptitle(f"Normalized Gold Mirror Hit Distribution for {concentrator}")

                    secondary_counts = {k: v for k, v in counts.items() if 'Secondary' in str(k)}
                    
                    plot_subplot(axes[0], primary_counts, total_gold_hits, "Primary Mirror")
                    plot_subplot(axes[1], secondary_counts, total_gold_hits, "Secondary Mirror")
                    
                    axes[1].set_xlabel("Mirror Shell Number ")
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
                
                # Save the figure
                fname = os.path.join(output_dir, f"gold_hits_{concentrator}.pdf")
                plt.savefig(fname)
                print(f"Saved plot: {fname}")
                plt.close(fig)

        # --- Plotting function for normalized aluminium mirror hits ---
        def plot_alu_hit_summary(hit_counts_dict):
            """Generates and saves bar plots for normalized aluminium mirror hits, split by primary/secondary."""
            output_dir = "figures/reflection_analysis"
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n--- Generating Aluminium Hit Distribution Plots (saved to {output_dir}) ---")

            # Helper function to plot on a subplot axis
            def plot_subplot(ax, data, total_hits, title):
                if not data:
                    ax.text(0.5, 0.5, 'No Hits Recorded', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title)
                    return

                sorted_data = sorted(data.items(), key=lambda item: get_lv_number(item[0]))
                segment_numbers = [get_lv_number(item[0]) for item in sorted_data]
                hit_values = [item[1] for item in sorted_data]
                
                normalized_hits = [h / total_hits for h in hit_values]

                ax.bar(range(len(segment_numbers)), normalized_hits, color='silver', edgecolor='black')
                # --- Remove hardcoded styles to use rcParams ---
                ax.set_title(title)
                ax.set_ylabel("Normalized Hit Count")
                ax.set_xticks(range(len(segment_numbers)))
                ax.set_xticklabels(segment_numbers)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_axisbelow(True)

            for concentrator, counts in sorted(hit_counts_dict.items()):
                if not counts:
                    print(f"Skipping plot for {concentrator}: No aluminium hits recorded.")
                    continue

                total_alu_hits = sum(counts.values())
                if total_alu_hits == 0:
                    continue

                primary_counts = {k: v for k, v in counts.items() if 'Primary' in str(k)}

                if concentrator == 'SP':
                    # --- SP: Plot only primary mirror on a single subplot ---
                    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
                    # fig.suptitle(f"Normalized Aluminium Mirror Hit Distribution for {concentrator}")

                    plot_subplot(ax, primary_counts, total_alu_hits, "Primary Mirror")
                    ax.set_xlabel("Mirror Shell Number ")

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                else:
                    # --- Other concentrators: Plot both primary and secondary ---
                    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
                    # fig.suptitle(f"Normalized Aluminium Mirror Hit Distribution for {concentrator}")

                    secondary_counts = {k: v for k, v in counts.items() if 'Secondary' in str(k)}

                    plot_subplot(axes[0], primary_counts, total_alu_hits, "Primary Mirror")
                    plot_subplot(axes[1], secondary_counts, total_alu_hits, "Secondary Mirror")
                    
                    axes[1].set_xlabel("Mirror Shell Number ")
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                fname = os.path.join(output_dir, f"alu_hits_{concentrator}.pdf")
                plt.savefig(fname)
                print(f"Saved plot: {fname}")
                plt.close(fig)

        # Use dictionaries to store counts: {concentrator: {mirror_segment: count}}
        gold_mirror_hit_counts = {}
        alu_mirror_hit_counts = {}

        # Iterate through each event that had at least one hit
        for _, row in master_df[master_df['nMirrorHits'] > 0].iterrows():
            concentrator = row['mirror']
            # Initialize dictionaries for the concentrator if not present
            if concentrator not in gold_mirror_hit_counts:
                gold_mirror_hit_counts[concentrator] = {}
            if concentrator not in alu_mirror_hit_counts:
                alu_mirror_hit_counts[concentrator] = {}

            # Check each potential hit in the event
            for i in range(1, int(row['nMirrorHits']) + 1):
                vol_col = f'hit{i}_vol'
                mat_col = f'hit{i}_mat'
                
                if vol_col in row and mat_col in row and pd.notna(row[vol_col]):
                    segment_name = row[vol_col]
                    material = row[mat_col]
                    
                    if material == 'G4_Au':
                        gold_mirror_hit_counts[concentrator][segment_name] = gold_mirror_hit_counts[concentrator].get(segment_name, 0) + 1
                    elif material == 'G4_Al':
                        alu_mirror_hit_counts[concentrator][segment_name] = alu_mirror_hit_counts[concentrator].get(segment_name, 0) + 1

        # Now, process the dictionaries to print the summaries
        print_material_summary("Summary of Hits on Individual Gold Mirrors", gold_mirror_hit_counts)
        print_material_summary("Summary of Hits on Individual Aluminium Mirrors", alu_mirror_hit_counts)

        # --- Call the plotting function for the gold mirror hits ---
        plot_gold_hit_summary(gold_mirror_hit_counts)
        # --- Call the plotting function for the aluminium mirror hits ---
        plot_alu_hit_summary(alu_mirror_hit_counts)

    print("\n" + "="*80)


if __name__ == "__main__":
    # --- Remove argparse and use fixed directories ---
    # This aligns the script with the behavior of calculate_flux.py
    dirs_to_scan = [
        "/home/frisoe/Desktop/Root/withhole/",
        "/home/frisoe/Desktop/Root/withouthole/",
    ]
    
    analyze_reflection_materials(dirs_to_scan)