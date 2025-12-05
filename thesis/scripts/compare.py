import os, glob, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot

def parse_metadata_from_name(fname):
    base = os.path.basename(fname)
    path = fname
    m_mirror = re.search(r"output_([^_]+)_", base)
    mirror = m_mirror.group(1) if m_mirror else ""
    m_filter = re.search(r"_(filterOn|filterOff)_", base)
    filter_flag = m_filter.group(1) if m_filter else ""
    m_energy = re.search(r"_([0-9]+)keV_", base)
    energy = int(m_energy.group(1)) if m_energy else None
    m_N = re.search(r"_N(\d+)([kKmM])?", base)
    N = None
    if m_N:
        n = int(m_N.group(1))
        suf = (m_N.group(2) or "").lower()
        if suf == 'k': n *= 1000
        if suf == 'm': n *= 1000000
        N = n
    
    hole_status = "withHole" if "withhole" in path.lower() else "withoutHole"
    
    return mirror, filter_flag, energy, N, hole_status

def count_hits(rootfile):
    # --- Only use SmallDetSummary for hits ---
    # return number of unique events that are recorded in the SmallDetSummary ntuple.
    target_tree_name = "SmallDetSummary"
    try:
        # Find the actual key in the file, which might include a cycle number (e.g., "SmallDetSummary;1")
        key_to_use = None
        for k in rootfile.keys():
            if k.startswith(target_tree_name):
                key_to_use = k
                break
        
        if key_to_use:
            arr = rootfile[key_to_use].arrays(["EventID"], library="np")
            evts = np.asarray(arr["EventID"]).astype(int)
            return len(np.unique(evts))
            
    except Exception as e:
        # This can happen if the tree exists but is empty or corrupt.
        # print(f"Could not count hits: {e}") # Optional: for debugging
        return 0
    
    # Return 0 if the target ntuple was not found
    return 0

def collect_entries(dirs):
    entries = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "output_*.root"), recursive=True)):
            try:
                rf = uproot.open(f)
            except Exception:
                continue
            mirror, filt, energy, N, hole_status = parse_metadata_from_name(f)
            hits = count_hits(rf)
            eff = None
            if N and N>0:
                eff = hits / float(N)
            entries.append({
                "path": f, "mirror": mirror, "filter": filt,
                "energy": energy, "N": N, "hits": hits, "eff": eff,
                "hole": hole_status
            })
    return entries

def plot_filter_effect(entries, out=None):
    # group by (mirror,energy) and compare filterOn/Off and with/without Hole
    groups = {}
    for e in entries:
        key = (e["mirror"], e["energy"])
        group_key = f'{e["hole"]}_{e["filter"]}'
        if key not in groups:
            groups[key] = {}
        groups[key][group_key] = e["eff"] if e["eff"] is not None else 0.0

    labels = []
    val_wh_on = []
    val_wh_off = []
    val_woh_on = []
    val_woh_off = []

    for (mirror, energy), v in sorted(groups.items()):
        labels.append(f"{mirror}\n{energy}keV")
        val_wh_on.append(v.get("withHole_filterOn", 0.0))
        val_wh_off.append(v.get("withHole_filterOff", 0.0))
        val_woh_on.append(v.get("withoutHole_filterOn", 0.0))
        val_woh_off.append(v.get("withoutHole_filterOff", 0.0))

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.8), 7))
    ax.bar(x - 1.5*width, val_woh_off, width, label='withoutHole_filterOff', color='C0', alpha=0.6)
    ax.bar(x - 0.5*width, val_woh_on, width, label='withoutHole_filterOn', color='C0')
    ax.bar(x + 0.5*width, val_wh_off, width, label='withHole_filterOff', color='C1', alpha=0.6)
    ax.bar(x + 1.5*width, val_wh_on, width, label='withHole_filterOn', color='C1')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Efficiency (hits / N)")
    ax.set_title("Effect of Optical Filter and Hole")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if out:
        plt.savefig(out)
    else:
        plt.show()

def plot_config_comparison(entries, energy, filter_flag, out=None):
    # compare mirrors at same energy and filter, showing with/without hole
    filtered = [e for e in entries if e["energy"]==energy and e["filter"]==filter_flag]
    if not filtered:
        print(f"No entries for energy {energy} and filter {filter_flag}")
        return
    
    groups = {}
    for e in filtered:
        key = e["mirror"]
        if key not in groups:
            groups[key] = {}
        groups[key][e["hole"]] = e["eff"] if e["eff"] is not None else 0.0

    labels = []
    with_hole_vals = []
    without_hole_vals = []
    for mirror, v in sorted(groups.items()):
        labels.append(mirror)
        with_hole_vals.append(v.get("withHole", 0.0))
        without_hole_vals.append(v.get("withoutHole", 0.0))

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.6), 6))
    ax.bar(x - width/2, without_hole_vals, width, label='withoutHole')
    ax.bar(x + width/2, with_hole_vals, width, label='withHole')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Efficiency (hits / N)")
    ax.set_title(f"Comparison of Optical Configs at {energy} keV ({filter_flag})")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if out:
        plt.savefig(out)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs","-d", nargs="+", default=[
        "/home/frisoe/Desktop/Thesis/build/build/root/withhole/",
        "/home/frisoe/Desktop/Thesis/build/build/root/withouthole/",
    ], help="Base directories containing simulation results.")
    parser.add_argument("--out-filter", default=None, help="output filename for filter effect plot")
    parser.add_argument("--out-config", default=None, help="output filename for config comparison plot")
    parser.add_argument("--energy", type=int, default=1000, help="energy (keV) for config comparison")
    parser.add_argument("--filter", dest="filter_flag", choices=["filterOn","filterOff"], default="filterOn")
    args = parser.parse_args()

    entries = collect_entries(args.dirs)
    if not entries:
        print("No ROOT files found in the specified directories. Exiting.")
        exit()
        
    plot_filter_effect(entries, out=args.out_filter)
    plot_config_comparison(entries, energy=args.energy, filter_flag=args.filter_flag, out=args.out_config)