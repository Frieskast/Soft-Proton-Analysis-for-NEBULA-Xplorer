import uproot
import numpy as np
import glob
import re
import os
from collections import defaultdict

file_pat = re.compile(r'output_([0-9.]+)deg_([0-9]+)keV_N([0-9\.kKmM]+)')

def parse_file_info(fname):
    b = os.path.basename(fname)
    m = file_pat.search(b)
    if not m:
        return None
    angle = float(m.group(1))
    energy = int(m.group(2))
    nEvents_token = m.group(3)
    nEvents = parse_nEvents_token(nEvents_token)
    return angle, energy, nEvents

def parse_nEvents_token(tok):
    tok = tok.strip().lower()
    try:
        if tok.endswith('k'):
            return int(float(tok[:-1]) * 1_000)
        if tok.endswith('m'):
            return int(float(tok[:-1]) * 1_000_000)
        return int(float(tok))
    except Exception:
        digits = re.findall(r'\d+', tok)
        return int(digits[0]) if digits else 0

def format_nEvents(n):
    if n >= 1_000_000:
        v = n / 1_000_000.0
        s = f"{v:.1f}m"
        if s.endswith(".0m"):
            s = s.replace(".0m", "m")
        return s
    if n >= 1_000:
        v = n // 1000
        return f"{v}k"
    return str(n)

# SOURCE directory (adjust if needed)
root_dir = "build/build/root/Singlescattering_final"
root_files = glob.glob(os.path.join(root_dir, "output_*.root"))

# OUTPUT directory
out_dir = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/combined"
os.makedirs(out_dir, exist_ok=True)

groups = defaultdict(list)
for fname in root_files:
    info = parse_file_info(fname)
    if info is None:
        print(f"Skipping (no match): {fname}")
        continue
    angle, energy, nEvents = info
    groups[(energy, angle)].append((fname, nEvents))

if not groups:
    print("No groups found. Check root_dir and filename pattern.")
    raise SystemExit(0)

for (energy, angle), files in groups.items():
    if len(files) < 2:
        print(f"Only one file for energy={energy} keV angle={angle} deg; skipping combine.")
        continue

    print(f"\nCombining {len(files)} files for energy={energy} keV, angle={angle} deg")
    hist_acc = {}
    tree_acc = defaultdict(lambda: defaultdict(list))
    total_nEvents = 0

    for fname, nEv in files:
        print(f"  Reading {fname}  (N token -> {nEv})")
        total_nEvents += nEv
        with uproot.open(fname) as f:
            keys = f.keys(recursive=True)
            if not keys:
                print(f"    Warning: file has no keys: {fname}")
            any_data_in_file = False
            for key in keys:
                try:
                    obj = f[key]
                except Exception as exc:
                    print(f"    Could not open key {key} in {fname}: {exc}")
                    continue
                cname = getattr(obj, "classname", "")
                # TH1-like
                if "TH1" in cname and hasattr(obj, "to_numpy"):
                    try:
                        counts, edges = obj.to_numpy()
                    except Exception as exc:
                        print(f"    Warning: cannot to_numpy {key} in {fname}: {exc}")
                        continue
                    counts = np.asarray(counts, dtype=float)
                    s = counts.sum()
                    print(f"    Hist: {key} : total_counts={s:.1f} bins={counts.size}")
                    if s <= 0.0:
                        # skip empty hist
                        continue
                    any_data_in_file = True
                    if key not in hist_acc:
                        hist_acc[key] = (counts.copy(), edges.copy())
                    else:
                        prev_counts, prev_edges = hist_acc[key]
                        if not np.allclose(prev_edges, edges):
                            print(f"    Warning: bin edges mismatch for {key} in {fname}; skipping this histogram")
                            continue
                        hist_acc[key] = (prev_counts + counts, prev_edges)
                # TTree-like
                elif "TTree" in cname or "TNtuple" in cname or cname.startswith("TTree"):
                    try:
                        arrays = obj.arrays(library="np")
                    except Exception as exc:
                        print(f"    Warning: cannot read tree {key} in {fname}: {exc}")
                        continue
                    # check entry count
                    first_branch = next(iter(arrays), None)
                    n_entries = arrays[first_branch].shape[0] if first_branch is not None else 0
                    print(f"    Tree: {key} : entries={n_entries} branches={len(arrays)}")
                    if n_entries == 0:
                        continue
                    any_data_in_file = True
                    for branch, arr in arrays.items():
                        tree_acc[key][branch].append(np.asarray(arr))
                else:
                    # other objects (ignore)
                    continue
            if not any_data_in_file:
                print(f"    Note: no non-empty hist/tree found in {fname}")

    # After reading all files for this group, verify we have something
    if not hist_acc and not tree_acc:
        print(f"  No data accumulated for energy={energy}keV angle={angle}deg -> skipping write")
        continue

    # prepare output trees
    out_trees = {}
    for treename, branches in tree_acc.items():
        out_br = {}
        for branch, arr_list in branches.items():
            try:
                out_br[branch] = np.concatenate(arr_list, axis=0)
            except Exception:
                out_br[branch] = np.hstack([np.atleast_1d(a) for a in arr_list])
        out_trees[treename] = out_br
        print(f"  Prepared tree {treename} with entries = {next(iter(out_br.values())).shape[0] if out_br else 0}")

    formatted_N = format_nEvents(total_nEvents)
    outname = os.path.join(out_dir, f"output_{angle:.2f}deg_{energy}keV_N{formatted_N}_combined.root")

    with uproot.recreate(outname) as fout:
        for hname, (counts, edges) in hist_acc.items():
            # avoid negative or nan counts
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            fout[hname] = (counts, edges)
            print(f"  Wrote histogram {hname} total={counts.sum():.1f}")
        for treename, brdict in out_trees.items():
            fout[treename] = brdict
            print(f"  Wrote tree {treename} entries={next(iter(brdict.values())).shape[0] if brdict else 0}")

    print(f"  -> Wrote {outname}")

print("\nDone.")