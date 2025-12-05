#!/usr/bin/env python3
import os, re, glob, shutil, subprocess
from collections import defaultdict

# --- Only singlescattering source (change if your path differs) ---
src_dir = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/uncombined"
out_dir = "/home/frisoe/Desktop/exp/build/build/root/Singlescattering_final/combined"
os.makedirs(out_dir, exist_ok=True)

# tolerant pattern: captures angle, energy, N (N may have k/m) and ignores any trailing _... before .root
pat = re.compile(r'^output_([0-9.+-]+)deg_([0-9]+)keV_N([0-9\.]+[kKmM]?)(?:_[^.]*)?\.root$', re.IGNORECASE)

def parse_N(tok):
    tok = tok.strip().lower()
    if tok.endswith('k'):
        try:
            return int(float(tok[:-1]) * 1_000)
        except:
            pass
    if tok.endswith('m'):
        try:
            v = float(tok[:-1])
            # user convention: 17m -> 1.7e6
            if v >= 10 and float(v).is_integer():
                return int((v / 10.0) * 1_000_000)
            return int(v * 1_000_000)
        except:
            pass
    try:
        return int(float(tok))
    except:
        return 0

def fmt_N(n):
    if n >= 1_000_000:
        v = n / 1_000_000.0
        s = f"{v:.1f}m"
        return s.replace(".0m","m")
    if n >= 1000:
        return f"{n//1000}k"
    return str(n)

hadd_path = shutil.which("hadd")
if hadd_path is None:
    print("Warning: 'hadd' not found in PATH. Install ROOT or ensure hadd is available.")
    # script will still copy single files

files = glob.glob(os.path.join(src_dir, "output_*.root"))
print("Found", len(files), "files in", src_dir)
groups = defaultdict(list)

for fn in files:
    b = os.path.basename(fn)
    m = pat.match(b)
    if not m:
        print("Skipping (pattern mismatch):", b)
        continue
    angle = round(float(m.group(1)), 2)   # group by angle (rounded to 2 decimals)
    energy = int(m.group(2))              # group by energy (keV)
    ntok = m.group(3)
    n = parse_N(ntok)
    # group key is only (energy, angle) -> ignores N and any run suffix
    groups[(energy, angle)].append((fn, n))

for (energy, angle), fl in sorted(groups.items()):
    total = sum(n for _, n in fl)
    outname = os.path.join(out_dir, f"output_{angle:.2f}deg_{energy}keV_N{fmt_N(total)}_combined.root")
    print(f"Group: energy={energy} keV angle={angle} deg files={len(fl)} -> {os.path.basename(outname)}")
    if len(fl) == 1:
        src = fl[0][0]
        print("  Single file -> copying")
        shutil.copy2(src, outname)
    else:
        if hadd_path is None:
            print("  Skipping merge (hadd unavailable).")
            continue
        cmd = [hadd_path, "-f", outname] + [fn for fn,_ in fl]
        print("  Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("  hadd failed:", e)

print("Done.")