import os
import sys
import numpy as np
import uproot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from collections import defaultdict

# --- user params ---
root_file = sys.argv[1] if len(sys.argv) > 1 else "build/build/root/output_DPH_filterOn_1000keV_N500.root"
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figures_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 1.0,
})

def load_tree(tree, cols):
    try:
        arrs = tree.arrays(cols, library="np")
    except Exception:
        return None
    return arrs

def group_indices_by_event_preserve_order(event_ids):
    # return list of index-arrays grouped by event id, preserving original row order
    unique_ids = np.unique(event_ids)
    groups = [np.nonzero(event_ids == uid)[0] for uid in unique_ids]
    return groups

# open file
f = uproot.open(root_file)
print("Opened:", root_file)
keys = list(f.keys())
print("ROOT keys:", keys)

# ---------------------------------------------------------------------
# SmallDet: per-step entries (x,y,z,Ekin,EventID,trajectory,refl_*)
small_cols = ["x", "y", "z", "Ekin", "EventID",
              "trajectory", "refl_x", "refl_y", "refl_z", "refl_mat"]
have_small = False
for k in ("SmallDet;1", "SmallDet"):
    if k in keys:
        tname = k
        have_small = True
        break

if have_small:
    tree = f[tname]
    # try to load extended set; fall back to minimal set if some columns missing
    data = load_tree(tree, small_cols)
    if data is None:
        cols = ["x", "y", "z", "Ekin", "EventID"]
        data = load_tree(tree, cols)
        if data is None:
            print("SmallDet tree exists but could not load columns", small_cols)
            data = None

    if data is not None:
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        z = np.asarray(data["z"])
        Ekin = np.asarray(data["Ekin"])     # stored in keV
        evt = np.asarray(data["EventID"]).astype(int)
        print("SmallDet entries:", x.size, "unique events:", np.unique(evt).size)

        # optional refl columns
        refl_x = np.asarray(data["refl_x"]) if "refl_x" in data.keys() else None
        refl_y = np.asarray(data["refl_y"]) if "refl_y" in data.keys() else None
        refl_z = np.asarray(data["refl_z"]) if "refl_z" in data.keys() else None
        refl_mat = np.asarray(data["refl_mat"]) if "refl_mat" in data.keys() else None

        groups = group_indices_by_event_preserve_order(evt)

        # Compute per-event entry/exit/loss (prefer SmallDetSummary if available)
        loss_per_event = []
        entryE_list = []
        exitE_list = []
        event_ids = []

        # if per-event summary exists, prefer it
        summary_exists = False
        for key in ("SmallDetSummary;1", "SmallDetSummary"):
            if key in keys:
                summary_exists = True
                stree = f[key]
                break
        if summary_exists:
            try:
                scols = ["EventID", "entryE", "exitE", "lossE"]
                sdata = stree.arrays(scols, library="np")
                s_evt = np.asarray(sdata["EventID"]).astype(int)
                entryE_list = np.asarray(sdata["entryE"])
                exitE_list = np.asarray(sdata["exitE"])
                loss_per_event = np.asarray(sdata["lossE"])
                event_ids = s_evt
                print("Loaded SmallDetSummary with", event_ids.size, "rows")
            except Exception:
                # fallback to computing from per-step
                summary_exists = False

        if not summary_exists:
            # compute from per-step groups (use first step as entry, last as exit)
            for inds in groups:
                if inds.size == 0:
                    continue
                entryE = float(Ekin[inds[0]])
                exitE = float(Ekin[inds[-1]])
                loss = entryE - exitE
                loss_per_event.append(loss)
                entryE_list.append(entryE)
                exitE_list.append(exitE)
                event_ids.append(int(evt[inds[0]]))
            loss_per_event = np.array(loss_per_event)
            entryE_list = np.array(entryE_list)
            exitE_list = np.array(exitE_list)
            event_ids = np.array(event_ids)
            print("Computed per-event loss for", loss_per_event.size, "events from per-step data")

        # 2D trajectories (Y vs Z) colored by accumulated loss along the track (first N events)
        fig, ax = plt.subplots(figsize=(9,7))
        max_traj = 200
        plotted = 0

        norm = colors.Normalize(vmin=0.0, vmax=max(1.0, np.nanpercentile(loss_per_event, 95)))
        cmap = cm.get_cmap("inferno")

        for gi, inds in enumerate(groups):
            if plotted >= max_traj:
                break
            if inds.size < 2:
                continue
            ys = y[inds]
            zs = z[inds]
            e = Ekin[inds]  # keV
            evtid = int(evt[inds[0]])

            # determine entry energy for this event (from computed lists)
            idx = np.where(event_ids == evtid)[0]
            if idx.size:
                entryE = float(entryE_list[idx[0]])
            else:
                entryE = float(e[0])

            # loss along track at each step
            loss_along = entryE - e  # keV

            # build line segments in YZ plane
            points = np.column_stack([ys, zs])
            segs = np.stack([points[:-1], points[1:]], axis=1)
            # color for each segment: average loss of the two endpoints
            seg_vals = 0.5 * (loss_along[:-1] + loss_along[1:])

            lc = LineCollection(segs, array=seg_vals, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.9)
            ax.add_collection(lc)

            # optionally mark start and end
            ax.scatter(ys[0], zs[0], color="green", s=8, marker="o", label="entry" if plotted==0 else "")
            ax.scatter(ys[-1], zs[-1], color="blue", s=8, marker="x", label="exit" if plotted==0 else "")

            # reflection markers if present
            if refl_x is not None and refl_x.size == x.size:
                rx = refl_x[inds]
                valid = np.isfinite(rx)
                if np.any(valid):
                    rys = refl_y[inds][valid]
                    rzs = refl_z[inds][valid]
                    ax.scatter(rys, rzs, color="red", s=30, marker='o', edgecolor='k', label="reflection" if plotted==0 else "")

            plotted += 1

        # autoscale and colorbar
        ax.autoscale_view()
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # for colorbar
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("accumulated loss (keV) since entry")
        ax.set_xlabel("y (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(f"SmallDet proton tracks (YZ, first {plotted} events) colored by accumulated loss")
        ax.legend(loc="upper right", fontsize=9)
        out = os.path.join(figures_dir, "smallDet_trajectories_yz_loss_along.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)

        # Energy loss distribution plots
        # Histogram of losses
        plt.figure(figsize=(6,4))
        plt.hist(loss_per_event, bins=80, color="tab:blue", alpha=0.8)
        plt.xlabel("Energy loss in SmallDet (keV)")
        plt.ylabel("Events")
        plt.title("Energy loss distribution (SmallDet)")
        out = os.path.join(figures_dir, "smallDet_loss_hist.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)

        # Scatter: loss vs entry energy
        plt.figure(figsize=(6,4))
        plt.scatter(entryE_list, loss_per_event, s=12, alpha=0.7, c='tab:orange', edgecolor='none')
        plt.xlabel("Entry kinetic energy (keV)")
        plt.ylabel("Energy loss (keV)")
        plt.title("SmallDet: loss vs entry energy")
        out = os.path.join(figures_dir, "smallDet_loss_vs_entryE.png")
        # plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        print("Saved:", out)

    else:
        print("SmallDet tree exists but could not load columns", small_cols)
else:
    print("SmallDet tree not found in ROOT file.")

# ---------------------------------------------------------------------
# FocalDet plotting (simple 3D trajectories, no reflection handling here)
if "FocalDet;1" in keys or "FocalDet" in keys:
    tname = "FocalDet;1" if "FocalDet;1" in keys else "FocalDet"
    tree = f[tname]
    cols = ["x", "y", "z", "Ekin", "EventID"]
    data = load_tree(tree, cols)
    if data is not None:
        fx = np.asarray(data["x"])
        fy = np.asarray(data["y"])
        fz = np.asarray(data["z"])
        fe = np.asarray(data["Ekin"])
        fevt = np.asarray(data["EventID"]).astype(int)
        print("FocalDet entries:", fx.size, "unique events:", np.unique(fevt).size)

        groups = group_indices_by_event_preserve_order(fevt)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        max_traj = 200
        plotted = 0
        for inds in groups:
            if plotted >= max_traj:
                break
            if inds.size < 2:
                continue
            ax.plot(fx[inds], fy[inds], fz[inds], color="teal", alpha=0.6)
            plotted += 1

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.set_title(f"Focal plane 3D trajectories (first {plotted} events)")
        out = os.path.join(figures_dir, "focalDet_trajectories_3d.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved:", out)
    else:
        print("FocalDet tree exists but could not load columns", cols)
else:
    print("FocalDet tree not found in ROOT file.")

print("Done.")