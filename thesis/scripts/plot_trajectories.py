import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <-- Import the 3D toolkit
import pandas as pd
import sys
import os
from scipy.spatial.transform import Rotation as R

def plot_3d_trajectory(ax, target_event, detector_df, mirror_geo_df):
    """
    Plots the mirror geometry and trajectory in a true 3D view.
    """
    event_id = target_event['EventID']
    num_hits = int(target_event['nMirrorHits'])
        
    detector_hit = detector_df[detector_df['EventID'] == event_id]
    if detector_hit.empty:
        return
    detector_pos = {'x': detector_hit['x'].iloc[0], 'y': detector_hit['y'].iloc[0], 'z': detector_hit['z'].iloc[0]}

    # --- Collect all 3D reflection points ---
    reflection_points = []
    hit_volume_names = set()
    for i in range(1, num_hits + 1):
        # Define the column names for the current hit
        x_col, y_col, z_col, vol_col = f'hit{i}_x', f'hit{i}_y', f'hit{i}_z', f'hit{i}_vol'
        
        if not pd.isna(target_event[x_col]):
            # --- FIX: Use the defined vol_col variable ---
            vol_name = target_event.get(vol_col, 'N/A')
            
            # Unscramble y and z coordinates to match the Geant4 rotation
            point = {'x': target_event[x_col], 'y': target_event[z_col], 'z': target_event[y_col], 'vol': vol_name}
            reflection_points.append(point)
            hit_volume_names.add(vol_name)

    print(f"\n--- Plotting Event: {int(event_id)} ---")
    print("Hit Volumes:", sorted(list(hit_volume_names)))

    # --- Plot the hit mirrors as 3D surfaces ---
    if mirror_geo_df is not None:
        hit_mirrors_df = mirror_geo_df[mirror_geo_df['name'].isin(hit_volume_names)]
        for _, row in hit_mirrors_df.iterrows():
            z_profile = np.array([float(p) for p in row['z_points'].split()])
            r_profile = np.array([float(p) for p in row['r_points'].split()])
            
            # Create a meshgrid for the surface of revolution
            theta = np.linspace(0, 2 * np.pi, 30)
            Z, THETA = np.meshgrid(z_profile, theta)
            # --- Rename R from meshgrid to avoid conflict ---
            R_mesh, _ = np.meshgrid(r_profile, theta)
            
            # Convert cylindrical coordinates to Cartesian for plotting
            # --- Use the renamed variable ---
            X = R_mesh * np.cos(THETA)
            Y = R_mesh * np.sin(THETA)
            
            # --- Apply the inverse rotation (+90 degrees around Y) ---
            # Create a rotation object for +90 degrees around the Y-axis
            rot = R.from_euler('y', 90, degrees=True)
            
            # Stack X, Y, and Z into a single array of points
            points = np.stack([X, Y, Z], axis=0)
            
            # --- Transpose the points array for rot.apply ---
            # The shape needs to be (num_points, 3), not (3, num_points)
            flat_points = points.reshape((3, -1)).T
            
            # Rotate the points
            rotated_flat_points = rot.apply(flat_points)
            
            # Reshape the rotated points back to the original meshgrid shape
            rotated_points = rotated_flat_points.T.reshape((3, X.shape[0], X.shape[1]))
            
            # Unpack the rotated points
            X_rotated, Y_rotated, Z_rotated = rotated_points[0], rotated_points[1], rotated_points[2]
            
            # Plot the mirror as a wireframe surface
            ax.plot_surface(X_rotated, Y_rotated, Z_rotated, color='orange', alpha=0.3, rstride=1, cstride=1, linewidth=0.5, edgecolors='k')

    # --- Plot the 3D trajectory ---
    source_pos = None
    if reflection_points:
        # Source is at z=901, aligned with the first hit's x and y
        source_pos = {'x': reflection_points[0]['x'], 'y': reflection_points[0]['y'], 'z': 901}

    full_path_points = [p for p in [source_pos] + reflection_points + [detector_pos] if p]
    
    if len(full_path_points) > 1:
        xs = [p['x'] for p in full_path_points]
        ys = [p['y'] for p in full_path_points]
        zs = [p['z'] for p in full_path_points]
        
        # Plot the trajectory line
        ax.plot(xs, ys, zs, marker='o', color='blue', label=f'Event {int(event_id)} Trajectory')
        
        # Plot special points
        if source_pos:
            ax.scatter(source_pos['x'], source_pos['y'], source_pos['z'], c='green', s=50, marker='s', label='Source')
        ax.scatter(detector_pos['x'], detector_pos['y'], detector_pos['z'], c='red', s=50, marker='X', label='Detector Hit')

    # --- Finalize 3D Plot ---
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"3D Trajectory for Event {int(event_id)}")
    
    # Improve axis scaling to be more representative
    max_range = np.ptp(np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]))
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.legend()


def main():
    # --- Hardcode paths ---
    root_file_path = "/home/frisoe/Desktop/Root/withhole/output_DCC_filterOn_1000keV_option4_N1m.root"
    geo_file_path = "/home/frisoe/Desktop/Thesis/build/mirror_geometry.csv"

    if not os.path.exists(root_file_path):
        print(f"Error: ROOT file not found at {root_file_path}")
        sys.exit(1)

    mirror_geo_df = None
    if os.path.exists(geo_file_path):
        try:
            mirror_geo_df = pd.read_csv(geo_file_path)
            print(f"Successfully loaded mirror geometry from {geo_file_path}")
        except Exception as e:
            print(f"Error reading geometry file {geo_file_path}: {e}")
    else:
        print(f"Warning: Geometry file not found at {geo_file_path}. Mirrors will not be plotted.")

    try:
        with uproot.open(root_file_path) as f:
            summary_tree = f["SmallDetSummary"]
            summary_cols = [
                "EventID", "nMirrorHits",
                "hit1_x", "hit1_y", "hit1_z", "hit1_mat", "hit1_vol",
                "hit2_x", "hit2_y", "hit2_z", "hit2_mat", "hit2_vol",
                "hit3_x", "hit3_y", "hit3_z", "hit3_mat", "hit3_vol",
            ]
            existing_summary_cols = [c for c in summary_cols if c in summary_tree]
            summary_df = pd.DataFrame(summary_tree.arrays(existing_summary_cols, library="np"))
            
            for col in summary_df.select_dtypes(include=[object]):
                summary_df[col] = summary_df[col].apply(
                    lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x
                )

            detector_tree = f["SmallDet"]
            detector_cols = ["x", "y", "z", "EventID"]
            existing_detector_cols = [c for c in detector_cols if c in detector_tree]
            detector_df = pd.DataFrame(detector_tree.arrays(existing_detector_cols, library="np"))
            
            detector_df.drop_duplicates(subset='EventID', keep='first', inplace=True)

    except Exception as e:
        print(f"Error reading data from ROOT file: {e}")
        sys.exit(1)

    condition = (
        (summary_df['nMirrorHits'] == 2) &
        (summary_df['hit1_mat'] == 'G4_Au') &
        (summary_df['hit2_mat'] == 'G4_Au')
    )
    target_events = summary_df[condition]

    if target_events.empty:
        print("Could not find any event that matches the criteria (2 reflections on Gold).")
        return

    event_to_plot = target_events.head(1).iloc[0]
    print(f"Found {len(target_events)} matching events. Plotting the first one in 3D.")

    # --- Create a 3D figure and axes ---
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_trajectory(ax, event_to_plot, detector_df, mirror_geo_df)
    
    plt.show()

if __name__ == "__main__":
    main()