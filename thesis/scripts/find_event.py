import uproot
import numpy as np
import sys
import os

# --- Hardcode the path to your ROOT file here ---
# Use the same file you plan to visualize.
root_file_path = "/home/frisoe/Desktop/Root/output_DCC_filterOn_1000keV_option4_N10k.root" # <-- EDIT THIS LINE

if not os.path.exists(root_file_path):
    print(f"Error: File not found at {root_file_path}")
    sys.exit(1)

try:
    with uproot.open(root_file_path) as f:
        # The SmallDetSummary ntuple only contains events that hit the detector.
        summary_tree = f["SmallDetSummary"]
        data = summary_tree.arrays(["EventID", "refl_mat"], library="np")

        # Create a boolean mask for rows where the reflection material is "G4_Au".
        # The actual material name is "G4_Au", not "Gold".
        gold_reflections_mask = np.char.find(data["refl_mat"].astype(str), "G4_Au") != -1
        
        # Get the EventIDs that satisfy the mask.
        gold_event_ids = data["EventID"][gold_reflections_mask]

        if gold_event_ids.size > 0:
            # Get the first event that matches.
            event_to_find = gold_event_ids[0]
            print(f"Found an event that hit the detector after reflecting on Gold.")
            print(f"EventID: {event_to_find}")
            print("\n---")
            print("To visualize this event in Geant4, use the following commands:")
            print(f"/run/skipEvents {event_to_find - 1}")
            print("/run/beamOn 1")
        else:
            print("No events found in this file that both hit the detector and reflected on Gold.")

except Exception as e:
    print(f"An error occurred: {e}")