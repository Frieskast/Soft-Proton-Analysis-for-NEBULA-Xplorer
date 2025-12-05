import uproot
import pandas as pd
import glob
import os
import argparse
import numpy as np
import awkward as ak 

def analyze_reflection_materials(file_pattern):
    """
    Reads ROOT files, extracts mirror hit data from the SmallDetSummary
    ntuple using the new unified format, and prints a summary.
    """
    root_files = glob.glob(file_pattern)
    if not root_files:
        print(f"Error: No files found matching pattern '{file_pattern}'")
        return

    print(f"Found {len(root_files)} files to analyze...")

    all_data = []
    ntuple_name = "SmallDetSummary"
    
    desired_columns = [
        "EventID", "entryE", "nMirrorHits",
        "hit1_mat", "hit1_vol",
        "hit2_mat", "hit2_vol",
        "hit3_mat", "hit3_vol"
    ]

    for fpath in root_files:
        try:
            with uproot.open(fpath) as f:
                if ntuple_name not in f:
                    print(f"  - Skipping {os.path.basename(fpath)} (no '{ntuple_name}' ntuple)")
                    continue
                
                tree = f[ntuple_name]
                
                columns_to_read = [col for col in desired_columns if col in tree.keys()]
                
                if "nMirrorHits" not in columns_to_read:
                    print(f"  - Skipping {os.path.basename(fpath)} (missing 'nMirrorHits', likely old format)")
                    continue

                df = tree.arrays(columns_to_read, library="pd")
                all_data.append(df)
                print(f"  - Processed {os.path.basename(fpath)}, found {len(df)} summary events.")

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")

    if not all_data:
        print("\nNo data could be extracted from any files.")
        return

    master_df = pd.concat(all_data, ignore_index=True)

    # Convert Awkward-backed columns to standard pandas Series
    string_cols = [col for col in master_df.columns if 'mat' in col or 'vol' in col]
    for col in string_cols:
        master_df[col] = pd.Series(master_df[col].to_list())


    print("\n" + "="*50)
    print("Unified Mirror Hit Analysis Summary")
    print("="*50)
    print(f"Total summary events analyzed: {len(master_df)}")

    # --- Analysis using the new unified columns ---

    # 1. Count events by number of mirror hits
    print("\n--- Events by Number of Mirror Hits ---")
    if 'nMirrorHits' in master_df.columns:
        print(master_df['nMirrorHits'].value_counts().sort_index())
    else:
        print("No 'nMirrorHits' data to analyze.")

    # 2. Analyze the material and volume of the FIRST hit
    print("\n--- Details of First Hit ---")
    hit1_df = master_df[master_df['nMirrorHits'] > 0]
    if not hit1_df.empty:
        print("Materials:")
        print(hit1_df['hit1_mat'].value_counts())
        print("\nVolumes:")
        print(hit1_df['hit1_vol'].value_counts().head(10))
    else:
        print("No events with at least one mirror hit were found.")

    # 3. --- Analyze details of the SECOND hit ---
    print("\n--- Details of Second Hit ---")
    hit2_df = master_df[master_df['nMirrorHits'] > 1]
    if not hit2_df.empty:
        print("Materials:")
        print(hit2_df['hit2_mat'].value_counts())
        print("\nVolumes:")
        print(hit2_df['hit2_vol'].value_counts().head(10))
    else:
        print("No events with at least two mirror hits were found.")
        
    # 4. --- Analyze details of the THIRD hit ---
    print("\n--- Details of Third Hit ---")
    hit3_df = master_df[master_df['nMirrorHits'] > 2]
    if not hit3_df.empty and 'hit3_mat' in hit3_df.columns:
        print("Materials:")
        print(hit3_df['hit3_mat'].value_counts())
        print("\nVolumes:")
        print(hit3_df['hit3_vol'].value_counts().head(10))
    else:
        print("No events with at least three mirror hits were found.")

    # 5. Compare kinetic energy for different event types
    print("\n--- Average Entry Energy at Small Detector (keV) ---")
    if not hit1_df.empty:
        avg_ekin_hit = hit1_df['entryE'].mean()
        print(f"Protons with Mirror Hits: {avg_ekin_hit:.2f} keV")
    
    direct_hit_df = master_df[master_df['nMirrorHits'] == 0]
    if not direct_hit_df.empty:
        avg_ekin_direct = direct_hit_df['entryE'].mean()
        print(f"Direct Hit Protons (no mirror interaction): {avg_ekin_direct:.2f} keV")

    # 6. --- Show example event trajectories ---
    print("\n--- Example Event Trajectories ---")
    if not hit1_df.empty:
        # Take up to 10 example events
        for index, row in hit1_df.head(10).iterrows():
            path = str(row['hit1_vol'])
            if row['nMirrorHits'] > 1 and pd.notna(row['hit2_vol']):
                path += f" -> {row['hit2_vol']}"
            if row['nMirrorHits'] > 2 and pd.notna(row['hit3_vol']):
                path += f" -> {row['hit3_vol']}"
            
            print(f"Event {int(row['EventID'])} (Hits: {int(row['nMirrorHits'])}): {path}")
    else:
        print("No events with mirror hits to display.")

    print("\n" + "="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze unified mirror hit data from Geant4 output ROOT files.")
    parser.add_argument(
        "file_pattern", 
        type=str,
        nargs='?', 
        default="/home/frisoe/Desktop/Root/withhole/output_DCC_filterOn_1000keV_option4_N1m.root", 
        help="Pattern to find ROOT files (defaults to /home/frisoe/Desktop/Root/withhole/*.root)"
    )
    args = parser.parse_args()
    
    analyze_reflection_materials(args.file_pattern)