#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:16:48 2024

@author: harrysmith
"""

import os
import pandas as pd

def main():
    # Base directory path (adjust based on the mounted path)
    base_path = "/Volumes/Ortho/COPDGeneScoli/CT Scans/Extracted/"

    # Initialize data storage
    results = []

    # Step 1: List all parent folders in the base directory
    parent_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for parent in parent_folders:
        parent_path = os.path.join(base_path, parent)

        # Step 2: List "COPD1" subfolders
        subfolders = [f for f in os.listdir(parent_path) if f == "COPD1"]

        for subfolder in subfolders:
            subfolder_path = os.path.join(parent_path, subfolder)

            # Step 3: List sub-subfolders inside "COPD1"
            sub_subfolders = [f for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]

            for sub_subfolder in sub_subfolders:
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)

                # Step 4: Look for "dicom" directory
                dicom_path = os.path.join(sub_subfolder_path, "dicom")
                if not os.path.isdir(dicom_path):
                    continue

                # Step 5: List .dcm files
                dcm_files = [f for f in os.listdir(dicom_path) if f.endswith(".dcm")]

                for dcm in dcm_files:
                    results.append([parent, subfolder, sub_subfolder, os.path.join(dicom_path, dcm)])

    # Save results to a CSV
    df = pd.DataFrame(results, columns=["Parent Folder", "COPD1 Subfolder", "Sub-Subfolder", "DICOM File"])
    output_file = "dcm_file_list_simple.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
