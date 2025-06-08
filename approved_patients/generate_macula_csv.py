import os
import csv
import re
from datetime import datetime

def generate_macula_csv(input_dir, output_file):
    """
    Generate a CSV file similar to approved_participants_global.csv but for macula data
    
    Args:
        input_dir: Path to directory containing macula data
        output_file: Path to output CSV file
    """
    base_dir = input_dir
    rows = []
    
    # Add header row
    header = ["id", "id_full", "id_date_adjust", "eye", "source", "hs_file", "rgb_file", 
              "label_file", "octa_file", "diabetes", "fovea_center", "dim", "project"]
    rows.append(header)
    
    # Walk through the directory structure
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Skip if not a directory or doesn't match our expected naming pattern (ID_EYE)
        if not os.path.isdir(folder_path):
            continue
            
        # Extract ID and eye from folder name (e.g., "12415_L")
        match = re.match(r'(\d+)_([LR])', folder_name)
        if not match:
            continue
            
        patient_id, eye = match.groups()
        
        # Find files in the folder
        files = os.listdir(folder_path)
        
        # Look for date information in filenames (expecting format like 12415L_240426)
        date_pattern = re.compile(rf'{patient_id}{eye}_(\d{{6}})')
        date_match = None
        
        for file in files:
            match = date_pattern.search(file)
            if match:
                date_match = match.group(1)
                break
                
        if not date_match:
            continue
            
        # Format dates correctly
        id_full = f"{patient_id}{eye}_{date_match}"
        
        # Convert date format from YYMMDD to YYYYMMDD
        year = int(date_match[:2])
        month = date_match[2:4]
        day = date_match[4:6]
        
        # Assuming 20xx for the year
        full_year = f"20{year}"
        id_date_adjust = f"{patient_id}{eye}_{full_year}{month}{day}"
        
        # Find specific files
        hs_file = ""
        rgb_file = ""
        label_file = ""
        octa_file = ""
        
        for file in files:
            if file.endswith("_C1.h5"):
                hs_file = os.path.join(folder_path, file)
            elif file.endswith("_C1.tif"):
                rgb_file = os.path.join(folder_path, file)
            elif file.endswith(".tiff") and not file.endswith("_RetinaAngiographyEnfacetif.tiff"):
                label_file = os.path.join(folder_path, file)
            elif file.endswith("_RetinaAngiographyEnfacetif.tiff"):
                octa_file = os.path.join(folder_path, file)
        
        # Create a row for this patient
        row = [
            patient_id,                      # id
            id_full,                         # id_full
            id_date_adjust,                  # id_date_adjust
            eye,                             # eye
            base_dir + "\\",                 # source
            hs_file,                         # hs_file
            rgb_file,                        # rgb_file
            label_file,                      # label_file
            octa_file,                       # octa_file
            "0",                             # diabetes (default to 0)
            "",                              # fovea_center (left empty)
            "800",                           # dim (default to 800)
            "MACULA"                         # project
        ]
        
        rows.append(row)
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    print(f"Generated CSV file with {len(rows)-1} entries at {output_file}")

if __name__ == "__main__":
    # Replace with your actual paths
    input_directory = r"Z:\Projects\Ophthalmic neuroscience\Projects\Control Database 2024\Kayvan_experiments\kayvan_octa_macula_updated"
    output_csv = r"C:\Users\kvgha\AI_Projects\OCTA_GAN\approved_patients\approved_participants_macula.csv"
    
    generate_macula_csv(input_directory, output_csv)