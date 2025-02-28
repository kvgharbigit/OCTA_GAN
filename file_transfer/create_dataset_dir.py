import os
import shutil
from pathlib import Path
import glob

# Define source and destination directories
source_dirs = [
    "/Volumes/Ophthalmic neuroscience/Projects/Control Database 2024/AIBL_coreg",
    "/Volumes/Ophthalmic neuroscience/Projects/Control Database 2024/HealthyBrain_coreg"
]
destination_dir = "/Volumes/Ophthalmic neuroscience/Projects/Control Database 2024/Kayvans_Model_Dataset"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Initialize counters
total_patients = 0
patients_with_both_files = 0
patients_missing_files = []
hsi_file_types = {'C1': 0, 'D1': 0, 'other_h5': 0}

# Process each source directory
for source_dir in source_dirs:
    print(f"Processing source directory: {source_dir}")

    # Get all patient subdirectories
    try:
        patient_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    except FileNotFoundError:
        print(f"Source directory not found: {source_dir}")
        continue

    print(f"Found {len(patient_dirs)} patient directories")
    total_patients += len(patient_dirs)

    # Process each patient directory
    for patient_dir in patient_dirs:
        source_patient_path = os.path.join(source_dir, patient_dir)
        dest_patient_path = os.path.join(destination_dir, patient_dir)

        # Create patient directory in destination
        os.makedirs(dest_patient_path, exist_ok=True)

        # Find the specific files with priority order for HSI files
        hsi_file = None
        hsi_type = None
        octa_file = None

        # First look for C1 files
        c1_files = glob.glob(os.path.join(source_patient_path, "*C1*.h5"))
        if c1_files:
            hsi_file = os.path.basename(c1_files[0])
            hsi_type = 'C1'
        else:
            # If no C1 files, look for D1 files
            d1_files = glob.glob(os.path.join(source_patient_path, "*D1*.h5"))
            if d1_files:
                hsi_file = os.path.basename(d1_files[0])
                hsi_type = 'D1'
            else:
                # If no C1 or D1 files, take any h5 file
                any_h5_files = glob.glob(os.path.join(source_patient_path, "*.h5"))
                if any_h5_files:
                    hsi_file = os.path.basename(any_h5_files[0])
                    hsi_type = 'other_h5'

        # Look for OCTA file
        octa_files = glob.glob(os.path.join(source_patient_path, "*RetinaAngiographyEnface*.tiff"))
        if octa_files:
            octa_file = os.path.basename(octa_files[0])

        # Copy the files if found
        has_both_files = True

        if hsi_file:
            shutil.copy2(
                os.path.join(source_patient_path, hsi_file),
                os.path.join(dest_patient_path, hsi_file)
            )
            hsi_file_types[hsi_type] += 1
            print(f"  Copied HSI file ({hsi_type}) for patient {patient_dir}")
        else:
            print(f"  WARNING: No HSI file found for patient {patient_dir}")
            has_both_files = False

        if octa_file:
            shutil.copy2(
                os.path.join(source_patient_path, octa_file),
                os.path.join(dest_patient_path, octa_file)
            )
            print(f"  Copied OCTA file for patient {patient_dir}")
        else:
            print(f"  WARNING: No OCTA file found for patient {patient_dir}")
            has_both_files = False

        if has_both_files:
            patients_with_both_files += 1
        else:
            patients_missing_files.append(patient_dir)

# Print summary
print("\nSummary:")
print(f"Total patient directories processed: {total_patients}")
print(f"Patients with both HSI and OCTA files: {patients_with_both_files}")
print(f"Patients missing one or both files: {len(patients_missing_files)}")

print("\nHSI file types copied:")
print(f"  C1 files: {hsi_file_types['C1']}")
print(f"  D1 files: {hsi_file_types['D1']}")
print(f"  Other h5 files: {hsi_file_types['other_h5']}")

if patients_missing_files:
    print("\nPatients missing files:")
    for patient in patients_missing_files:
        print(f"  {patient}")

print(f"\nAll files have been copied to: {destination_dir}")