import os
import shutil

# Define the root directory
root_dir = "/Volumes/Ophthalmic neuroscience/Projects/Control Database 2024"

# Define the relevant folders
folders = {
    "AIBL": "AIBL_coreg",
    "Healthy Brain": "Healthy Brain_coreg"
}


# Function to extract patient ID, eye, and date from folder name
def parse_folder_name(folder, is_coreg):
    parts = folder.split("_")
    if is_coreg:
        patient_id = parts[0][:-1]  # Remove last character (R or L)
        eye = "OD" if parts[0][-1] == "R" else "OS"
        date = parts[1]  # Already in YYMMDD format
    else:
        patient_id = parts[0]
        eye = parts[1]  # OD or OS
        date = parts[2][2:]  # Convert YYYYMMDD to YYMMDD
    return patient_id, eye, date


# Function to find and copy the RetinaAngiographyEnface TIFF file
def copy_retina_tiff():
    copied_count = 0
    missing_count = 0
    total_count = 0

    for source_folder, target_folder in folders.items():
        source_path = os.path.join(root_dir, source_folder)
        target_path = os.path.join(root_dir, target_folder)

        if not os.path.exists(source_path) or not os.path.exists(target_path):
            print(f"Skipping {source_folder} as one of the directories does not exist.")
            continue

        target_patients = os.listdir(target_path)
        total_patients = len(target_patients)
        print(f"Processing {total_patients} patients in {target_folder}...")

        for index, target_patient in enumerate(target_patients, start=1):
            print(f"Processing patient {index}/{total_patients} in {target_folder}...")
            target_patient_path = os.path.join(target_path, target_patient)
            if not os.path.isdir(target_patient_path):
                continue

            patient_id, eye, date = parse_folder_name(target_patient, is_coreg=True)
            source_patient_folder = f"{patient_id}_{eye}_{'20' + date}"  # Convert back to YYYYMMDD
            source_patient_path = os.path.join(source_path, source_patient_folder)

            if not os.path.exists(source_patient_path):
                print(f"[MISSING] Source folder not found for: {target_patient}")
                missing_count += 1
                continue

            # Find the RetinaAngiographyEnface TIFF file
            found = False
            for file in os.listdir(source_patient_path):
                if "RetinaAngiographyEnface" in file and file.endswith(".tiff"):
                    source_file = os.path.join(source_patient_path, file)
                    target_file = os.path.join(target_patient_path, file)
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    found = True
                    print(f"[COPIED] {file} -> {target_patient_path}")
                    break

            if not found:
                print(f"[MISSING] No RetinaAngiographyEnface TIFF file found for {target_patient}")
                missing_count += 1

            total_count += 1

    print("\n--- Summary ---")
    print(f"Total patients processed: {total_count}")
    print(f"Total files copied: {copied_count}")
    print(f"Total missing files: {missing_count}")


# Run the function
copy_retina_tiff()