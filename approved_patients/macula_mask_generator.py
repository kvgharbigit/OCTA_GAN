#!/usr/bin/env python3
"""
Script to create masked versions of RetinaEnface TIFF files by applying
a macula mask based on HSI data.

The mask is created by finding all spatial pixels where the average value
across all wavelengths is <0.000001 (these pixels are 0 in the mask, all others are 1).
"""

import os
import csv
import h5py
import numpy as np
import tifffile
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time

# Paths
BASE_DIR = r"Z:\Projects\Ophthalmic neuroscience\Projects\Control Database 2024\Kayvan_experiments\kayvan_octa_macula_updated"
# Use local path for the CSV files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "approved_participants_macula.csv")
NEW_CSV_PATH = os.path.join(SCRIPT_DIR, "approved_participants_macula_masked.csv")

def generate_mask_from_hsi(hsi_path, threshold=0.000001, save_mask=True):
    """
    Generate a binary mask from HSI data where pixels with average value
    across all wavelengths < threshold are 0, all others are 1.
    
    Args:
        hsi_path: Path to the HSI .h5 file
        threshold: Threshold value for masking
        save_mask: Whether to save the mask as an image for visualization
        
    Returns:
        Binary mask as a 2D numpy array
    """
    try:
        print(f"Processing HSI file: {hsi_path}")
        with h5py.File(hsi_path, 'r') as f:
            # Determine the dataset keys
            keys = list(f.keys())
            print(f"Available keys in H5 file: {keys}")
            
            # Special handling for 'Cube' key which is common in these files
            if 'Cube' in keys:
                # Get attributes and properties of the Cube
                print("Exploring Cube structure...")
                cube_obj = f['Cube']
                if isinstance(cube_obj, h5py.Dataset):
                    print(f"Cube is a dataset with shape {cube_obj.shape}")
                    # Check if shape is available and reasonable
                    if len(cube_obj.shape) >= 2:
                        hsi_data = cube_obj[:]
                    else:
                        print(f"Cube has unexpected shape: {cube_obj.shape}")
                        return None
                else:
                    # Explore inside the cube group
                    print("Cube is a group, attempting to find data inside it")
                    # Look for the largest dataset in the file
                    biggest_dataset = None
                    biggest_size = 0
                    
                    def find_largest_dataset(name, obj):
                        nonlocal biggest_dataset, biggest_size
                        if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                            size = np.prod(obj.shape)
                            if size > biggest_size:
                                biggest_size = size
                                biggest_dataset = obj
                    
                    # Visit all items in the file
                    f.visititems(find_largest_dataset)
                    
                    if biggest_dataset is not None:
                        print(f"Using largest dataset found with shape {biggest_dataset.shape}")
                        hsi_data = biggest_dataset[:]
                    else:
                        print("Could not find suitable dataset in the file")
                        return None
            else:
                # No 'Cube' key found, try to find any suitable dataset
                data_key = None
                max_ndim = 0
                max_size = 0
                
                # Find the dataset with highest dimensionality
                for key in keys:
                    if hasattr(f[key], 'shape'):
                        shape = f[key].shape
                        ndim = len(shape)
                        size = np.prod(shape)
                        
                        # Prefer higher dimensionality, then larger size
                        if (ndim > max_ndim) or (ndim == max_ndim and size > max_size):
                            max_ndim = ndim
                            max_size = size
                            data_key = key
                
                if data_key:
                    print(f"Using best dataset found: {data_key} with shape {f[data_key].shape}")
                    hsi_data = f[data_key][:]
                else:
                    print("No suitable dataset found in the file")
                    return None
            
            print(f"HSI data shape: {hsi_data.shape}")
            
            # Handle different data shapes
            if len(hsi_data.shape) == 2:
                # Already a 2D array, use as is
                print("Data is already 2D, using directly")
                mean_data = hsi_data
            elif len(hsi_data.shape) == 3:
                # 3D array, need to determine which dimension is spectral
                # Heuristic: smallest dimension is likely spectral
                smallest_dim = np.argmin(hsi_data.shape)
                print(f"Using dimension {smallest_dim} as spectral dimension")
                
                # Average across the spectral dimension
                mean_data = np.mean(hsi_data, axis=smallest_dim)
                
                # Ensure result is 2D
                if len(mean_data.shape) != 2:
                    print(f"Warning: After averaging, result shape is {mean_data.shape}")
                    if len(mean_data.shape) == 3 and mean_data.shape[0] == 1:
                        mean_data = mean_data[0]  # Extract the single slice
                    elif len(mean_data.shape) == 1:
                        # 1D result, cannot create mask
                        print("Cannot create 2D mask from 1D data")
                        return None
            else:
                print(f"Cannot handle data with shape {hsi_data.shape}")
                return None
            
            print(f"Mean data shape: {mean_data.shape}")
            print(f"Min value: {np.min(mean_data)}, Max value: {np.max(mean_data)}")
            
            # Create the binary mask (1 where mean >= threshold, 0 otherwise)
            mask = (mean_data >= threshold).astype(np.uint8)
            print(f"Mask shape: {mask.shape}, Sum of mask: {np.sum(mask)}")
            print(f"Percentage of non-zero pixels: {np.sum(mask) / mask.size * 100:.2f}%")
            
            # Save the mask for visualization
            if save_mask:
                mask_dir = os.path.dirname(hsi_path)
                mask_filename = os.path.basename(hsi_path).replace('.h5', '_mask.png')
                mask_path = os.path.join(mask_dir, mask_filename)
                
                plt.figure(figsize=(10, 10))
                plt.imshow(mask, cmap='gray')
                plt.title(f"Macula Mask (threshold={threshold})")
                plt.colorbar()
                plt.savefig(mask_path)
                plt.close()
                print(f"Saved mask visualization to: {mask_path}")
            
            return mask
    except Exception as e:
        print(f"Error processing {hsi_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def apply_mask_to_tiff(tiff_path, mask, output_path):
    """
    Apply a binary mask to a TIFF file and save the result.
    
    Args:
        tiff_path: Path to the input TIFF file
        mask: Binary mask as a 2D numpy array
        output_path: Path to save the masked TIFF file
    """
    try:
        print(f"Applying mask to TIFF file: {tiff_path}")
        
        # Read the TIFF file
        img = tifffile.imread(tiff_path)
        print(f"TIFF image shape: {img.shape}, dtype: {img.dtype}")
        
        # Ensure the mask has the same dimensions as the TIFF
        if img.shape != mask.shape:
            print(f"Resizing mask from {mask.shape} to {img.shape}")
            
            # Import resize function
            from skimage.transform import resize
            
            # Handle different dimensionality cases
            if len(img.shape) == 2 and len(mask.shape) == 2:
                # Both are 2D
                mask_resized = resize(mask, img.shape, order=0, preserve_range=True).astype(np.uint8)
            elif len(img.shape) == 3 and len(mask.shape) == 2:
                # TIFF is 3D (e.g., multi-channel or multi-frame) but mask is 2D
                # First resize the mask to match the spatial dimensions
                if img.shape[0] < 5:  # Likely channels-first format
                    mask_resized = resize(mask, img.shape[1:], order=0, preserve_range=True).astype(np.uint8)
                    # Repeat the mask for each channel
                    mask_resized = np.repeat(mask_resized[np.newaxis, :, :], img.shape[0], axis=0)
                else:  # Likely channels-last format or multi-frame
                    mask_resized = resize(mask, img.shape[:2], order=0, preserve_range=True).astype(np.uint8)
                    # Add third dimension if needed
                    if img.shape[2] > 1:
                        mask_resized = np.repeat(mask_resized[:, :, np.newaxis], img.shape[2], axis=2)
            else:
                # For any other case, try to intelligently resize
                print(f"Complex reshape needed from {mask.shape} to match {img.shape}")
                # Start with 2D resize
                if len(mask.shape) == 2:
                    # Determine which dimensions to use for resizing
                    if len(img.shape) == 3:
                        if img.shape[0] < 5:  # Likely channels-first
                            target_shape = img.shape[1:]
                        else:  # Likely height, width, channels
                            target_shape = img.shape[:2]
                    else:
                        target_shape = img.shape
                    
                    # Resize mask to target shape
                    mask_2d = resize(mask, target_shape, order=0, preserve_range=True).astype(np.uint8)
                    
                    # Expand dimensions to match TIFF if needed
                    if len(img.shape) == 3:
                        if img.shape[0] < 5:  # Channels-first
                            mask_resized = np.repeat(mask_2d[np.newaxis, :, :], img.shape[0], axis=0)
                        else:  # Channels-last
                            mask_resized = np.repeat(mask_2d[:, :, np.newaxis], img.shape[2], axis=2)
                    else:
                        mask_resized = mask_2d
                else:
                    # If mask is already 3D, try to match dimensions more directly
                    # This is a more complex case and may need custom handling
                    raise ValueError(f"Cannot automatically resize mask with shape {mask.shape} to match TIFF with shape {img.shape}")
            
            print(f"Resized mask shape: {mask_resized.shape}")
        else:
            mask_resized = mask
        
        # Apply the mask
        masked_img = img * mask_resized
        print(f"Masked image shape: {masked_img.shape}, dtype: {masked_img.dtype}")
        
        # Save the masked image
        tifffile.imwrite(output_path, masked_img)
        print(f"Saved masked TIFF to: {output_path}")
        
        # Save a side-by-side comparison for visual verification
        comparison_path = output_path.replace('.tiff', '_comparison.png')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(mask_resized, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        axes[2].imshow(masked_img, cmap='gray')
        axes[2].set_title('Masked Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
        print(f"Saved comparison image to: {comparison_path}")
        
        return True
    except Exception as e:
        print(f"Error applying mask to {tiff_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_csv():
    """
    Process the CSV file to create masked versions of all RetinaEnface TIFF files.
    Also generates a new CSV file that is identical to the original but with masked TIFF paths.
    """
    # Read the CSV file
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        # Keep the exact same fieldnames - no new columns
        rows = list(reader)
    
    # Process each row
    print(f"Processing {len(rows)} entries from CSV...")
    
    # Create a copy of rows for the new CSV with masked paths
    new_rows = []
    
    for row in tqdm(rows):
        new_row = row.copy()  # Create a copy of the original row
        
        hsi_path = row['hs_file']
        octa_path = row['octa_file']
        
        if not hsi_path or not octa_path:
            print(f"Warning: Missing HSI or OCTA path for ID {row.get('id', 'unknown')}")
            new_rows.append(new_row)  # Add unchanged row to new CSV
            continue
        
        # Generate the output path for the masked TIFF
        octa_dir = os.path.dirname(octa_path)
        octa_filename = os.path.basename(octa_path)
        octa_basename, octa_ext = os.path.splitext(octa_filename)
        masked_filename = f"{octa_basename}_masked{octa_ext}"
        masked_path = os.path.join(octa_dir, masked_filename)
        
        # Generate the path for the mask image (for internal use)
        mask_filename = os.path.basename(hsi_path).replace('.h5', '_mask.png')
        mask_path = os.path.join(os.path.dirname(hsi_path), mask_filename)
        
        # Generate the mask from HSI data
        print(f"\nProcessing entry {row['id_full']}:")
        mask = generate_mask_from_hsi(hsi_path)
        
        if mask is not None:
            # Apply the mask to the TIFF file
            success = apply_mask_to_tiff(octa_path, mask, masked_path)
            if success:
                # Replace the original OCTA path with the masked path ONLY in the new CSV
                new_row['octa_file'] = masked_path
                print(f"Updated CSV entry to use masked file: {masked_path}")
            else:
                print(f"Failed to create masked file for {octa_path}")
        
        # Add the modified row to our new list
        new_rows.append(new_row)
        
        # Add a small delay to allow for viewing console output
        time.sleep(0.5)
    
    # Write the new CSV with masked TIFF paths - ensure proper encoding and line endings
    with open(NEW_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    print(f"Created new CSV file: {NEW_CSV_PATH}")

def process_single_sample(hsi_path, tiff_path, threshold=0.0001):
    """
    Process a single sample for testing purposes.
    
    Args:
        hsi_path: Path to the HSI .h5 file
        tiff_path: Path to the TIFF file
        threshold: Threshold value for masking
    """
    print(f"Processing single sample:")
    print(f"HSI path: {hsi_path}")
    print(f"TIFF path: {tiff_path}")
    
    # Generate the output path for the masked TIFF
    tiff_dir = os.path.dirname(tiff_path)
    tiff_filename = os.path.basename(tiff_path)
    tiff_basename, tiff_ext = os.path.splitext(tiff_filename)
    masked_filename = f"{tiff_basename}_masked{tiff_ext}"
    masked_path = os.path.join(tiff_dir, masked_filename)
    
    # Generate the mask
    mask = generate_mask_from_hsi(hsi_path, threshold=threshold)
    
    if mask is not None:
        # Apply the mask to the TIFF file
        success = apply_mask_to_tiff(tiff_path, mask, masked_path)
        if success:
            print(f"Successfully created masked file: {masked_path}")
        else:
            print(f"Failed to create masked file for {tiff_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate masked versions of RetinaEnface TIFF files.')
    parser.add_argument('--single', action='store_true', help='Process a single sample for testing')
    parser.add_argument('--hsi', type=str, help='Path to a single HSI file (required if --single is used)')
    parser.add_argument('--tiff', type=str, help='Path to a single TIFF file (required if --single is used)')
    parser.add_argument('--threshold', type=float, default=0.000001, help='Threshold value for masking (default: 0.000001)')
    
    args = parser.parse_args()
    
    print("Starting macula mask generation...")
    
    if args.single:
        if not args.hsi or not args.tiff:
            print("Error: --hsi and --tiff arguments are required with --single")
            return
        process_single_sample(args.hsi, args.tiff, args.threshold)
    else:
        process_csv()
    
    print("Done!")

if __name__ == "__main__":
    main()